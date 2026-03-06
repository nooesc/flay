// Refusal direction extraction for Qwen3.5.
//
// Runs contrastive harmful/harmless prompt pairs through the model,
// captures activations at TokenMixerOut, and computes the mean
// difference vector per layer. This "refusal direction" can then
// be used for activation steering.
//
// Per Codex advice:
// - Capture at first assistant decode token position
// - Extract directions in F32
// - Start with TokenMixerOut, use ResidualPostMlp as cross-check

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::qwen35::Qwen35Model;
use super::qwen35_generate::format_chat_prompt;
use super::qwen35_steering::{CaptureTrace, HookPoint, RuntimeCtx};

/// A pair of contrastive prompts for direction extraction.
pub struct ContrastivePair {
    pub harmful: String,
    pub harmless: String,
}

/// Extracted refusal directions per layer.
pub struct RefusalDirections {
    /// Per-layer direction vectors [hidden_size], F32, unit-normalized.
    pub directions: Vec<Tensor>,
    /// Per-layer magnitude of the mean difference (before normalization).
    pub magnitudes: Vec<f32>,
    /// Which hook point these were extracted from.
    pub hook_point: HookPoint,
    /// Number of pairs used.
    pub num_pairs: usize,
}

/// Extract refusal directions from contrastive prompt pairs.
///
/// For each pair, runs both prompts through the model in prefill mode
/// and captures activations at `hook_point` for the last prompt token.
/// Then computes the mean difference (harmful - harmless) per layer.
pub fn extract_directions(
    model: &Qwen35Model,
    tokenizer: &tokenizers::Tokenizer,
    pairs: &[ContrastivePair],
    hook_point: HookPoint,
    device: &Device,
) -> Result<RefusalDirections> {
    let num_layers = model.num_layers();
    let hidden_size = model.config().hidden_size;

    // Accumulate per-layer difference vectors
    let mut diff_accum: Vec<Vec<f64>> = vec![vec![0.0; hidden_size]; num_layers];
    let mut valid_count = 0usize;

    for (i, pair) in pairs.iter().enumerate() {
        // Capture harmful prompt
        let harmful_capture = capture_prefill_activations(
            model, tokenizer, &pair.harmful, hook_point, device,
        )?;

        // Capture harmless prompt
        let harmless_capture = capture_prefill_activations(
            model, tokenizer, &pair.harmless, hook_point, device,
        )?;

        // Accumulate difference per layer
        match (harmful_capture, harmless_capture) {
            (Some(harmful_vecs), Some(harmless_vecs)) => {
                for layer_idx in 0..num_layers {
                    if let (Some(h_vec), Some(hl_vec)) =
                        (&harmful_vecs[layer_idx], &harmless_vecs[layer_idx])
                    {
                        for j in 0..hidden_size {
                            diff_accum[layer_idx][j] += (h_vec[j] - hl_vec[j]) as f64;
                        }
                    }
                }
                valid_count += 1;
                if (i + 1) % 10 == 0 || i + 1 == pairs.len() {
                    println!("    Processed {}/{} pairs", i + 1, pairs.len());
                }
            }
            _ => {
                eprintln!("    Warning: skipping pair {} (capture failed)", i);
            }
        }
    }

    if valid_count == 0 {
        anyhow::bail!("No valid pairs processed");
    }

    // Compute mean and normalize
    let mut directions = Vec::with_capacity(num_layers);
    let mut magnitudes = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let mean_diff: Vec<f32> = diff_accum[layer_idx]
            .iter()
            .map(|&v| (v / valid_count as f64) as f32)
            .collect();

        // Compute L2 norm
        let norm: f32 = mean_diff.iter().map(|&v| v * v).sum::<f32>().sqrt();
        magnitudes.push(norm);

        // Normalize to unit vector
        let unit: Vec<f32> = if norm > 1e-8 {
            mean_diff.iter().map(|&v| v / norm).collect()
        } else {
            mean_diff
        };

        let dir = Tensor::new(unit, device)?.to_dtype(DType::F32)?;
        directions.push(dir);
    }

    Ok(RefusalDirections {
        directions,
        magnitudes,
        hook_point,
        num_pairs: valid_count,
    })
}

/// Run a single prompt through the model in prefill mode and capture activations.
/// Returns per-layer activation vectors at the last prompt token, or None on failure.
fn capture_prefill_activations(
    model: &Qwen35Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    hook_point: HookPoint,
    device: &Device,
) -> Result<Option<Vec<Option<Vec<f32>>>>> {
    let formatted = format_chat_prompt(prompt);
    let encoding = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let input_ids = encoding.get_ids();
    let prompt_len = input_ids.len();

    let mut cache = model.make_cache();
    let mut trace = CaptureTrace::default();
    let mut ctx = RuntimeCtx {
        capture: Some(&mut trace),
        steering: None,
        is_prefill: true,
        decode_step: 0,
        seq_offset: 0,
        capture_pos: Some(prompt_len - 1), // last prompt token
    };

    let input = Tensor::new(input_ids, device)?.unsqueeze(0)?;
    model.forward_with_ctx(&input, &mut cache, &mut ctx)?;

    // Extract captured vectors
    if let Some(cap) = trace.prefill.first() {
        let vecs: Vec<Option<Vec<f32>>> = cap
            .layers
            .iter()
            .map(|l| match hook_point {
                HookPoint::TokenMixerOut => l.token_mixer_out.clone(),
                HookPoint::ResidualPostMlp => l.residual_post_mlp.clone(),
            })
            .collect();
        Ok(Some(vecs))
    } else {
        Ok(None)
    }
}

impl RefusalDirections {
    /// Print a summary of the extracted directions.
    pub fn print_summary(&self) {
        println!("  Refusal directions ({} pairs, {:?}):", self.num_pairs, self.hook_point);
        println!("  Layer magnitudes:");
        for (i, mag) in self.magnitudes.iter().enumerate() {
            let bar_len = (mag * 20.0).min(40.0) as usize;
            let bar: String = "#".repeat(bar_len);
            println!("    L{:2}: {:.4} {}", i, mag, bar);
        }
        let max_mag = self.magnitudes.iter().cloned().fold(0.0f32, f32::max);
        let max_layer = self
            .magnitudes
            .iter()
            .position(|&m| m == max_mag)
            .unwrap_or(0);
        println!(
            "  Strongest direction: layer {} (magnitude {:.4})",
            max_layer, max_mag
        );
    }
}
