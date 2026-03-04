// Expert routing capture + residual stream activation recording
// Per-expert, per-layer activation tracking during inference

use candle_core::{DType, Device, Tensor};
use indicatif::{ProgressBar, ProgressStyle};

use crate::model::arch::{MoeModel, ModelOutput};

/// Accumulated per-expert and per-layer activation statistics.
///
/// All means are computed from the LAST token position of each prompt,
/// since that is where refusal signals are strongest in autoregressive models.
pub struct ExpertStats {
    /// Per-MoE-layer, per-expert mean expert output for harmful prompts.
    /// Shape: `[num_moe_layers][num_experts]`, `None` if expert was never routed.
    pub harmful_means: Vec<Vec<Option<Tensor>>>,
    /// Per-MoE-layer, per-expert mean expert output for harmless prompts.
    pub harmless_means: Vec<Vec<Option<Tensor>>>,
    /// Per-layer mean residual stream (input to MoE block) for harmful prompts.
    pub harmful_residual_means: Vec<Tensor>,
    /// Per-layer mean residual stream (input to MoE block) for harmless prompts.
    pub harmless_residual_means: Vec<Tensor>,
    /// Per-MoE-layer, per-expert activation count for harmful prompts.
    pub harmful_counts: Vec<Vec<usize>>,
    /// Per-MoE-layer, per-expert activation count for harmless prompts.
    pub harmless_counts: Vec<Vec<usize>>,
    /// The decoder layer indices that correspond to MoE layers (in order).
    pub moe_layer_indices: Vec<usize>,
    /// Number of experts per MoE layer.
    pub num_experts: usize,

    // ----- Raw per-prompt tensors for SVD-based multi-direction extraction -----

    /// Per-MoE-layer raw residual stream tensors for harmful prompts.
    /// `[moe_layer_idx][prompt_idx]` -> [hidden_dim] tensor.
    pub harmful_residuals_raw: Vec<Vec<Tensor>>,
    /// Per-MoE-layer raw residual stream tensors for harmless prompts.
    pub harmless_residuals_raw: Vec<Vec<Tensor>>,
    /// Per-MoE-layer, per-expert raw output tensors for harmful prompts.
    /// `[moe_layer_idx][expert_idx][prompt_idx]` -> [hidden_dim] tensor.
    pub harmful_expert_raw: Vec<Vec<Vec<Tensor>>>,
    /// Per-MoE-layer, per-expert raw output tensors for harmless prompts.
    pub harmless_expert_raw: Vec<Vec<Vec<Tensor>>>,
}

/// Collect per-expert activation statistics by running harmful and harmless prompts
/// through the model one at a time with capture enabled.
///
/// Each prompt's token IDs are passed as a 1D tensor of shape `[seq_len]`.
/// We unsqueeze to `[1, seq_len]` for the model's expected `[B, S]` input.
pub fn collect_expert_stats(
    model: &dyn MoeModel,
    harmful_tokens: &[Tensor],
    harmless_tokens: &[Tensor],
    _device: &Device,
) -> anyhow::Result<ExpertStats> {
    let moe_indices = model.moe_layer_indices();
    let num_moe_layers = moe_indices.len();
    let num_experts = model.num_experts();

    // -----------------------------------------------------------------------
    // Accumulators: sums in F32 for numerical stability, divided by count at end.
    // -----------------------------------------------------------------------

    // Expert output sums: [moe_layer][expert] -> Option<Tensor>
    let mut harmful_expert_sums: Vec<Vec<Option<Tensor>>> =
        vec![vec![None; num_experts]; num_moe_layers];
    let mut harmless_expert_sums: Vec<Vec<Option<Tensor>>> =
        vec![vec![None; num_experts]; num_moe_layers];

    // Expert activation counts
    let mut harmful_counts: Vec<Vec<usize>> = vec![vec![0; num_experts]; num_moe_layers];
    let mut harmless_counts: Vec<Vec<usize>> = vec![vec![0; num_experts]; num_moe_layers];

    // Residual stream sums: [moe_layer] -> Option<Tensor>
    let mut harmful_residual_sums: Vec<Option<Tensor>> = vec![None; num_moe_layers];
    let mut harmless_residual_sums: Vec<Option<Tensor>> = vec![None; num_moe_layers];
    let mut harmful_prompt_count: usize = 0;
    let mut harmless_prompt_count: usize = 0;

    // Raw per-prompt tensors for SVD-based multi-direction extraction
    let mut harmful_residuals_raw: Vec<Vec<Tensor>> = vec![Vec::new(); num_moe_layers];
    let mut harmless_residuals_raw: Vec<Vec<Tensor>> = vec![Vec::new(); num_moe_layers];
    let mut harmful_expert_raw: Vec<Vec<Vec<Tensor>>> =
        vec![vec![Vec::new(); num_experts]; num_moe_layers];
    let mut harmless_expert_raw: Vec<Vec<Vec<Tensor>>> =
        vec![vec![Vec::new(); num_experts]; num_moe_layers];

    // Build a lookup: decoder layer_idx -> moe_layer position in our arrays
    let mut layer_to_moe: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (moe_pos, &layer_idx) in moe_indices.iter().enumerate() {
        layer_to_moe.insert(layer_idx, moe_pos);
    }

    // -----------------------------------------------------------------------
    // Process harmful prompts
    // -----------------------------------------------------------------------
    let total_prompts = (harmful_tokens.len() + harmless_tokens.len()) as u64;
    let pb = ProgressBar::new(total_prompts);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("       Activations [{bar:30}] {pos}/{len} prompts ({eta})")
            .unwrap()
            .progress_chars("=> "),
    );

    tracing::info!("Collecting activations from {} harmful prompts...", harmful_tokens.len());
    for tokens in harmful_tokens.iter() {
        let input = tokens.unsqueeze(0)?; // [1, seq_len]
        let output = model.forward(&input, true)?;

        accumulate_captures(
            &output,
            &layer_to_moe,
            &mut AccumulationBuffers {
                expert_sums: &mut harmful_expert_sums,
                expert_counts: &mut harmful_counts,
                residual_sums: &mut harmful_residual_sums,
                residuals_raw: &mut harmful_residuals_raw,
                expert_raw: &mut harmful_expert_raw,
            },
        )?;
        harmful_prompt_count += 1;
        pb.inc(1);
    }

    // -----------------------------------------------------------------------
    // Process harmless prompts
    // -----------------------------------------------------------------------
    tracing::info!("Collecting activations from {} harmless prompts...", harmless_tokens.len());
    for tokens in harmless_tokens.iter() {
        let input = tokens.unsqueeze(0)?; // [1, seq_len]
        let output = model.forward(&input, true)?;

        accumulate_captures(
            &output,
            &layer_to_moe,
            &mut AccumulationBuffers {
                expert_sums: &mut harmless_expert_sums,
                expert_counts: &mut harmless_counts,
                residual_sums: &mut harmless_residual_sums,
                residuals_raw: &mut harmless_residuals_raw,
                expert_raw: &mut harmless_expert_raw,
            },
        )?;
        harmless_prompt_count += 1;
        pb.inc(1);
    }
    pb.finish_and_clear();

    // -----------------------------------------------------------------------
    // Divide sums by counts to get means
    // -----------------------------------------------------------------------
    let harmful_means = compute_expert_means(harmful_expert_sums, &harmful_counts)?;
    let harmless_means = compute_expert_means(harmless_expert_sums, &harmless_counts)?;

    let harmful_residual_means =
        compute_residual_means(harmful_residual_sums, harmful_prompt_count)?;
    let harmless_residual_means =
        compute_residual_means(harmless_residual_sums, harmless_prompt_count)?;

    tracing::info!(
        "Activation collection complete: {} MoE layers, {} experts each",
        num_moe_layers,
        num_experts,
    );

    Ok(ExpertStats {
        harmful_means,
        harmless_means,
        harmful_residual_means,
        harmless_residual_means,
        harmful_counts,
        harmless_counts,
        moe_layer_indices: moe_indices,
        num_experts,
        harmful_residuals_raw,
        harmless_residuals_raw,
        harmful_expert_raw,
        harmless_expert_raw,
    })
}

/// Grouped mutable buffers for `accumulate_captures`, avoiding too many arguments.
struct AccumulationBuffers<'a> {
    expert_sums: &'a mut [Vec<Option<Tensor>>],
    expert_counts: &'a mut [Vec<usize>],
    residual_sums: &'a mut [Option<Tensor>],
    residuals_raw: &'a mut [Vec<Tensor>],
    expert_raw: &'a mut [Vec<Vec<Tensor>>],
}

/// Accumulate a single prompt's forward pass captures into running sums
/// and raw per-prompt tensors.
///
/// Only the LAST token position's activations are accumulated.
fn accumulate_captures(
    output: &ModelOutput,
    layer_to_moe: &std::collections::HashMap<usize, usize>,
    buffers: &mut AccumulationBuffers,
) -> anyhow::Result<()> {
    // --- Residual streams ---
    for (layer_idx, residual) in &output.residual_states {
        let moe_pos = match layer_to_moe.get(layer_idx) {
            Some(&pos) => pos,
            None => continue,
        };

        // residual shape: [1, seq_len, hidden_size] -- take last token position
        let seq_len = residual.dim(1)?;
        let last_tok = residual
            .narrow(1, seq_len - 1, 1)? // [1, 1, hidden_size]
            .squeeze(0)?                  // [1, hidden_size]
            .squeeze(0)?                  // [hidden_size]
            .to_dtype(DType::F32)?;

        // Store raw tensor for SVD
        buffers.residuals_raw[moe_pos].push(last_tok.clone());

        buffers.residual_sums[moe_pos] = Some(match buffers.residual_sums[moe_pos].take() {
            Some(existing) => (existing + &last_tok)?,
            None => last_tok,
        });
    }

    // --- Expert outputs ---
    // The MoE forward pass only captures outputs for the last token position,
    // so each expert has at most one entry in its outputs list.
    for (layer_idx, capture) in &output.moe_captures {
        let moe_pos = match layer_to_moe.get(layer_idx) {
            Some(&pos) => pos,
            None => continue,
        };

        for (expert_idx, token_outputs) in capture.expert_outputs.iter().enumerate() {
            if let Some((_tok_idx, expert_output)) = token_outputs.first() {
                let output_f32 = expert_output.to_dtype(DType::F32)?;

                // Store raw tensor for SVD
                buffers.expert_raw[moe_pos][expert_idx].push(output_f32.clone());

                buffers.expert_sums[moe_pos][expert_idx] =
                    Some(match buffers.expert_sums[moe_pos][expert_idx].take() {
                        Some(existing) => (existing + &output_f32)?,
                        None => output_f32,
                    });
                buffers.expert_counts[moe_pos][expert_idx] += 1;
            }
        }
    }

    Ok(())
}

/// Divide expert output sums by their counts to produce means.
fn compute_expert_means(
    sums: Vec<Vec<Option<Tensor>>>,
    counts: &[Vec<usize>],
) -> anyhow::Result<Vec<Vec<Option<Tensor>>>> {
    let mut means = Vec::with_capacity(sums.len());
    for (moe_pos, expert_sums) in sums.into_iter().enumerate() {
        let mut layer_means = Vec::with_capacity(expert_sums.len());
        for (expert_idx, sum_opt) in expert_sums.into_iter().enumerate() {
            let mean = match sum_opt {
                Some(sum) if counts[moe_pos][expert_idx] > 0 => {
                    let count = counts[moe_pos][expert_idx] as f64;
                    Some((sum / count)?)
                }
                _ => None,
            };
            layer_means.push(mean);
        }
        means.push(layer_means);
    }
    Ok(means)
}

/// Divide residual stream sums by prompt count to produce means.
fn compute_residual_means(
    sums: Vec<Option<Tensor>>,
    prompt_count: usize,
) -> anyhow::Result<Vec<Tensor>> {
    if prompt_count == 0 {
        anyhow::bail!("Cannot compute residual means with zero prompts");
    }
    let count = prompt_count as f64;
    let mut means = Vec::with_capacity(sums.len());
    for sum_opt in sums {
        match sum_opt {
            Some(sum) => means.push((sum / count)?),
            None => anyhow::bail!("Missing residual sum for a MoE layer — model produced no residual captures"),
        }
    }
    Ok(means)
}
