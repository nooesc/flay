// Teacher-forced decode parity check for Qwen3.5.
//
// Verifies that prefill logits match step-by-step cached decode logits.
// This is critical before steering experiments: if these diverge,
// steering results will be misleading.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::qwen35::Qwen35Model;

/// Run teacher-forced parity check.
///
/// Processes `input_ids` two ways:
/// 1. Single prefill pass (all tokens at once)
/// 2. Step-by-step: first token as prefill, then one token at a time with cache
///
/// Returns max absolute logit difference across all positions.
pub fn check_decode_parity(
    model: &Qwen35Model,
    input_ids: &[u32],
    device: &Device,
) -> Result<ParityReport> {
    let seq_len = input_ids.len();
    if seq_len < 2 {
        anyhow::bail!("Need at least 2 tokens for parity check");
    }

    // 1. Full prefill: get logits for all positions
    let mut prefill_cache = model.make_cache();
    let input = Tensor::new(input_ids, device)?.unsqueeze(0)?;
    let prefill_logits = model.forward(&input, &mut prefill_cache)?; // [1, S, vocab]
    let prefill_logits_f32 = prefill_logits.to_dtype(DType::F32)?;

    // 2. Step-by-step decode: process one token at a time
    let mut step_cache = model.make_cache();
    let mut step_logits_list = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let token = Tensor::new(&[input_ids[t]], device)?.unsqueeze(0)?; // [1, 1]
        let logits = model.forward(&token, &mut step_cache)?; // [1, 1, vocab]
        let logits_f32 = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?; // [vocab]
        step_logits_list.push(logits_f32);
    }

    // 3. Compare logits at each position
    let mut max_diff = 0.0f32;
    let mut mean_diff = 0.0f32;
    let mut per_position = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let prefill_t = prefill_logits_f32
            .narrow(1, t, 1)?
            .squeeze(0)?
            .squeeze(0)?; // [vocab]
        let step_t = &step_logits_list[t]; // [vocab]

        let diff = (prefill_t - step_t)?.abs()?;
        let pos_max: f32 = diff.max(0)?.to_scalar()?;
        let pos_mean: f32 = diff.mean(0)?.to_scalar()?;

        // Check if top-1 token matches
        let prefill_top1: u32 = prefill_logits_f32
            .narrow(1, t, 1)?
            .squeeze(0)?
            .squeeze(0)?
            .argmax(0)?
            .to_scalar()?;
        let step_top1: u32 = step_t.argmax(0)?.to_scalar()?;

        per_position.push(PositionParity {
            position: t,
            max_logit_diff: pos_max,
            mean_logit_diff: pos_mean,
            top1_match: prefill_top1 == step_top1,
        });

        max_diff = max_diff.max(pos_max);
        mean_diff += pos_mean;
    }
    mean_diff /= seq_len as f32;

    Ok(ParityReport {
        seq_len,
        max_logit_diff: max_diff,
        mean_logit_diff: mean_diff,
        per_position,
    })
}

/// Parity results for a single position.
pub struct PositionParity {
    pub position: usize,
    pub max_logit_diff: f32,
    pub mean_logit_diff: f32,
    pub top1_match: bool,
}

/// Full parity report.
pub struct ParityReport {
    pub seq_len: usize,
    pub max_logit_diff: f32,
    pub mean_logit_diff: f32,
    pub per_position: Vec<PositionParity>,
}

impl ParityReport {
    /// Check if parity is acceptable (max diff < threshold).
    pub fn is_ok(&self, threshold: f32) -> bool {
        self.max_logit_diff < threshold
    }

    /// Print a summary.
    pub fn print_summary(&self) {
        println!("  Decode parity check ({} positions):", self.seq_len);
        println!(
            "    Max logit diff: {:.6}, Mean: {:.6}",
            self.max_logit_diff, self.mean_logit_diff
        );
        let top1_matches = self.per_position.iter().filter(|p| p.top1_match).count();
        println!(
            "    Top-1 match: {}/{} positions",
            top1_matches, self.seq_len
        );

        // Show worst positions
        let mut worst: Vec<_> = self.per_position.iter().collect();
        worst.sort_by(|a, b| b.max_logit_diff.partial_cmp(&a.max_logit_diff).unwrap());
        if worst[0].max_logit_diff > 0.01 {
            println!("    Worst positions:");
            for p in worst.iter().take(3) {
                println!(
                    "      pos {}: max_diff={:.6} mean_diff={:.6} top1_match={}",
                    p.position, p.max_logit_diff, p.mean_logit_diff, p.top1_match
                );
            }
        }
    }
}
