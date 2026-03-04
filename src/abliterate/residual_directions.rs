// Per-layer refusal directions for residual-stream abliteration.
// Sign-aligned and blended with global reference for consistency.

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::abliterate::directions::l2_normalize;
use crate::abliterate::multi_direction::WeightedDirection;
use crate::analysis::activations::ExpertStats;

/// Per-layer refusal directions for all decoder layers, ready for o_proj orthogonalization.
pub struct ResidualDirections {
    /// One refusal direction per decoder layer (sign-aligned, blended).
    /// Length = num_decoder_layers (e.g. 48 for Qwen3-30B-A3B).
    pub per_layer: Vec<Tensor>,
    /// Per-layer blending lambda values (how much global reference was mixed in).
    pub lambda: Vec<f32>,
    /// Index of the reference layer (strongest signal).
    pub reference_layer: usize,
}

/// Compute per-layer refusal directions from ExpertStats residual captures.
///
/// Since Qwen3-30B-A3B has MoE in all 48 layers, `stats.moe_layer_indices` covers
/// all decoder layers. For models with mixed dense/MoE layers, this function
/// requires that all decoder layers have captured residuals (asserted).
///
/// Algorithm:
/// 1. Compute raw direction per layer: `d_l = normalize(harmful_mean - harmless_mean)`
/// 2. Find reference layer: strongest signal by L2 norm of mean-diff
/// 3. Sign-align all directions to reference
/// 4. Blend weak layers with reference: `v_l = normalize((1 - lambda_l) * d_l + lambda_l * d_ref)`
///    where lambda_l = clamp(1.0 - s_l/s_max, 0.0, 0.8)
pub fn compute_residual_directions(stats: &ExpertStats) -> Result<ResidualDirections> {
    let num_layers = stats.moe_layer_indices.len();
    if num_layers == 0 {
        anyhow::bail!("No residual captures available");
    }

    // Step 1: Raw directions and signal strengths
    let mut raw_directions = Vec::with_capacity(num_layers);
    let mut signal_strengths = Vec::with_capacity(num_layers);

    for moe_pos in 0..num_layers {
        let diff = (&stats.harmful_residual_means[moe_pos]
            - &stats.harmless_residual_means[moe_pos])?;
        let diff_f32 = diff.to_dtype(DType::F32)?;
        let norm: f32 = diff_f32.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
        signal_strengths.push(norm);
        raw_directions.push(l2_normalize(&diff_f32)?);
    }

    // Step 2: Reference layer = strongest signal
    let (ref_idx, &s_max) = signal_strengths
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    if s_max < 1e-10 {
        anyhow::bail!(
            "All residual direction signal strengths are near zero — \
             harmful and harmless residuals are indistinguishable"
        );
    }
    let d_ref = &raw_directions[ref_idx];

    tracing::info!(
        "Reference layer: {} (decoder layer {}, signal strength {:.4})",
        ref_idx,
        stats.moe_layer_indices[ref_idx],
        s_max,
    );

    // Step 3: Sign-align all directions to reference
    let mut aligned = Vec::with_capacity(num_layers);
    for (_i, dir) in raw_directions.iter().enumerate() {
        let dot: f32 = dir
            .unsqueeze(0)?
            .matmul(&d_ref.unsqueeze(1)?)?
            .squeeze(0)?
            .squeeze(0)?
            .to_scalar()?;
        if dot < 0.0 {
            aligned.push(dir.neg()?);
        } else {
            aligned.push(dir.clone());
        }
    }

    // Step 4: Blend with reference
    let mut per_layer = Vec::with_capacity(num_layers);
    let mut lambda_values = Vec::with_capacity(num_layers);

    for (i, dir) in aligned.iter().enumerate() {
        let s_l = signal_strengths[i];
        let lambda_l = (1.0 - s_l / s_max).clamp(0.0, 0.8);
        lambda_values.push(lambda_l);

        // v_l = normalize((1 - lambda_l) * d_l + lambda_l * d_ref)
        let local_weight = (1.0 - lambda_l) as f64;
        let ref_weight = lambda_l as f64;
        let blended = (dir * local_weight)?.add(&(d_ref * ref_weight)?)?;
        per_layer.push(l2_normalize(&blended)?);
    }

    tracing::info!(
        "Computed {} per-layer residual directions (lambda range: {:.2}..{:.2})",
        per_layer.len(),
        lambda_values.iter().cloned().fold(f32::INFINITY, f32::min),
        lambda_values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
    );

    Ok(ResidualDirections {
        per_layer,
        lambda: lambda_values,
        reference_layer: ref_idx,
    })
}

/// Collapse multi-directional (SVD) directions to a single direction per layer
/// for o_proj orthogonalization.
///
/// Uses energy-weighted sum: `v_l = normalize(sum(energy_i * d_i))` for each layer.
/// This preserves the most important refusal signal while reducing to a single vector.
///
/// NOTE (Codex review #7): This function is defined but not wired into the pipeline
/// for v1. The pipeline uses `compute_residual_directions` which computes mean-diff
/// directions directly. Multi-direction collapse is only needed when combining
/// `--abliteration-mode multi*` with `--residual`, which is a future enhancement.
#[allow(dead_code)]
pub fn collapse_multi_directions(multi_dirs: &[Vec<WeightedDirection>]) -> Result<Vec<Tensor>> {
    let mut collapsed = Vec::with_capacity(multi_dirs.len());
    for dirs in multi_dirs {
        if dirs.is_empty() {
            anyhow::bail!("Empty direction set for a layer");
        }
        if dirs.len() == 1 {
            collapsed.push(dirs[0].direction.clone());
            continue;
        }
        // Energy-weighted sum
        let mut sum = (dirs[0].direction.to_dtype(DType::F32)? * dirs[0].weight as f64)?;
        for wd in &dirs[1..] {
            sum = sum.add(&(wd.direction.to_dtype(DType::F32)? * wd.weight as f64)?)?;
        }
        collapsed.push(l2_normalize(&sum)?);
    }
    Ok(collapsed)
}
