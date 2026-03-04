// Per-expert refusal direction computation
// Difference-of-means on expert-specific activations
// Multi-directional SVD-based extraction

use candle_core::Tensor;

use crate::abliterate::multi_direction::{extract_directions_svd, MultiRefusalDirections};
use crate::analysis::activations::ExpertStats;

/// Computed refusal directions: one global direction per MoE layer, plus
/// optional per-expert directions where sufficient data exists.
pub struct RefusalDirections {
    /// Per-MoE-layer global refusal direction (L2-normalized).
    /// Computed as `normalize(harmful_residual_mean - harmless_residual_mean)`.
    pub global: Vec<Tensor>,
    /// Per-MoE-layer, per-expert refusal direction (L2-normalized).
    /// `None` if either harmful or harmless activation count was below the threshold.
    pub per_expert: Vec<Vec<Option<Tensor>>>,
    /// Decoder layer indices corresponding to each MoE position.
    /// `moe_layer_indices[moe_pos] = decoder_layer_idx`.
    pub moe_layer_indices: Vec<usize>,
}

/// Compute refusal directions from collected expert statistics.
///
/// - **Global directions**: difference of mean residual streams (harmful - harmless),
///   L2-normalized, one per MoE layer.
/// - **Per-expert directions**: difference of mean expert outputs (harmful - harmless),
///   L2-normalized, only computed when both harmful and harmless activation counts
///   meet `min_activation_count`.
pub fn compute_refusal_directions(
    stats: &ExpertStats,
    min_activation_count: usize,
) -> anyhow::Result<RefusalDirections> {
    let num_moe_layers = stats.moe_layer_indices.len();

    // -----------------------------------------------------------------------
    // Global directions: harmful_residual_mean - harmless_residual_mean
    // -----------------------------------------------------------------------
    let mut global = Vec::with_capacity(num_moe_layers);
    for moe_pos in 0..num_moe_layers {
        let diff = (&stats.harmful_residual_means[moe_pos]
            - &stats.harmless_residual_means[moe_pos])?;
        global.push(l2_normalize(&diff)?);
    }

    // -----------------------------------------------------------------------
    // Per-expert directions
    // -----------------------------------------------------------------------
    let mut per_expert = Vec::with_capacity(num_moe_layers);
    for moe_pos in 0..num_moe_layers {
        let num_experts = stats.harmful_means[moe_pos].len();
        let mut layer_directions = Vec::with_capacity(num_experts);

        for expert_idx in 0..num_experts {
            let harmful_count = stats.harmful_counts[moe_pos][expert_idx];
            let harmless_count = stats.harmless_counts[moe_pos][expert_idx];

            let direction = if harmful_count >= min_activation_count
                && harmless_count >= min_activation_count
            {
                match (
                    &stats.harmful_means[moe_pos][expert_idx],
                    &stats.harmless_means[moe_pos][expert_idx],
                ) {
                    (Some(harmful_mean), Some(harmless_mean)) => {
                        let diff = (harmful_mean - harmless_mean)?;
                        Some(l2_normalize(&diff)?)
                    }
                    _ => None,
                }
            } else {
                None
            };

            layer_directions.push(direction);
        }

        per_expert.push(layer_directions);
    }

    tracing::info!(
        "Computed refusal directions: {} global, {} per-expert layers",
        global.len(),
        per_expert.len(),
    );

    Ok(RefusalDirections {
        global,
        per_expert,
        moe_layer_indices: stats.moe_layer_indices.clone(),
    })
}

/// Compute multi-directional refusal directions via SVD on per-prompt activation data.
///
/// Instead of a single refusal direction (mean difference), extracts multiple
/// directions that capture refusal signals in higher-dimensional subspaces.
///
/// - **Global directions**: SVD on the centered difference matrix of residual streams
///   (harmful vs harmless), one set per MoE layer.
/// - **Per-expert directions**: SVD on the centered difference matrix of expert outputs,
///   only computed when both harmful and harmless activation counts meet `min_activation_count`.
///
/// `max_directions` controls the maximum number of SVD directions to extract.
/// `energy_threshold` controls the minimum relative singular value (ratio to largest)
/// for a direction to be included (e.g., 0.1 means keep directions with >= 10% of
/// the top singular value).
pub fn compute_multi_refusal_directions(
    stats: &ExpertStats,
    min_activation_count: usize,
    max_directions: usize,
    energy_threshold: f32,
) -> anyhow::Result<MultiRefusalDirections> {
    let moe_indices = stats.moe_layer_indices.clone();
    let num_experts = stats.num_experts;
    let mut global = Vec::new();
    let mut per_expert = Vec::new();

    for (moe_idx, _layer_idx) in moe_indices.iter().enumerate() {
        // Global directions from residual streams
        let global_dirs = extract_directions_svd(
            &stats.harmful_residuals_raw[moe_idx],
            &stats.harmless_residuals_raw[moe_idx],
            max_directions,
            energy_threshold,
        )?;
        global.push(global_dirs);

        // Per-expert directions
        let mut expert_dirs = Vec::new();
        for e in 0..num_experts {
            let h_raw = &stats.harmful_expert_raw[moe_idx][e];
            let b_raw = &stats.harmless_expert_raw[moe_idx][e];
            if h_raw.len() >= min_activation_count && b_raw.len() >= min_activation_count {
                let dirs =
                    extract_directions_svd(h_raw, b_raw, max_directions, energy_threshold)?;
                expert_dirs.push(Some(dirs));
            } else {
                expert_dirs.push(None); // Fall back to global directions
            }
        }
        per_expert.push(expert_dirs);
    }

    tracing::info!(
        "Computed multi-directional refusal directions: {} MoE layers, max {} directions each",
        global.len(),
        max_directions,
    );

    Ok(MultiRefusalDirections {
        global,
        per_expert,
        moe_layer_indices: moe_indices,
    })
}

/// L2-normalize a vector tensor: `v / ||v||_2`.
///
/// Input: 1D tensor of any dtype (will be cast to F32 internally).
/// Output: 1D F32 tensor with unit L2 norm.
/// Returns zero vector if input norm is below epsilon (avoids NaN).
pub fn l2_normalize(v: &Tensor) -> candle_core::Result<Tensor> {
    let v = v.to_dtype(candle_core::DType::F32)?;
    let norm: f32 = v.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
    if norm < 1e-10 {
        return Tensor::zeros_like(&v);
    }
    let norm_t = Tensor::new(norm, v.device())?;
    v.broadcast_div(&norm_t)
}
