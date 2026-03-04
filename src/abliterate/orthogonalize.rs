// Weight orthogonalization for selected experts
// Variable-strength abliteration based on refusal score

use candle_core::{DType, Tensor};

use crate::abliterate::direction_set::{DirectionSet, DirectionSlice};
use crate::abliterate::directions::RefusalDirections;
use crate::abliterate::scoring::ExpertScore;
use crate::abliterate::weight_key::WeightKey;

/// Abliteration mode controlling direction extraction and projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum AbliterationMode {
    /// Single mean-difference direction, no projection (v1 behavior)
    Single,
    /// Multiple SVD directions, no projection
    Multi,
    /// Single direction with projected decomposition
    Projected,
    /// Multiple SVD directions with projected decomposition (recommended)
    MultiProjected,
}

impl std::fmt::Display for AbliterationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single => write!(f, "single"),
            Self::Multi => write!(f, "multi"),
            Self::Projected => write!(f, "projected"),
            Self::MultiProjected => write!(f, "multi-projected"),
        }
    }
}

impl AbliterationMode {
    /// Convert from the usize index used by HyperParams in the Bayesian optimizer.
    /// 0=Single, 1=Multi, 2=Projected, 3=MultiProjected.
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Single,
            1 => Self::Multi,
            2 => Self::Projected,
            _ => Self::MultiProjected,
        }
    }

    /// Whether this mode uses multi-directional SVD extraction.
    pub fn is_multi(&self) -> bool {
        matches!(self, Self::Multi | Self::MultiProjected)
    }

    /// Whether this mode applies projected decomposition.
    pub fn is_projected(&self) -> bool {
        matches!(self, Self::Projected | Self::MultiProjected)
    }
}

/// A single abliterated weight matrix, ready to be written back to safetensors.
pub struct AbliteratedWeight {
    /// Typed key identifying which weight matrix was modified.
    pub key: WeightKey,
    /// The modified weight tensor with refusal direction projected out.
    pub new_weight: Tensor,
    /// Abliteration strength applied (0.0 = no change, 1.0 = full projection removal).
    pub strength: f32,
}

/// Orthogonalize a weight matrix with respect to a refusal direction.
///
/// Removes the component of each row of `weight` that lies along `refusal_direction`,
/// scaled by `strength`:
///
/// ```text
/// W' = W - strength * r * (r^T @ W)
/// ```
///
/// where `r` is the L2-normalized refusal direction vector.
///
/// - `weight`: shape `[d_out, d_in]` (expert's down_proj weight)
/// - `refusal_direction`: shape `[d_out]` (unit vector)
/// - `strength`: scalar in `[0, 1]`
///
/// Computation is done in F32 for stability, then cast back to the original dtype.
pub fn orthogonalize_weight(
    weight: &Tensor,
    refusal_direction: &Tensor,
    strength: f32,
) -> candle_core::Result<Tensor> {
    let orig_dtype = weight.dtype();

    let w = weight.to_dtype(DType::F32)?;
    let r = refusal_direction.to_dtype(DType::F32)?;

    // r_row: [1, d_out]
    let r_row = r.unsqueeze(0)?;
    // rt_w = r_row @ W -> [1, d_in]
    let rt_w = r_row.matmul(&w)?;

    // r_col: [d_out, 1]
    let r_col = r.unsqueeze(1)?;
    // projection = r_col @ rt_w -> [d_out, d_in]
    let projection = r_col.matmul(&rt_w)?;

    // w_new = w - strength * projection
    let w_new = (w - (projection * strength as f64)?)?;

    w_new.to_dtype(orig_dtype)
}

/// Abliterate a set of guilty experts by orthogonalizing their down_proj weights.
///
/// Applies variable strength: the highest-scoring expert gets `strength_max`,
/// the lowest-scoring gets `strength_min`, with linear interpolation between.
///
/// For each guilty expert, uses its per-expert refusal direction if available,
/// otherwise falls back to the global direction for that MoE layer.
///
/// Handles both single and multi-directional modes via `DirectionSet`:
/// - **Single**: one call to `orthogonalize_weight` per expert.
/// - **Multi**: sequential orthogonalization across all SVD directions, with
///   weight-proportional strength splitting so total energy removed is bounded.
///
/// `original_weights` is a slice of `(layer_idx, expert_idx, weight_tensor)` tuples
/// providing the current down_proj weights for matching.
pub fn abliterate_experts(
    guilty: &[&ExpertScore],
    directions: &DirectionSet,
    original_weights: &[(usize, usize, Tensor)],
    strength_min: f32,
    strength_max: f32,
) -> anyhow::Result<Vec<AbliteratedWeight>> {
    if guilty.is_empty() {
        return Ok(Vec::new());
    }

    // Determine strength range from normalized scores
    let max_score = guilty
        .iter()
        .map(|s| s.combined_score)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_score = guilty
        .iter()
        .map(|s| s.combined_score)
        .fold(f32::INFINITY, f32::min);
    let score_range = max_score - min_score;

    // Build lookup: decoder_layer_idx -> moe_pos (index into directions vectors)
    let layer_to_moe: std::collections::HashMap<usize, usize> = directions
        .moe_layer_indices()
        .iter()
        .enumerate()
        .map(|(moe_pos, &layer_idx)| (layer_idx, moe_pos))
        .collect();

    let mut results = Vec::with_capacity(guilty.len());

    for expert_score in guilty {
        // --- Find the MoE position for this layer_idx ---
        let moe_pos = *layer_to_moe
            .get(&expert_score.layer_idx)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Decoder layer {} is not a MoE layer (known MoE layers: {:?})",
                    expert_score.layer_idx,
                    directions.moe_layer_indices(),
                )
            })?;

        // --- Compute variable strength ---
        let strength = if score_range > 0.0 {
            let normalized = (expert_score.combined_score - min_score) / score_range;
            strength_min + (strength_max - strength_min) * normalized
        } else {
            strength_max
        };

        // --- Find matching original weight ---
        let weight = original_weights
            .iter()
            .find(|(l, e, _)| *l == expert_score.layer_idx && *e == expert_score.expert_idx)
            .map(|(_, _, w)| w)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No original weight found for layer {} expert {}",
                    expert_score.layer_idx,
                    expert_score.expert_idx,
                )
            })?;

        // --- Orthogonalize (dispatch on direction type) ---
        let new_weight = match directions.directions_for(moe_pos, expert_score.expert_idx) {
            DirectionSlice::Single(dir) => orthogonalize_weight(weight, dir, strength)?,
            DirectionSlice::Multi(dirs) => {
                // Sequential orthogonalization across SVD directions.
                // Strength is split proportionally by singular value weight so total
                // energy removed is bounded by `strength`.
                let weight_sum: f32 = dirs.iter().map(|d| d.weight).sum();
                let mut w = weight.clone();
                for wd in dirs {
                    let sub_strength = if weight_sum > 0.0 {
                        strength * (wd.weight / weight_sum)
                    } else {
                        strength
                    };
                    w = orthogonalize_weight(&w, &wd.direction, sub_strength)?;
                }
                w
            }
        };

        tracing::debug!(
            "Abliterated layer {} expert {} (strength={:.2}, score={:.4})",
            expert_score.layer_idx,
            expert_score.expert_idx,
            strength,
            expert_score.combined_score,
        );

        results.push(AbliteratedWeight {
            key: WeightKey::MoeDownProj {
                layer: expert_score.layer_idx,
                expert: expert_score.expert_idx,
            },
            new_weight,
            strength,
        });
    }

    tracing::info!(
        "Abliterated {} experts (strength range: {:.2}..{:.2})",
        results.len(),
        results
            .iter()
            .map(|r| r.strength)
            .fold(f32::INFINITY, f32::min),
        results
            .iter()
            .map(|r| r.strength)
            .fold(f32::NEG_INFINITY, f32::max),
    );

    Ok(results)
}

/// Legacy wrapper for single-direction abliteration (used by existing tests).
#[allow(dead_code)]
pub(crate) fn abliterate_experts_single(
    guilty: &[&ExpertScore],
    directions: &RefusalDirections,
    original_weights: &[(usize, usize, Tensor)],
) -> anyhow::Result<Vec<AbliteratedWeight>> {
    abliterate_experts(
        guilty,
        &DirectionSet::Single(RefusalDirections {
            global: directions.global.clone(),
            per_expert: directions.per_expert.clone(),
            moe_layer_indices: directions.moe_layer_indices.clone(),
        }),
        original_weights,
        0.5,
        1.0,
    )
}
