// Expert refusal participation scoring
// Rank experts by how strongly they contribute to refusal behavior

use std::collections::{HashMap, HashSet};

use candle_core::DType;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::abliterate::direction_set::DirectionSet;
use crate::abliterate::directions::RefusalDirections;
use crate::abliterate::multi_direction::MultiRefusalDirections;
use crate::analysis::activations::ExpertStats;

/// Score describing how strongly an individual expert participates in refusal behavior.
#[derive(Debug, Clone)]
pub struct ExpertScore {
    /// Decoder layer index of this expert's MoE block.
    pub layer_idx: usize,
    /// Expert index within the MoE block.
    pub expert_idx: usize,
    /// Absolute projection of the expert's harmful-harmless difference onto the
    /// global refusal direction: `|dot(expert_diff, global_dir)|`.
    pub refusal_projection: f32,
    /// Ratio of harmful to harmless routing frequency.
    /// Higher values indicate the router preferentially selects this expert for harmful content.
    pub routing_bias: f32,
    /// Combined score: `refusal_projection * min(routing_bias, 3.0)`.
    /// Caps routing bias to prevent a rarely-activated expert from dominating.
    pub combined_score: f32,
    /// Whether a per-expert refusal direction was available for this expert.
    pub has_per_expert_direction: bool,
}

/// Score every expert in the model by refusal participation.
///
/// For each MoE layer and expert, computes:
/// - `refusal_projection`: how aligned the expert's activation difference is with the
///   global refusal direction at that layer.
/// - `routing_bias`: how much more frequently the router selects this expert for
///   harmful vs harmless prompts.
///
/// Results are sorted by `combined_score` descending.
pub fn score_experts(
    stats: &ExpertStats,
    directions: &RefusalDirections,
) -> anyhow::Result<Vec<ExpertScore>> {
    let num_moe_layers = stats.moe_layer_indices.len();
    let mut scores = Vec::new();

    for moe_pos in 0..num_moe_layers {
        let layer_idx = stats.moe_layer_indices[moe_pos];
        let num_experts = stats.harmful_counts[moe_pos].len();
        let global_dir = &directions.global[moe_pos];

        for expert_idx in 0..num_experts {
            let harmful_count = stats.harmful_counts[moe_pos][expert_idx];
            let harmless_count = stats.harmless_counts[moe_pos][expert_idx];

            // Skip experts that were never activated on either side
            if harmful_count == 0 && harmless_count == 0 {
                continue;
            }

            // --- Refusal projection ---
            // Use expert-level direction if available, otherwise compute from means
            let refusal_projection = match (
                &stats.harmful_means[moe_pos][expert_idx],
                &stats.harmless_means[moe_pos][expert_idx],
            ) {
                (Some(harmful_mean), Some(harmless_mean)) => {
                    let diff = (harmful_mean - harmless_mean)?;
                    let diff_f32 = diff.to_dtype(DType::F32)?;
                    let global_f32 = global_dir.to_dtype(DType::F32)?;
                    let dot = diff_f32.mul(&global_f32)?.sum_all()?.to_scalar::<f32>()?;
                    dot.abs()
                }
                _ => 0.0,
            };

            // --- Routing bias ---
            // Ratio of harmful to harmless activation frequency.
            // Add 1 to denominator to avoid division by zero.
            let routing_bias = if harmless_count > 0 {
                harmful_count as f32 / harmless_count as f32
            } else if harmful_count > 0 {
                // Expert only activated for harmful prompts — maximum bias
                3.0
            } else {
                0.0
            };

            let combined_score = refusal_projection * routing_bias.min(3.0);

            // Skip NaN scores — degenerate tensor ops can produce NaN, and
            // total_cmp sorts NaN above all finite values, which would cause
            // a NaN-scored expert to be selected for abliteration.
            if !combined_score.is_finite() {
                tracing::warn!(
                    "Skipping expert (layer {} expert {}) with non-finite score: {}",
                    layer_idx, expert_idx, combined_score,
                );
                continue;
            }

            let has_per_expert_direction = directions.per_expert.get(moe_pos)
                .and_then(|layer| layer.get(expert_idx))
                .and_then(|d| d.as_ref())
                .is_some();

            scores.push(ExpertScore {
                layer_idx,
                expert_idx,
                refusal_projection,
                routing_bias,
                combined_score,
                has_per_expert_direction,
            });
        }
    }

    // Sort descending by combined score
    scores.sort_by(|a, b| {
        b.combined_score.total_cmp(&a.combined_score)
    });

    tracing::info!("Scored {} experts across {} MoE layers", scores.len(), num_moe_layers);
    if let Some(top) = scores.first() {
        tracing::info!(
            "Top expert: layer {} expert {} (score={:.4}, projection={:.4}, bias={:.2})",
            top.layer_idx,
            top.expert_idx,
            top.combined_score,
            top.refusal_projection,
            top.routing_bias,
        );
    }

    Ok(scores)
}

/// Score experts using multi-directional refusal subspace projection.
///
/// For each expert, computes the weighted sum of squared projections across all SVD
/// directions: `score = sqrt(sum(w_k * dot(diff, dir_k)^2) / sum(w_k))`.
///
/// This catches experts involved in secondary refusal patterns that a single-direction
/// projection would miss. When there is only one direction with weight 1.0, this
/// collapses exactly to the single-direction formula.
pub fn score_experts_multi(
    stats: &ExpertStats,
    directions: &MultiRefusalDirections,
) -> anyhow::Result<Vec<ExpertScore>> {
    let num_moe_layers = stats.moe_layer_indices.len();
    let mut scores = Vec::new();

    for moe_pos in 0..num_moe_layers {
        let layer_idx = stats.moe_layer_indices[moe_pos];
        let num_experts = stats.harmful_counts[moe_pos].len();
        let global_dirs = &directions.global[moe_pos];

        for expert_idx in 0..num_experts {
            let harmful_count = stats.harmful_counts[moe_pos][expert_idx];
            let harmless_count = stats.harmless_counts[moe_pos][expert_idx];

            if harmful_count == 0 && harmless_count == 0 {
                continue;
            }

            // Use per-expert directions if available, else global
            let dirs_to_use = directions
                .per_expert
                .get(moe_pos)
                .and_then(|layer| layer.get(expert_idx))
                .and_then(|d| d.as_deref())
                .unwrap_or(global_dirs.as_slice());

            let has_per_expert = directions
                .per_expert
                .get(moe_pos)
                .and_then(|layer| layer.get(expert_idx))
                .and_then(|d| d.as_ref())
                .is_some();

            // Weighted sum of squared projections across all directions
            let refusal_projection = match (
                &stats.harmful_means[moe_pos][expert_idx],
                &stats.harmless_means[moe_pos][expert_idx],
            ) {
                (Some(harmful_mean), Some(harmless_mean)) => {
                    let diff = (harmful_mean - harmless_mean)?;
                    let diff_f32 = diff.to_dtype(DType::F32)?;
                    let mut weighted_sum = 0.0f32;
                    let mut weight_total = 0.0f32;
                    for wd in dirs_to_use {
                        let dir_f32 = wd.direction.to_dtype(DType::F32)?;
                        let dot = diff_f32.mul(&dir_f32)?.sum_all()?.to_scalar::<f32>()?;
                        weighted_sum += wd.weight * dot * dot;
                        weight_total += wd.weight;
                    }
                    if weight_total > 0.0 {
                        (weighted_sum / weight_total).sqrt()
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };

            let routing_bias = if harmless_count > 0 {
                harmful_count as f32 / harmless_count as f32
            } else if harmful_count > 0 {
                3.0
            } else {
                0.0
            };

            let combined_score = refusal_projection * routing_bias.min(3.0);

            if !combined_score.is_finite() {
                tracing::warn!(
                    "Skipping expert (layer {} expert {}) with non-finite score: {}",
                    layer_idx, expert_idx, combined_score,
                );
                continue;
            }

            scores.push(ExpertScore {
                layer_idx,
                expert_idx,
                refusal_projection,
                routing_bias,
                combined_score,
                has_per_expert_direction: has_per_expert,
            });
        }
    }

    scores.sort_by(|a, b| {
        b.combined_score.total_cmp(&a.combined_score)
    });

    tracing::info!(
        "Multi-directional scoring: {} experts across {} MoE layers",
        scores.len(),
        num_moe_layers
    );

    Ok(scores)
}

/// Dispatch scoring based on DirectionSet variant.
pub fn score_experts_dispatch(
    stats: &ExpertStats,
    directions: &DirectionSet,
) -> anyhow::Result<Vec<ExpertScore>> {
    match directions {
        DirectionSet::Single(d) => score_experts(stats, d),
        DirectionSet::Multi(d) => score_experts_multi(stats, d),
    }
}

/// Select the "guilty" experts whose combined score is high enough to warrant abliteration.
///
/// - If `threshold` is `Some(t)`, filter by `combined_score >= t`.
/// - If `threshold` is `None`, use the automatic elbow method: find the largest
///   relative gap between consecutive sorted scores and split there if the gap
///   exceeds 30% of the higher score.
pub fn select_guilty_experts<'a>(
    scores: &'a [ExpertScore],
    threshold: Option<f32>,
) -> Vec<&'a ExpertScore> {
    if scores.is_empty() {
        return Vec::new();
    }

    match threshold {
        Some(t) => {
            let selected: Vec<_> = scores.iter().filter(|s| s.combined_score >= t).collect();
            tracing::info!(
                "Manual threshold {:.4}: selected {} / {} experts",
                t,
                selected.len(),
                scores.len(),
            );
            selected
        }
        None => {
            let selected = auto_threshold(scores);
            tracing::info!(
                "Auto-threshold (elbow method): selected {} / {} experts",
                selected.len(),
                scores.len(),
            );
            selected
        }
    }
}

/// Elbow method with percentage-of-max fallback for smooth distributions.
///
/// Multi-directional scoring produces a long tail of near-zero scores. Without
/// filtering, the globally largest relative gap often occurs between noise-level
/// values (e.g., 0.002 → 0.001 is a 50% gap), causing the method to select
/// nearly all experts. We first exclude scores below 5% of the maximum, then
/// search for the elbow within the remaining meaningful range.
///
/// Walk through consecutive pairs in the meaningful range. For each gap, compute
/// the relative size as `(score[i] - score[i+1]) / score[i]`. If the largest
/// such gap exceeds 0.30 (30%), split there — everything at index <= i is "guilty".
///
/// If no clear elbow exists but the meaningful range has real variance (max > 3x
/// min meaningful score), use 15% of the max score as a cutoff. This handles
/// smooth, gradually tapering distributions from multi-directional scoring where
/// there's no single sharp break point.
///
/// If the distribution is truly flat (all scores similar), return just the top expert.
fn auto_threshold(scores: &[ExpertScore]) -> Vec<&ExpertScore> {
    if scores.len() <= 1 {
        return scores.iter().collect();
    }

    // Filter to meaningful scores: ignore the noise tail below 5% of max.
    let max_score = scores[0].combined_score;
    let score_floor = max_score * 0.05;
    let meaningful_end = scores
        .iter()
        .position(|s| s.combined_score < score_floor)
        .unwrap_or(scores.len());

    // Need at least 2 scores to find a gap; cap at the meaningful range
    let search_end = meaningful_end.max(2).min(scores.len());

    let mut best_gap = 0.0_f32;
    let mut best_split = 0_usize; // index of last "guilty" expert

    for i in 0..search_end - 1 {
        let higher = scores[i].combined_score;
        let lower = scores[i + 1].combined_score;

        if higher <= 0.0 {
            // No positive scores left
            break;
        }

        let relative_gap = (higher - lower) / higher;
        if relative_gap > best_gap {
            best_gap = relative_gap;
            best_split = i;
        }
    }

    if best_gap > 0.30 {
        // Clear elbow found — split there
        let cutoff = best_split + 1;
        tracing::debug!(
            "Elbow at index {} (gap={:.1}%): scores {:.4} -> {:.4}, floor={:.4}",
            best_split,
            best_gap * 100.0,
            scores[best_split].combined_score,
            scores.get(best_split + 1).map_or(0.0, |s| s.combined_score),
            score_floor,
        );
        scores[..cutoff].iter().collect()
    } else if meaningful_end > 1
        && max_score > 0.0
        && scores[meaningful_end - 1].combined_score > 0.0
        && max_score > scores[meaningful_end - 1].combined_score * 3.0
    {
        // Smooth distribution with real variance but no sharp elbow.
        // Use 15% of max score as cutoff — selects experts with meaningful
        // refusal participation while excluding the low-scoring tail.
        let cutoff_score = max_score * 0.15;
        let cutoff_idx = scores
            .iter()
            .position(|s| s.combined_score < cutoff_score)
            .unwrap_or(scores.len())
            .max(1);
        tracing::debug!(
            "No elbow (max gap={:.1}%), using 15% of max ({:.4}) as cutoff: {} / {} experts (meaningful: {})",
            best_gap * 100.0,
            cutoff_score,
            cutoff_idx,
            scores.len(),
            meaningful_end,
        );
        scores[..cutoff_idx].iter().collect()
    } else {
        // Truly flat or narrow range — conservatively select just the top expert
        tracing::debug!(
            "No clear elbow (max gap={:.1}%), selecting top expert only",
            best_gap * 100.0,
        );
        vec![&scores[0]]
    }
}

/// Select experts using Stability-Based Selection (SES).
///
/// Scores K bootstrap subsamples of the input scores, ranks experts on each,
/// and returns experts that appear in the top-N across all K samples (strict
/// intersection). Falls back to frequency >= (K-1) if strict intersection
/// yields fewer than `min_cardinality` (3) experts.
///
/// When K=1, equivalent to taking the top-N experts from the full score list.
pub fn select_experts_stable(
    scores: &[ExpertScore],
    k: usize,
    top_n: usize,
    seed: u64,
) -> Vec<&ExpertScore> {
    if scores.is_empty() {
        return vec![];
    }

    let k = k.max(1);
    let top_n = top_n.min(scores.len());
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Count how many times each (layer, expert) appears in top-N across K folds
    let mut frequency: HashMap<(usize, usize), usize> = HashMap::new();
    let sample_size = if k == 1 {
        scores.len()
    } else {
        let s = (scores.len() as f64 * 0.8).ceil() as usize;
        s.max(1).min(scores.len())
    };

    let mut indices: Vec<usize> = (0..scores.len()).collect();

    for _ in 0..k {
        indices.shuffle(&mut rng);
        let mut subsample: Vec<&ExpertScore> = indices[..sample_size]
            .iter()
            .map(|&i| &scores[i])
            .collect();
        subsample.sort_by(|a, b| b.combined_score.total_cmp(&a.combined_score));

        for s in subsample.iter().take(top_n) {
            *frequency.entry((s.layer_idx, s.expert_idx)).or_insert(0) += 1;
        }
    }

    // Strict intersection: experts in top-N of ALL K folds
    let min_cardinality = 3;
    let strict: Vec<&ExpertScore> = scores
        .iter()
        .filter(|s| frequency.get(&(s.layer_idx, s.expert_idx)).copied().unwrap_or(0) == k)
        .collect();

    if strict.len() >= min_cardinality {
        let mut strict = strict;
        strict.sort_by(|a, b| b.combined_score.total_cmp(&a.combined_score));
        tracing::info!(
            "SES: strict intersection found {} experts (K={}, top_n={})",
            strict.len(),
            k,
            top_n,
        );
        return strict;
    }

    // Fallback: frequency >= K-1
    let fallback_threshold = k.saturating_sub(1).max(1);
    let mut fallback: Vec<&ExpertScore> = scores
        .iter()
        .filter(|s| {
            frequency
                .get(&(s.layer_idx, s.expert_idx))
                .copied()
                .unwrap_or(0)
                >= fallback_threshold
        })
        .collect();
    fallback.sort_by(|a, b| {
        b.combined_score.total_cmp(&a.combined_score)
    });

    tracing::info!(
        "SES: strict intersection too small ({}), using frequency >= {} fallback ({} experts)",
        strict.len(),
        fallback_threshold,
        fallback.len(),
    );

    if fallback.is_empty() {
        // Ultimate fallback: return top expert by score
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.combined_score.total_cmp(&b.combined_score)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        vec![&scores[max_idx]]
    } else {
        fallback
    }
}

/// Decompose stable experts into HCDG (detection) and HRCG (control) groups.
///
/// - **HCDG** (Harmful Content Detection Group): experts present in both regular
///   AND jailbreak stable sets. These experts detect harmful content regardless
///   of whether the prompt uses jailbreak techniques.
/// - **HRCG** (Harmful Response Control Group): experts present in the regular
///   stable set but NOT the jailbreak set. These experts enforce refusal behavior.
///
/// Returns `(hcdg, hrcg)`. **HRCG experts are the abliteration targets** — disabling
/// enforcement while preserving detection means the model recognises harmful content
/// but does not refuse to answer.
pub fn decompose_hcdg_hrcg<'a>(
    regular_stable: &[&'a ExpertScore],
    jailbreak_stable: &[&'a ExpertScore],
) -> (Vec<&'a ExpertScore>, Vec<&'a ExpertScore>) {
    let jailbreak_set: HashSet<(usize, usize)> = jailbreak_stable
        .iter()
        .map(|s| (s.layer_idx, s.expert_idx))
        .collect();

    let mut hcdg = Vec::new();
    let mut hrcg = Vec::new();

    for &score in regular_stable {
        let key = (score.layer_idx, score.expert_idx);
        if jailbreak_set.contains(&key) {
            hcdg.push(score);
        } else {
            hrcg.push(score);
        }
    }

    tracing::info!(
        "HCDG/HRCG decomposition: {} detection, {} control (abliteration targets)",
        hcdg.len(),
        hrcg.len(),
    );

    (hcdg, hrcg)
}
