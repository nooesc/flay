//! Decode-time expert routing analysis.

use std::collections::{HashMap, HashSet};

use crate::eval::generate::DecodeCaptures;

/// Accumulated decode-time routing statistics.
pub struct DecodeRoutingStats {
    pub moe_layer_indices: Vec<usize>,
    pub num_experts: usize,
    /// [moe_pos][expert_idx] -> (harmful_count, harmless_count, harmful_weight_sum, harmless_weight_sum)
    counts: Vec<Vec<(usize, usize, f32, f32)>>,
}

impl DecodeRoutingStats {
    pub fn new(moe_layer_indices: &[usize], num_experts: usize) -> Self {
        let counts = vec![vec![(0, 0, 0.0f32, 0.0f32); num_experts]; moe_layer_indices.len()];
        Self {
            moe_layer_indices: moe_layer_indices.to_vec(),
            num_experts,
            counts,
        }
    }

    pub fn accumulate(&mut self, captures: &DecodeCaptures, is_harmful: bool) {
        let layer_to_moe: HashMap<usize, usize> = self
            .moe_layer_indices
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();

        for step_captures in &captures.steps {
            for (layer_idx, capture) in step_captures {
                let Some(&moe_pos) = layer_to_moe.get(layer_idx) else {
                    continue;
                };
                for (tok_idx, expert_indices) in capture.expert_indices.iter().enumerate() {
                    let weights = &capture.routing_weights[tok_idx];
                    for (rank, &expert_id) in expert_indices.iter().enumerate() {
                        let eid = expert_id as usize;
                        if eid >= self.num_experts {
                            continue;
                        }
                        let weight = weights.get(rank).copied().unwrap_or(0.0);
                        let entry = &mut self.counts[moe_pos][eid];
                        if is_harmful {
                            entry.0 += 1;
                            entry.2 += weight;
                        } else {
                            entry.1 += 1;
                            entry.3 += weight;
                        }
                    }
                }
            }
        }
    }

    /// Score using corrected formula: max(ln(lift), 0) * sqrt(harmful_count)
    pub fn score_experts(&self, min_count: usize) -> Vec<(usize, usize, f32)> {
        let eps = 1e-6;
        let mut scores = Vec::new();

        for (moe_pos, &layer_idx) in self.moe_layer_indices.iter().enumerate() {
            for expert_idx in 0..self.num_experts {
                let (hc, lc, hw, lw) = self.counts[moe_pos][expert_idx];
                if hc < min_count {
                    continue;
                }

                let harmful_mass = hw / hc as f32;
                let harmless_mass = if lc > 0 { lw / lc as f32 } else { eps };
                let lift = (harmful_mass + eps) / (harmless_mass + eps);
                let log_lift = lift.ln().max(0.0);
                let score = log_lift * (hc as f32).sqrt();

                if score > 0.0 {
                    scores.push((layer_idx, expert_idx, score));
                }
            }
        }
        scores.sort_by(|a, b| b.2.total_cmp(&a.2));
        scores
    }

    pub fn jaccard_vs_prefill(
        &self,
        prefill_top: &[(usize, usize)],
        k: usize,
        min_count: usize,
    ) -> Vec<(usize, f32)> {
        let mut prefill_by_layer: HashMap<usize, HashSet<usize>> = HashMap::new();
        for &(layer, expert) in prefill_top {
            prefill_by_layer.entry(layer).or_default().insert(expert);
        }

        let scores = self.score_experts(min_count);
        let mut decode_by_layer: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(layer, expert, _) in &scores {
            decode_by_layer.entry(layer).or_default().push(expert);
        }

        let mut results = Vec::new();
        for &layer_idx in &self.moe_layer_indices {
            let ps: HashSet<usize> =
                prefill_by_layer.get(&layer_idx).cloned().unwrap_or_default();
            let ds: HashSet<usize> = decode_by_layer
                .get(&layer_idx)
                .map(|v| v.iter().take(k).copied().collect())
                .unwrap_or_default();
            if ps.is_empty() && ds.is_empty() {
                continue;
            }
            let inter = ps.intersection(&ds).count();
            let union = ps.union(&ds).count();
            results.push((
                layer_idx,
                if union > 0 {
                    inter as f32 / union as f32
                } else {
                    0.0
                },
            ));
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corrected_scoring_ignores_neutral() {
        let mut stats = DecodeRoutingStats::new(&[0], 4);
        // Expert 0: equal harmful/harmless -> lift=1 -> ln(1)=0 -> score=0
        stats.counts[0][0] = (10, 10, 5.0, 5.0);
        // Expert 1: harmful-biased -> higher per-activation harmful weight
        // harmful_mass = 8.0/10 = 0.8, harmless_mass = 1.0/10 = 0.1, lift >> 1
        stats.counts[0][1] = (10, 10, 8.0, 1.0);

        let scores = stats.score_experts(3);
        assert_eq!(scores.len(), 1, "only the harmful-biased expert should score");
        assert_eq!(scores[0].1, 1, "expert 1 should be the only scorer");
        assert!(scores[0].2 > 0.0, "score should be positive");
    }

    #[test]
    fn test_min_count_floor() {
        let mut stats = DecodeRoutingStats::new(&[0], 4);
        // Expert 0: only 1 harmful activation (below min_count=3)
        stats.counts[0][0] = (1, 0, 5.0, 0.0);
        let scores = stats.score_experts(3);
        assert!(scores.is_empty(), "below min_count should be filtered");
    }
}
