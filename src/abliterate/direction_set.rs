// Unified direction container for single and multi-directional abliteration modes.
// Provides DirectionSet (enum) and DirectionSlice (borrowed view) for polymorphic dispatch.

use anyhow::Result;
use candle_core::Tensor;

use crate::abliterate::directions::RefusalDirections;
use crate::abliterate::multi_direction::{MultiRefusalDirections, WeightedDirection};
use crate::abliterate::projected::project_refusal_direction;
use crate::analysis::activations::ExpertStats;

/// Unified container for either single or multi-directional refusal directions.
pub enum DirectionSet {
    Single(RefusalDirections),
    Multi(MultiRefusalDirections),
}

/// Borrowed view of directions for a specific expert.
pub enum DirectionSlice<'a> {
    Single(&'a Tensor),
    Multi(&'a [WeightedDirection]),
}

impl DirectionSet {
    pub fn moe_layer_indices(&self) -> &[usize] {
        match self {
            Self::Single(d) => &d.moe_layer_indices,
            Self::Multi(d) => &d.moe_layer_indices,
        }
    }

    /// Get the directions for a specific expert, falling back to global if per-expert
    /// is unavailable.
    pub fn directions_for(&self, moe_pos: usize, expert_idx: usize) -> DirectionSlice<'_> {
        match self {
            Self::Single(d) => {
                let dir = d
                    .per_expert
                    .get(moe_pos)
                    .and_then(|layer| layer.get(expert_idx))
                    .and_then(|d| d.as_ref())
                    .unwrap_or_else(|| {
                        d.global.get(moe_pos).unwrap_or_else(|| {
                            panic!("moe_pos {moe_pos} out of bounds for global directions (len={})", d.global.len())
                        })
                    });
                DirectionSlice::Single(dir)
            }
            Self::Multi(d) => {
                let dirs = d
                    .per_expert
                    .get(moe_pos)
                    .and_then(|layer| layer.get(expert_idx))
                    .and_then(|d| d.as_deref())
                    .unwrap_or_else(|| {
                        d.global.get(moe_pos).unwrap_or_else(|| {
                            panic!("moe_pos {moe_pos} out of bounds for global directions (len={})", d.global.len())
                        })
                    });
                DirectionSlice::Multi(dirs)
            }
        }
    }

    /// Apply projected decomposition: replace each refusal direction with its
    /// compliance-suppression component relative to the harmless mean at each layer.
    pub fn project(self, stats: &ExpertStats) -> Result<Self> {
        match self {
            Self::Single(d) => Ok(Self::Single(project_single(d, stats)?)),
            Self::Multi(d) => Ok(Self::Multi(project_multi(d, stats)?)),
        }
    }
}

fn project_single(d: RefusalDirections, stats: &ExpertStats) -> Result<RefusalDirections> {
    let global = d
        .global
        .iter()
        .enumerate()
        .map(|(i, dir)| project_refusal_direction(dir, &stats.harmless_residual_means[i]))
        .collect::<Result<Vec<_>>>()?;

    let per_expert = d
        .per_expert
        .into_iter()
        .enumerate()
        .map(|(moe_pos, layer)| {
            layer
                .into_iter()
                .map(|opt| {
                    opt.map(|dir| {
                        project_refusal_direction(&dir, &stats.harmless_residual_means[moe_pos])
                    })
                    .transpose()
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RefusalDirections {
        global,
        per_expert,
        moe_layer_indices: d.moe_layer_indices,
    })
}

fn project_multi(d: MultiRefusalDirections, stats: &ExpertStats) -> Result<MultiRefusalDirections> {
    let global = d
        .global
        .into_iter()
        .enumerate()
        .map(|(i, dirs)| {
            dirs.into_iter()
                .map(|wd| {
                    Ok(WeightedDirection {
                        direction: project_refusal_direction(
                            &wd.direction,
                            &stats.harmless_residual_means[i],
                        )?,
                        weight: wd.weight,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    let per_expert = d
        .per_expert
        .into_iter()
        .enumerate()
        .map(|(moe_pos, layer)| {
            layer
                .into_iter()
                .map(|opt| {
                    opt.map(|dirs| {
                        dirs.into_iter()
                            .map(|wd| {
                                Ok(WeightedDirection {
                                    direction: project_refusal_direction(
                                        &wd.direction,
                                        &stats.harmless_residual_means[moe_pos],
                                    )?,
                                    weight: wd.weight,
                                })
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .transpose()
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(MultiRefusalDirections {
        global,
        per_expert,
        moe_layer_indices: d.moe_layer_indices,
    })
}
