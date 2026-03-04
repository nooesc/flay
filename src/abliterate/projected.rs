use anyhow::Result;
use candle_core::{DType, Tensor};

/// Decompose a refusal direction into its compliance-suppression component.
///
/// The refusal direction `r` has two parts:
/// 1. A component aligned with the compliance direction `c` (refusal-push)
/// 2. A component orthogonal to `c` (compliance-suppression)
///
/// We remove only the compliance-suppression part, preserving the model's
/// ability to generate compliant responses.
///
/// Returns the compliance-suppression component (normalized).
pub fn project_refusal_direction(
    refusal_dir: &Tensor,   // L2-normalized refusal direction [hidden_dim]
    harmless_mean: &Tensor,  // Mean harmless activation at this layer [hidden_dim]
) -> Result<Tensor> {
    let r = refusal_dir.to_dtype(DType::F32)?;
    let b = harmless_mean.to_dtype(DType::F32)?;

    // Compliance direction = normalized harmless mean
    let b_norm = b.sqr()?.sum_all()?.sqrt()?;
    let c = b.broadcast_div(&b_norm)?;

    // Project refusal onto compliance: r_compliance = (r . c) * c
    let dot = r
        .unsqueeze(0)?
        .matmul(&c.unsqueeze(1)?)?
        .squeeze(0)?
        .squeeze(0)?;
    let r_compliance = c.broadcast_mul(&dot)?;

    // Compliance-suppression = r - r_compliance
    let r_suppress = (&r - &r_compliance)?;

    // Normalize
    let suppress_norm = r_suppress.sqr()?.sum_all()?.sqrt()?;
    let suppress_norm_val: f32 = suppress_norm.to_scalar()?;

    if suppress_norm_val < 1e-8 {
        // Refusal direction is fully aligned with compliance — nothing to suppress
        tracing::warn!(
            "Refusal direction fully aligned with compliance direction, \
             falling back to unprojected direction"
        );
        return Ok(r);
    }

    let result = r_suppress.broadcast_div(&suppress_norm)?;
    Ok(result)
}
