// Multi-directional refusal extraction via SVD (power iteration)
//
// Instead of a single refusal direction (mean harmful - mean harmless),
// extracts MULTIPLE refusal directions via SVD on the difference matrix.
// This catches refusal signals in higher-dimensional subspaces.

use anyhow::Result;
use candle_core::{DType, Tensor};

/// A refusal direction with its associated singular value weight.
#[derive(Clone)]
pub struct WeightedDirection {
    /// L2-normalized direction vector, shape [hidden_dim].
    pub direction: Tensor,
    /// Singular value (importance weight).
    pub weight: f32,
}

/// Multi-directional refusal directions for MoE layers.
#[derive(Clone)]
pub struct MultiRefusalDirections {
    /// Global directions extracted via SVD on residual stream differences.
    /// Indexed as `[moe_layer_idx][direction_idx]`.
    pub global: Vec<Vec<WeightedDirection>>,
    /// Per-expert directions (SVD on expert output differences).
    /// Indexed as `[moe_layer_idx][expert_idx]`, `None` if insufficient data.
    pub per_expert: Vec<Vec<Option<Vec<WeightedDirection>>>>,
    /// Decoder layer indices for MoE layers.
    pub moe_layer_indices: Vec<usize>,
}

/// Extract top-k refusal directions via SVD on the difference matrix.
///
/// Given harmful and harmless activation tensors (one per prompt, each [hidden_dim]),
/// computes D = H_centered - B_centered, then SVD(D) to get principal directions.
///
/// Returns directions where `singular_value / max_singular_value > energy_threshold`.
pub fn extract_directions_svd(
    harmful: &[Tensor],
    harmless: &[Tensor],
    max_directions: usize,
    energy_threshold: f32,
) -> Result<Vec<WeightedDirection>> {
    if harmful.is_empty() || harmless.is_empty() {
        anyhow::bail!("Cannot extract directions from empty activation data");
    }

    let n = harmful.len().min(harmless.len());
    if n < 2 {
        // Not enough samples for SVD -- fall back to mean difference
        let h_mean = Tensor::stack(harmful, 0)?.mean(0)?;
        let b_mean = Tensor::stack(harmless, 0)?.mean(0)?;
        let diff = (&h_mean - &b_mean)?;
        let norm = diff.sqr()?.sum_all()?.sqrt()?;
        let dir = diff.broadcast_div(&norm)?;
        return Ok(vec![WeightedDirection {
            direction: dir,
            weight: 1.0,
        }]);
    }

    // Build difference matrix D [n, hidden_dim]
    let h_stack = Tensor::stack(&harmful[..n], 0)?.to_dtype(DType::F32)?;
    let b_stack = Tensor::stack(&harmless[..n], 0)?.to_dtype(DType::F32)?;
    let h_mean = h_stack.mean(0)?;
    let b_mean = b_stack.mean(0)?;
    let h_centered = h_stack.broadcast_sub(&h_mean)?;
    let b_centered = b_stack.broadcast_sub(&b_mean)?;
    let diff_matrix = (&h_centered - &b_centered)?;

    // Use power iteration to extract top-k singular vectors
    let (singular_values, right_vectors): (Vec<f32>, Vec<Tensor>) = power_iteration_svd(
        &diff_matrix,
        max_directions.min(n),
        100, // iterations
    )?;

    // Select directions by energy threshold
    if singular_values.is_empty() || singular_values[0] < 1e-8 {
        anyhow::bail!("All singular values near zero -- no refusal signal found");
    }
    let s_max = singular_values[0];

    let mut directions = Vec::new();
    for (i, sv) in singular_values.iter().enumerate() {
        let ratio = sv / s_max;
        if ratio < energy_threshold {
            break;
        }
        let dir = &right_vectors[i];
        let norm = dir.sqr()?.sum_all()?.sqrt()?;
        let dir_normalized = dir.broadcast_div(&norm)?;
        directions.push(WeightedDirection {
            direction: dir_normalized,
            weight: *sv,
        });
    }

    // Defensive guard: should be unreachable since s[0]/s[0] = 1.0 always passes
    // the energy threshold, but kept for safety.
    if directions.is_empty() {
        // Energy threshold too aggressive -- return at least the top direction
        let dir = &right_vectors[0];
        let norm = dir.sqr()?.sum_all()?.sqrt()?;
        let dir_normalized = dir.broadcast_div(&norm)?;
        directions.push(WeightedDirection {
            direction: dir_normalized,
            weight: singular_values[0],
        });
    }

    tracing::debug!(
        "SVD extracted {} directions (energy threshold {}), top singular values: {:?}",
        directions.len(),
        energy_threshold,
        &singular_values[..directions.len().min(5)],
    );

    Ok(directions)
}

/// Power iteration to extract top-k singular vectors of a matrix.
///
/// For matrix A [m, n], computes A^T A [n, n], then uses power iteration
/// with deflation to find the top-k eigenvectors (= right singular vectors).
/// Singular values = sqrt(eigenvalues).
///
/// Sufficient for k <= 10 (our use case).
pub fn power_iteration_svd(
    matrix: &Tensor,
    k: usize,
    num_iterations: usize,
) -> Result<(Vec<f32>, Vec<Tensor>)> {
    let (_m, n) = matrix.dims2()?;
    let ata = matrix.t()?.matmul(matrix)?; // [n, n]

    let mut singular_values = Vec::new();
    let mut right_vectors = Vec::new();
    let mut deflated = ata.clone();

    for _ in 0..k {
        // Random init
        let mut v = Tensor::randn(0f32, 1f32, &[n], matrix.device())?;

        // Power iteration on the deflated A^T A
        for _ in 0..num_iterations {
            let av = deflated.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
            let norm = av.sqr()?.sum_all()?.sqrt()?;
            let norm_scalar: f32 = norm.to_scalar()?;
            if norm_scalar < 1e-12 {
                // Converged to zero -- no more directions
                return Ok((singular_values, right_vectors));
            }
            v = av.broadcast_div(&norm)?;
        }

        // Eigenvalue = v^T @ A^T A @ v
        let av = deflated.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
        let eigenvalue: f32 = v
            .unsqueeze(0)?
            .matmul(&av.unsqueeze(1)?)?
            .squeeze(0)?
            .squeeze(0)?
            .to_scalar()?;
        let sv = eigenvalue.abs().sqrt();

        if sv < 1e-8 {
            break; // No more significant directions
        }

        singular_values.push(sv);
        right_vectors.push(v.clone());

        // Deflate: remove this component from A^T A
        // deflated = deflated - eigenvalue * v @ v^T
        let vvt = v.unsqueeze(1)?.matmul(&v.unsqueeze(0)?)?; // [n, n]
        let scaled_vvt = vvt.broadcast_mul(
            &Tensor::new(eigenvalue.abs(), matrix.device())?.reshape(&[1, 1])?,
        )?;
        deflated = (&deflated - &scaled_vvt)?;
    }

    Ok((singular_values, right_vectors))
}
