//! Custom ops for Qwen3.5 GDN model that candle doesn't provide.

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

// ---------------------------------------------------------------------------
// RmsNormGated: (1 + weight) * rms_norm(x), with optional SiLU gate
// ---------------------------------------------------------------------------

/// RMS normalization with optional SiLU gating.
///
/// Qwen3.5 GDN uses this as: rms_norm(hidden_states) * silu(gate)
/// where hidden_states and gate are separate tensors.
///
/// The norm weight here is NOT shifted (unlike the decoder layernorms).
/// HF stores the GDN norm weight as-is; only standard RMSNorm weights get +1 shift.
pub struct RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    pub fn new(hidden_size: usize, eps: f64, _gated: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps })
    }

    /// Forward with optional gate: rms_norm(xs, weight) * silu(gate).
    pub fn forward_gated(&self, xs: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let normed = rms_norm(xs, &self.weight, self.eps)?;
        let gate_activated = gate.silu()?;
        gate_activated.mul(&normed)
    }

    /// Forward without gating.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        rms_norm(xs, &self.weight, self.eps)
    }
}

/// Standard RMS normalization: weight * x / rms(x).
/// Uses candle's fused Metal kernel — no F32 round-trip.
fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // Ensure weight matches input dtype (GDN norm weight may be F32 while model runs BF16)
    let weight = weight.to_dtype(xs.dtype())?;
    candle_nn::ops::rms_norm(xs, &weight, eps as f32)
}

// ---------------------------------------------------------------------------
// Softplus: log(1 + exp(x))
// ---------------------------------------------------------------------------

/// Numerically stable softplus: log(1 + exp(x)).
/// Uses branch-free form: relu(x) + log(1 + exp(-|x|)) to avoid NaN from overflow.
/// Stays in native dtype (BF16/F16/F32) — no round-trip.
pub fn softplus(xs: &Tensor) -> Result<Tensor> {
    // softplus(x) = relu(x) + log(1 + exp(-|x|))
    // exp(-|x|) is always <= 1, so no overflow risk.
    let abs_x = xs.abs()?;
    let relu_x = xs.relu()?;
    let exp_neg_abs = abs_x.neg()?.exp()?;
    relu_x + (exp_neg_abs + 1.0)?.log()?
}

// ---------------------------------------------------------------------------
// L2 normalize
// ---------------------------------------------------------------------------

/// RMS-normalize along the last dimension (no learnable weight).
/// Computes: x / sqrt(mean(x^2) + eps).
/// Uses candle's fused Metal kernel with unit weight.
pub fn rms_normalize(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let weight = Tensor::ones(last_dim, xs.dtype(), xs.device())?;
    candle_nn::ops::rms_norm(xs, &weight, eps as f32)
}

// ---------------------------------------------------------------------------
// Partial RoPE
// ---------------------------------------------------------------------------

/// Rotary position embeddings applied to only the first `rotary_dim` dimensions.
/// Remaining dimensions pass through unchanged.
pub struct PartialRotaryEmbedding {
    pub cos: Tensor,
    pub sin: Tensor,
    pub rotary_dim: usize,
}

impl PartialRotaryEmbedding {
    /// Create rotary embeddings for `max_seq_len` positions.
    /// `rotary_dim` = head_dim * partial_rotary_factor (e.g., 256 * 0.25 = 64).
    pub fn new(
        rotary_dim: usize,
        max_seq_len: usize,
        theta: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let half = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / rotary_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?; // [half]
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?; // [seq, 1]
        let inv_freq = inv_freq.unsqueeze(0)?; // [1, half]
        let freqs = positions.matmul(&inv_freq)?; // [seq, half]
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self {
            cos,
            sin,
            rotary_dim,
        })
    }

    /// Apply partial RoPE to Q or K tensor.
    /// Input shape: [batch, heads, seq_len, head_dim].
    /// `offset` is the position offset for KV-cache continuation.
    pub fn apply(&self, xs: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, _seq_len, head_dim) = xs.dims4()?;
        if self.rotary_dim >= head_dim {
            // Full RoPE
            return self.apply_rotary(xs, offset);
        }
        // Split: first rotary_dim dims get RoPE, rest pass through
        let x_rot = xs.narrow(D::Minus1, 0, self.rotary_dim)?;
        let x_pass = xs.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        let x_rot = self.apply_rotary(&x_rot, offset)?;
        Tensor::cat(&[x_rot, x_pass], D::Minus1)
    }

    fn apply_rotary(&self, xs: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, seq_len, dim) = xs.dims4()?;
        let half = dim / 2;
        // Get cos/sin for the right positions
        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, half]
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let cos = cos.to_dtype(xs.dtype())?;
        let sin = sin.to_dtype(xs.dtype())?;
        // Split into halves and rotate
        let x1 = xs.narrow(D::Minus1, 0, half)?;
        let x2 = xs.narrow(D::Minus1, half, half)?;
        let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[rotated_x1, rotated_x2], D::Minus1)
    }
}
