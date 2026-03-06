// Activation capture and steering for Qwen3.5 hybrid GDN/attention model.
//
// Hook points:
//   TokenMixerOut   — after out_proj/o_proj, before residual add
//   ResidualPostMlp — after MLP + residual add (layer output)
//
// Capture stores F32 vectors at selected positions.
// Steering subtracts lambda * direction at hook points during forward pass.

use anyhow::Result;
use candle_core::{DType, Tensor};
use std::fmt;

// ---------------------------------------------------------------------------
// Hook points and capture storage
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookPoint {
    TokenMixerOut,
    ResidualPostMlp,
}

impl fmt::Display for HookPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HookPoint::TokenMixerOut => write!(f, "TMO"),
            HookPoint::ResidualPostMlp => write!(f, "RPM"),
        }
    }
}

/// Captured activations for a single layer at a single token position.
#[derive(Clone, Debug, Default)]
pub struct LayerCapture {
    pub token_mixer_out: Option<Vec<f32>>,   // [hidden_size]
    pub residual_post_mlp: Option<Vec<f32>>, // [hidden_size]
}

/// Captured activations at a single token position across all layers.
#[derive(Clone, Debug)]
pub struct TokenCapture {
    pub abs_pos: usize,
    pub decode_step: Option<usize>, // None for prefill
    pub layers: Vec<LayerCapture>,
}

/// Full capture trace for a forward pass session.
#[derive(Clone, Debug, Default)]
pub struct CaptureTrace {
    pub prefill: Vec<TokenCapture>,
    pub decode: Vec<TokenCapture>,
}

// ---------------------------------------------------------------------------
// Steering specification
// ---------------------------------------------------------------------------

/// Per-layer steering: subtract lambda * direction at a hook point.
#[derive(Clone, Debug)]
pub struct LayerSteerSpec {
    pub point: HookPoint,
    pub direction: Tensor, // [hidden_size], unit norm, on device
    pub lambda: f32,
}

/// Full steering plan across layers.
#[derive(Clone, Debug)]
pub struct SteeringPlan {
    pub per_layer: Vec<Option<LayerSteerSpec>>, // len = num_layers
    pub apply_prefill: bool,
    pub apply_decode: bool,
    /// Scale factor for prefill steering (0.0-1.0). Decode uses full lambda.
    /// Literature suggests weaker prefill (0.2-0.4) + full decode for thinking models.
    pub prefill_scale: f32,
    /// Only steer the last K tokens during prefill (None = steer all).
    /// Avoids GDN state compounding from early-token perturbations.
    pub prefill_last_k: Option<usize>,
}

impl SteeringPlan {
    /// Create a plan that steers all layers at TokenMixerOut with the same direction.
    pub fn uniform(
        direction: &Tensor,
        lambda: f32,
        num_layers: usize,
    ) -> Result<Self> {
        let spec = LayerSteerSpec {
            point: HookPoint::TokenMixerOut,
            direction: direction.clone(),
            lambda,
        };
        Ok(Self {
            per_layer: vec![Some(spec); num_layers],
            apply_prefill: false,
            apply_decode: true,
            prefill_scale: 1.0,
            prefill_last_k: None,
        })
    }

    /// Create a plan that only steers specific layer indices.
    pub fn selective(
        direction: &Tensor,
        lambda: f32,
        num_layers: usize,
        layer_indices: &[usize],
        point: HookPoint,
    ) -> Result<Self> {
        let mut per_layer = vec![None; num_layers];
        for &idx in layer_indices {
            per_layer[idx] = Some(LayerSteerSpec {
                point,
                direction: direction.clone(),
                lambda,
            });
        }
        Ok(Self {
            per_layer,
            apply_prefill: false,
            apply_decode: true,
            prefill_scale: 1.0,
            prefill_last_k: None,
        })
    }

    /// Create a weighted multi-layer steering plan from per-layer directions and magnitudes.
    ///
    /// Uses softmax-temperature weighting: alpha_l = lambda * softmax(magnitude_l / tau)
    /// so layers with stronger refusal signal get proportionally more steering.
    /// Per-layer lambda is capped at `cap_factor * lambda` to prevent over-concentration.
    pub fn weighted(
        directions: &[Tensor],    // per-layer direction vectors
        magnitudes: &[f32],       // per-layer raw magnitudes
        lambda: f32,              // global steering strength
        tau: f32,                 // softmax temperature (lower = sharper focus on peak layers)
        layer_range: std::ops::Range<usize>,
        num_layers: usize,
        point: HookPoint,
        prefill: bool,
        prefill_scale: f32,
        prefill_last_k: Option<usize>,
    ) -> Result<Self> {
        // Compute softmax weights over the selected layer range
        let range_len = layer_range.len();
        let scores: Vec<f32> = layer_range.clone().map(|i| magnitudes[i] / tau).collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Cap per-layer lambda at 2.5x the uniform rate to prevent over-concentration
        let cap = 2.5 * lambda;

        let mut per_layer = vec![None; num_layers];
        for (i, layer_idx) in layer_range.enumerate() {
            let layer_lambda = (lambda * weights[i] * (range_len as f32)).min(cap);
            per_layer[layer_idx] = Some(LayerSteerSpec {
                point,
                direction: directions[layer_idx].clone(),
                lambda: layer_lambda,
            });
        }

        Ok(Self {
            per_layer,
            apply_prefill: prefill,
            apply_decode: true,
            prefill_scale,
            prefill_last_k,
        })
    }

    /// Blended weighting: mix uniform with softmax-weighted to prevent collapse.
    ///
    /// w_l = (1-rho)/N + rho * softmax(score_l / tau)
    /// where score_l = log(magnitude_l + eps) to compress monotonic RPM profiles.
    /// Per-layer lambda clamped to [0.5*lambda, cap_factor*lambda].
    pub fn blended(
        directions: &[Tensor],
        magnitudes: &[f32],
        lambda: f32,
        tau: f32,
        rho: f32,               // blend factor: 0=uniform, 1=pure softmax
        layer_range: std::ops::Range<usize>,
        num_layers: usize,
        point: HookPoint,
        prefill: bool,
        prefill_scale: f32,
        prefill_last_k: Option<usize>,
    ) -> Result<Self> {
        let range_len = layer_range.len();
        let n = range_len as f32;

        // Log-compress scores to prevent monotonic magnitude from dominating
        let scores: Vec<f32> = layer_range.clone()
            .map(|i| (magnitudes[i] + 1e-6).ln() / tau)
            .collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let softmax_w: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Blend: uniform floor + softmax signal
        let uniform_w = 1.0 / n;
        let weights: Vec<f32> = softmax_w.iter()
            .map(|&sw| (1.0 - rho) * uniform_w + rho * sw)
            .collect();

        // Clamp per-layer lambda to [0.5*lambda, 2.0*lambda]
        let floor = 0.5 * lambda;
        let cap = 2.0 * lambda;

        let mut per_layer = vec![None; num_layers];
        for (i, layer_idx) in layer_range.enumerate() {
            let layer_lambda = (lambda * weights[i] * n).clamp(floor, cap);
            per_layer[layer_idx] = Some(LayerSteerSpec {
                point,
                direction: directions[layer_idx].clone(),
                lambda: layer_lambda,
            });
        }

        Ok(Self {
            per_layer,
            apply_prefill: prefill,
            apply_decode: true,
            prefill_scale,
            prefill_last_k,
        })
    }

    /// Uniform weighting across a layer range with per-layer directions.
    ///
    /// Each layer in the range gets the same lambda.
    pub fn uniform_range(
        directions: &[Tensor],
        lambda: f32,
        layer_range: std::ops::Range<usize>,
        num_layers: usize,
        point: HookPoint,
        prefill: bool,
        prefill_scale: f32,
        prefill_last_k: Option<usize>,
    ) -> Result<Self> {
        let mut per_layer = vec![None; num_layers];
        for layer_idx in layer_range {
            per_layer[layer_idx] = Some(LayerSteerSpec {
                point,
                direction: directions[layer_idx].clone(),
                lambda,
            });
        }
        Ok(Self {
            per_layer,
            apply_prefill: prefill,
            apply_decode: true,
            prefill_scale,
            prefill_last_k,
        })
    }
}

// ---------------------------------------------------------------------------
// Runtime context threaded through forward pass
// ---------------------------------------------------------------------------

/// Runtime context for capture and steering during forward pass.
pub struct RuntimeCtx<'a> {
    pub capture: Option<&'a mut CaptureTrace>,
    pub steering: Option<&'a SteeringPlan>,
    pub is_prefill: bool,
    pub decode_step: usize,
    pub seq_offset: usize, // absolute position of first token in current input
    // Which token position to capture (relative to current input)
    // For prefill: last token. For decode: 0 (only token).
    pub capture_pos: Option<usize>,
}

impl<'a> RuntimeCtx<'a> {
    /// Create a no-op context (no capture, no steering).
    pub fn noop() -> Self {
        Self {
            capture: None,
            steering: None,
            is_prefill: true,
            decode_step: 0,
            seq_offset: 0,
            capture_pos: None,
        }
    }

    /// Try to capture an activation at a hook point for a given layer.
    pub fn maybe_capture(
        &mut self,
        layer_idx: usize,
        point: HookPoint,
        xs: &Tensor, // [1, S, hidden_size]
        num_layers: usize,
    ) -> Result<()> {
        let Some(trace) = self.capture.as_deref_mut() else {
            return Ok(());
        };
        let Some(cap_pos) = self.capture_pos else {
            return Ok(());
        };

        // Extract the activation at the capture position
        let vec = xs
            .narrow(1, cap_pos, 1)?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        // Get or create the TokenCapture for this position
        let captures = if self.is_prefill {
            &mut trace.prefill
        } else {
            &mut trace.decode
        };

        // Find existing capture for this position or create new one
        let abs_pos = self.seq_offset + cap_pos;
        let token_cap = if let Some(tc) = captures.last_mut().filter(|tc| tc.abs_pos == abs_pos) {
            tc
        } else {
            captures.push(TokenCapture {
                abs_pos,
                decode_step: if self.is_prefill {
                    None
                } else {
                    Some(self.decode_step)
                },
                layers: vec![LayerCapture::default(); num_layers],
            });
            captures.last_mut().unwrap()
        };

        match point {
            HookPoint::TokenMixerOut => {
                token_cap.layers[layer_idx].token_mixer_out = Some(vec);
            }
            HookPoint::ResidualPostMlp => {
                token_cap.layers[layer_idx].residual_post_mlp = Some(vec);
            }
        }
        Ok(())
    }

    /// Apply steering to a tensor at a hook point for a given layer.
    pub fn maybe_steer(
        &self,
        layer_idx: usize,
        point: HookPoint,
        xs: Tensor, // [1, S, hidden_size]
    ) -> candle_core::Result<Tensor> {
        let Some(plan) = &self.steering else {
            return Ok(xs);
        };
        if self.is_prefill && !plan.apply_prefill {
            return Ok(xs);
        }
        if !self.is_prefill && !plan.apply_decode {
            return Ok(xs);
        }
        let Some(spec) = plan.per_layer.get(layer_idx).and_then(|s| s.as_ref()) else {
            return Ok(xs);
        };
        if spec.point != point {
            return Ok(xs);
        }

        // x - scale * lambda * direction
        let scale = if self.is_prefill { plan.prefill_scale } else { 1.0 };
        let effective_lambda = spec.lambda * scale;
        if effective_lambda.abs() < 1e-8 {
            return Ok(xs);
        }

        let dir = spec
            .direction
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(xs.dtype())?;
        let lambda_t = Tensor::new(&[effective_lambda], xs.device())?.to_dtype(xs.dtype())?;
        let delta = dir.broadcast_mul(&lambda_t)?;

        // During prefill with last-K restriction, only steer the last K tokens
        if self.is_prefill {
            if let Some(k) = plan.prefill_last_k {
                let seq_len = xs.dim(1)?;
                if k < seq_len {
                    let pass = xs.narrow(1, 0, seq_len - k)?;
                    let steer = xs.narrow(1, seq_len - k, k)?;
                    let steered = steer.broadcast_sub(&delta)?;
                    return Tensor::cat(&[pass, steered], 1);
                }
            }
        }

        xs.broadcast_sub(&delta)
    }
}
