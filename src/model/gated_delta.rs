// Gated DeltaNet (GDN) linear attention layer for Qwen3.5.
//
// Implements the delta rule recurrence with gated decay:
//   g = exp(-exp(A_log) * softplus(a + dt_bias))
//   beta = sigmoid(b)
//   state = g * state + k * ((v - state @ k) * beta)
//   y = state @ q
//
// Reference: mlx_lm/models/gated_delta.py, mlx_lm/models/qwen3_5.py

use candle_core::{DType, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder, linear_no_bias};

use super::ops::{softplus, rms_normalize, RmsNormGated};
use super::qwen35_config::Qwen35Config;

/// Recurrent state for a single GDN layer: [B, Hv, Dv, Dk].
/// `None` means uninitialized (will be zero-filled on first use).
pub type GdnState = Option<Tensor>;

/// Gated DeltaNet linear attention layer.
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d: Conv1d,
    norm: RmsNormGated,
    pub out_proj: Linear,
    a_log: Tensor,  // [Hv], float32
    dt_bias: Tensor, // [Hv]
    // Dimensions
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel_size: usize,
}

impl GatedDeltaNet {
    pub fn new(config: &Qwen35Config, vb: VarBuilder) -> anyhow::Result<Self> {
        let num_k_heads = config.linear_num_key_heads;
        let num_v_heads = config.linear_num_value_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let hidden_size = config.hidden_size;

        let in_proj_qkv = linear_no_bias(hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_a"))?;

        // Depthwise conv1d: groups = conv_dim, kernel_size = 4
        // Weight shape from HF: [conv_dim, 1, kernel_size]
        let conv_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: conv_dim,
            cudnn_fwd_algo: None,
        };
        let conv_weight = vb.pp("conv1d").get((conv_dim, 1, config.linear_conv_kernel_dim), "weight")?;
        let conv1d = Conv1d::new(conv_weight, None, conv_cfg);

        let norm = RmsNormGated::new(head_v_dim, config.rms_norm_eps, true, vb.pp("norm"))?;
        let out_proj = linear_no_bias(value_dim, hidden_size, vb.pp("out_proj"))?;

        // A_log and dt_bias are 1D parameters, not linear layers
        let a_log = vb.get(num_v_heads, "A_log")?.to_dtype(DType::F32)?;
        let dt_bias = vb.get(num_v_heads, "dt_bias")?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            norm,
            out_proj,
            a_log,
            dt_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: config.linear_conv_kernel_dim,
        })
    }

    /// Forward pass with optional recurrent state.
    ///
    /// `conv_state`: previous conv1d buffer [B, kernel-1, conv_dim], or None.
    /// `recurrent_state`: previous recurrent state [B, Hv, Dv, Dk], or None.
    /// `mask`: optional boolean mask [B, S] (true = valid token).
    ///
    /// Returns (output, new_conv_state, new_recurrent_state).
    pub fn forward(
        &self,
        xs: &Tensor,
        conv_state: Option<&Tensor>,
        recurrent_state: GdnState,
        _mask: Option<&Tensor>,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = xs.dims3()?;

        // Projections
        let qkv = self.in_proj_qkv.forward(xs)?; // [B, S, conv_dim]
        let z = self.in_proj_z.forward(xs)?
            .reshape((b, s, self.num_v_heads, self.head_v_dim))?; // [B, S, Hv, Dv]
        let beta_raw = self.in_proj_b.forward(xs)?; // [B, S, Hv]
        let alpha_raw = self.in_proj_a.forward(xs)?; // [B, S, Hv]

        // Conv1d with state management
        let conv_prefix = match conv_state {
            Some(st) => st.clone(),
            None => Tensor::zeros(
                (b, self.conv_kernel_size - 1, self.conv_dim),
                xs.dtype(),
                xs.device(),
            )?,
        };

        let (conv_out, new_conv_state) = if s == 1 {
            // Fast S=1 decode path: shift conv state buffer, apply depthwise dot product
            // New state = [old_state[:,1:,:], qkv] (shift left, append new token)
            let shifted = conv_prefix.narrow(1, 1, self.conv_kernel_size - 2)?;
            let new_conv_state = Tensor::cat(&[&shifted, &qkv], 1)?; // [B, K-1, conv_dim]

            // Full window = [conv_prefix, qkv] = [B, K, conv_dim]
            let window = Tensor::cat(&[&conv_prefix, &qkv], 1)?;

            // Depthwise dot product: sum(window * conv_weight, dim=time) for each channel
            // conv_weight: [conv_dim, 1, K] -> squeeze to [conv_dim, K] -> transpose [K, conv_dim]
            let weight = self.conv1d.weight().squeeze(1)?.transpose(0, 1)?; // [K, conv_dim]
            // window: [B, K, conv_dim] * weight: [K, conv_dim] -> sum over K
            let conv_out = window.broadcast_mul(&weight.unsqueeze(0)?)?.sum(1)?; // [B, conv_dim]
            let conv_out = conv_out.silu()?.unsqueeze(1)?; // [B, 1, conv_dim]

            (conv_out, new_conv_state)
        } else {
            // General prefill path: full grouped conv1d
            let conv_input = Tensor::cat(&[&conv_prefix, &qkv], 1)?; // [B, S+K-1, conv_dim]
            let new_conv_state = conv_input.narrow(
                1,
                conv_input.dim(1)? - (self.conv_kernel_size - 1),
                self.conv_kernel_size - 1,
            )?;
            let conv_in_t = conv_input.transpose(1, 2)?; // [B, conv_dim, S+K-1]
            let conv_out_t = self.conv1d.forward(&conv_in_t)?; // [B, conv_dim, S]
            let conv_out = conv_out_t.transpose(1, 2)?.silu()?; // [B, S, conv_dim]
            (conv_out, new_conv_state)
        };

        // Split into Q, K, V
        let q = conv_out.narrow(D::Minus1, 0, self.key_dim)?
            .reshape((b, s, self.num_k_heads, self.head_k_dim))?;
        let k = conv_out.narrow(D::Minus1, self.key_dim, self.key_dim)?
            .reshape((b, s, self.num_k_heads, self.head_k_dim))?;
        let v = conv_out.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?
            .reshape((b, s, self.num_v_heads, self.head_v_dim))?;

        // RMS-normalize Q and K with scaling (matches mx.fast.rms_norm)
        // q = (1/dk) * rms_norm(q), k = (1/sqrt(dk)) * rms_norm(k)
        let inv_scale = (self.head_k_dim as f64).powf(-0.5);
        let q = (rms_normalize(&q, 1e-6)? * (inv_scale * inv_scale))?;
        let k = (rms_normalize(&k, 1e-6)? * inv_scale)?;

        // Compute gating: g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
        // Only exp(A_log) needs F32 precision; the rest stays in native dtype.
        let dt_bias_expanded = self.dt_bias.to_dtype(alpha_raw.dtype())?;
        let a_plus_bias = alpha_raw.broadcast_add(&dt_bias_expanded)?;
        let sp = softplus(&a_plus_bias)?;
        let exp_a_log = self.a_log.exp()?.to_dtype(sp.dtype())?; // F32 -> model dtype once
        let decay_arg = exp_a_log.broadcast_mul(&sp)?;
        let g = decay_arg.neg()?.exp()?; // [B, S, Hv]
        let beta = candle_nn::ops::sigmoid(&beta_raw)?; // [B, S, Hv]

        // Delta rule recurrence (ops-based, sequential over time steps)
        let new_state = self.delta_recurrence(&q, &k, &v, &g, &beta, recurrent_state)?;

        // Output: norm(out, gate=z) then out_proj
        let (out, final_state) = &new_state;
        let normed = self.norm.forward_gated(out, &z)?;
        let flat = normed.reshape((b, s, self.value_dim))?;
        let output = self.out_proj.forward(&flat)?;

        Ok((output, new_conv_state, final_state.clone()))
    }

    /// Delta rule recurrence. Dispatches to fast S=1 path for decode.
    /// Returns (y: [B, S, Hv, Dv], final_state: [B, Hv, Dv, Dk]).
    fn delta_recurrence(
        &self,
        q: &Tensor,   // [B, S, Hk, Dk]
        k: &Tensor,   // [B, S, Hk, Dk]
        v: &Tensor,   // [B, S, Hv, Dv]
        g: &Tensor,   // [B, S, Hv]
        beta: &Tensor, // [B, S, Hv]
        state: GdnState,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let (b, s, _hk, dk) = q.dims4()?;
        let hv = self.num_v_heads;
        let dv = self.head_v_dim;
        let hk = self.num_k_heads;

        let mut state = match state {
            Some(st) => st,
            None => Tensor::zeros((b, hv, dv, dk), q.dtype(), q.device())?,
        };

        let repeat_factor = hv / hk;

        // Fast path for decode (S=1): no loop, use matmul instead of broadcast+sum
        if s == 1 {
            let q_t = q.squeeze(1)?; // [B, Hk, Dk]
            let k_t = k.squeeze(1)?;
            let v_t = v.squeeze(1)?; // [B, Hv, Dv]
            let g_t = g.squeeze(1)?; // [B, Hv]
            let beta_t = beta.squeeze(1)?;

            // Repeat-interleave K heads
            let (q_t, k_t) = if repeat_factor > 1 {
                let q_r = q_t.unsqueeze(2)?
                    .expand((b, hk, repeat_factor, dk))?
                    .reshape((b, hv, dk))?;
                let k_r = k_t.unsqueeze(2)?
                    .expand((b, hk, repeat_factor, dk))?
                    .reshape((b, hv, dk))?;
                (q_r, k_r)
            } else {
                (q_t, k_t)
            };

            // Decay: state = g * state
            let g_expanded = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // [B, Hv, 1, 1]
            state = state.broadcast_mul(&g_expanded)?;

            // kv_mem = state @ k^T  (batched matmul over heads)
            // state: [B, Hv, Dv, Dk], k: [B, Hv, Dk] -> k: [B, Hv, Dk, 1]
            let k_col = k_t.unsqueeze(D::Minus1)?; // [B, Hv, Dk, 1]
            let kv_mem = state.matmul(&k_col)?.squeeze(D::Minus1)?; // [B, Hv, Dv]

            // delta = beta * (v - kv_mem)
            let beta_expanded = beta_t.unsqueeze(D::Minus1)?; // [B, Hv, 1]
            let delta = (v_t - kv_mem)?.broadcast_mul(&beta_expanded)?; // [B, Hv, Dv]

            // state += delta @ k  (outer product update)
            // delta: [B, Hv, Dv, 1], k: [B, Hv, 1, Dk]
            let delta_col = delta.unsqueeze(D::Minus1)?; // [B, Hv, Dv, 1]
            let k_row = k_t.unsqueeze(2)?; // [B, Hv, 1, Dk]
            state = (state + delta_col.matmul(&k_row)?)?;

            // y = state @ q  (batched matmul)
            let q_col = q_t.unsqueeze(D::Minus1)?; // [B, Hv, Dk, 1]
            let y_t = state.matmul(&q_col)?.squeeze(D::Minus1)?; // [B, Hv, Dv]

            return Ok((y_t.unsqueeze(1)?, state));
        }

        // General path for prefill (S > 1): sequential loop
        let mut outputs = Vec::with_capacity(s);

        for t in 0..s {
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?;
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?;
            let g_t = g.narrow(1, t, 1)?.squeeze(1)?;
            let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;

            let (q_t, k_t) = if repeat_factor > 1 {
                let q_r = q_t.unsqueeze(2)?
                    .expand((b, hk, repeat_factor, dk))?
                    .reshape((b, hv, dk))?;
                let k_r = k_t.unsqueeze(2)?
                    .expand((b, hk, repeat_factor, dk))?
                    .reshape((b, hv, dk))?;
                (q_r, k_r)
            } else {
                (q_t, k_t)
            };

            let g_expanded = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            state = state.broadcast_mul(&g_expanded)?;

            let k_col = k_t.unsqueeze(D::Minus1)?;
            let kv_mem = state.matmul(&k_col)?.squeeze(D::Minus1)?;

            let beta_expanded = beta_t.unsqueeze(D::Minus1)?;
            let delta = (v_t - kv_mem)?.broadcast_mul(&beta_expanded)?;

            let delta_col = delta.unsqueeze(D::Minus1)?;
            let k_row = k_t.unsqueeze(2)?;
            state = (state + delta_col.matmul(&k_row)?)?;

            let q_col = q_t.unsqueeze(D::Minus1)?;
            let y_t = state.matmul(&q_col)?.squeeze(D::Minus1)?;

            outputs.push(y_t.unsqueeze(1)?);
        }

        let y = Tensor::cat(&outputs, 1)?;
        Ok((y, state))
    }
}
