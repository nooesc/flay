// Qwen3 MoE model definition for candle
// Implements inference with expert routing + residual activation capture
//
// This is a purpose-built implementation for abliteration: single-pass prefill only,
// no KV caching, no autoregressive generation. The key addition over a standard
// implementation is per-expert activation capture in the MoE layers.

use candle_core::{DType, Device, Module, Result, Tensor, D};

use crate::abliterate::weight_key::WeightKey;
use crate::model::arch::{ExpertMask, MoeLayerCapture, MoeModel, ModelOutput};
use candle_nn::{
    embedding, linear_b, linear_no_bias, ops::softmax_last_dim, rms_norm, Embedding, Linear,
    RmsNorm, VarBuilder,
};

use super::config::Qwen3MoeConfig;

// ---------------------------------------------------------------------------
// 1. Expert MLP
// ---------------------------------------------------------------------------

/// A single expert MLP: gate/up projections with SiLU gating, then down projection.
pub struct Expert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Expert {
    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        vb: VarBuilder,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, moe_intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, moe_intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(moe_intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }
}

impl Expert {
    /// Swap the down_proj weight matrix in-place, returning the old weight.
    pub fn swap_down_proj(&mut self, new_weight: Tensor) -> Tensor {
        let old = self.down_proj.weight().clone();
        self.down_proj = Linear::new(new_weight, None);
        old
    }

    /// Get a reference to the down_proj weight tensor.
    pub fn down_proj_weight(&self) -> &Tensor {
        self.down_proj.weight()
    }
}

impl Module for Expert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        let fused = (gate * up)?;
        self.down_proj.forward(&fused)
    }
}

// ---------------------------------------------------------------------------
// 2. SparseMoeBlock
// ---------------------------------------------------------------------------

/// Sparse Mixture-of-Experts block with optional activation capture.
pub struct SparseMoeBlock {
    gate: Linear,
    experts: Vec<Expert>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let gate = linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?;
        let vb_e = vb.pp("experts");
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for idx in 0..cfg.num_experts {
            experts.push(Expert::new(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                vb_e.pp(idx),
            )?);
        }
        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    /// Forward pass with optional per-expert activation capture.
    ///
    /// When `capture` is true, routing decisions and individual expert outputs
    /// are recorded in the returned `MoeLayerCapture`. Only the last token
    /// position's expert outputs are captured to avoid unnecessary compute and
    /// memory on long prompts (refusal signals are strongest at the last token).
    pub fn forward_with_capture(
        &self,
        xs: &Tensor,
        capture: bool,
        masked_experts: Option<&std::collections::HashSet<usize>>,
    ) -> Result<(Tensor, Option<MoeLayerCapture>)> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let n_tokens = b_size * seq_len;
        let last_token_idx = n_tokens.saturating_sub(1);

        // Router: compute gate logits
        let mut router_logits = self.gate.forward(&xs_flat)?;

        // Apply expert mask: set gate logits to -inf for masked experts
        if let Some(mask_set) = masked_experts {
            if !mask_set.is_empty() {
                let num_experts = self.experts.len();
                let n = router_logits.dim(0)?;
                let mut logit_data: Vec<f32> = router_logits
                    .to_dtype(candle_core::DType::F32)?
                    .to_vec2::<f32>()?
                    .into_iter()
                    .flatten()
                    .collect();
                for expert_idx in mask_set {
                    if *expert_idx < num_experts {
                        for batch_idx in 0..n {
                            logit_data[batch_idx * num_experts + *expert_idx] = f32::NEG_INFINITY;
                        }
                    }
                }
                router_logits = Tensor::from_vec(
                    logit_data,
                    router_logits.shape(),
                    router_logits.device(),
                )?
                .to_dtype(router_logits.dtype())?;
            }
        }

        let routing_probs = softmax_last_dim(&router_logits)?;

        // Top-k selection via descending arg_sort then narrow
        let sorted_indices = routing_probs
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let topk_weights = routing_probs.gather(&sorted_indices, D::Minus1)?;

        // Pull routing data to CPU for scatter-gather dispatch
        let topk_weights_cpu = topk_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let topk_indices_cpu = sorted_indices.to_vec2::<u32>()?;

        // Build per-expert token lists with (optionally normalized) weights
        let num_experts = self.experts.len();
        let mut expert_token_ids: Vec<Vec<u32>> = vec![vec![]; num_experts];
        let mut expert_weights: Vec<Vec<f32>> = vec![vec![]; num_experts];

        for (tok_idx, (weights, indices)) in topk_weights_cpu
            .iter()
            .zip(topk_indices_cpu.iter())
            .enumerate()
        {
            let sum_w: f32 = weights.iter().sum();
            for (&w, &expert_idx) in weights.iter().zip(indices.iter()) {
                let w = if self.norm_topk_prob { w / sum_w } else { w };
                expert_token_ids[expert_idx as usize].push(tok_idx as u32);
                expert_weights[expert_idx as usize].push(w);
            }
        }

        // Track which experts handle the last token (for capture)
        let last_token_experts: std::collections::HashSet<usize> = if capture {
            topk_indices_cpu
                .get(last_token_idx)
                .map(|indices| indices.iter().map(|&idx| idx as usize).collect())
                .unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        };

        // Process each expert and accumulate results
        let mut ys = xs_flat.zeros_like()?;
        let mut captured_outputs: Vec<Vec<(usize, Tensor)>> = if capture {
            (0..num_experts).map(|_| Vec::new()).collect()
        } else {
            Vec::new()
        };

        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let token_ids = &expert_token_ids[expert_idx];
            if token_ids.is_empty() {
                continue;
            }

            let token_ids_tensor = Tensor::new(token_ids.as_slice(), xs.device())?;
            let weights_tensor = Tensor::new(expert_weights[expert_idx].as_slice(), xs.device())?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;

            // Select tokens for this expert, run expert forward
            let expert_input = xs_flat
                .index_select(&token_ids_tensor, 0)?
                .reshape(((), hidden_dim))?;
            let expert_output = expert.forward(&expert_input)?;

            // Capture only the last token's expert output (not all tokens)
            if capture && last_token_experts.contains(&expert_idx) {
                if let Some(local_idx) = token_ids
                    .iter()
                    .position(|&tid| tid as usize == last_token_idx)
                {
                    let single_output = expert_output.narrow(0, local_idx, 1)?.squeeze(0)?;
                    captured_outputs[expert_idx].push((last_token_idx, single_output));
                }
            }

            // Weight and scatter-add back
            let weighted = expert_output.broadcast_mul(&weights_tensor)?;
            ys = ys.index_add(&token_ids_tensor, &weighted, 0)?;
        }

        let ys = ys.reshape((b_size, seq_len, hidden_dim))?;

        let moe_capture = if capture {
            let normalized_weights: Vec<Vec<f32>> = topk_weights_cpu
                .iter()
                .map(|weights| {
                    if self.norm_topk_prob {
                        let sum_w: f32 = weights.iter().sum();
                        weights.iter().map(|w| w / sum_w).collect()
                    } else {
                        weights.clone()
                    }
                })
                .collect();
            Some(MoeLayerCapture {
                expert_indices: topk_indices_cpu,
                routing_weights: normalized_weights,
                expert_outputs: captured_outputs,
            })
        } else {
            None
        };

        Ok((ys, moe_capture))
    }

    /// Get a reference to an expert's down_proj linear layer (for abliteration).
    pub fn get_expert_down_proj(&self, expert_idx: usize) -> &Linear {
        &self.experts[expert_idx].down_proj
    }

    /// Total number of experts in this MoE block.
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Get a mutable reference to an expert (for weight modification).
    pub fn expert_mut(&mut self, expert_idx: usize) -> &mut Expert {
        &mut self.experts[expert_idx]
    }

    /// Get an immutable reference to an expert.
    pub fn expert(&self, expert_idx: usize) -> &Expert {
        &self.experts[expert_idx]
    }

    /// Swap an expert's down_proj weight, returning the old weight.
    pub fn swap_expert_weight(&mut self, expert_idx: usize, new_weight: Tensor) -> Tensor {
        self.experts[expert_idx].swap_down_proj(new_weight)
    }

    /// Get a reference to the gate (router) weight tensor.
    pub fn gate_weight(&self) -> &Tensor {
        self.gate.weight()
    }

    /// Swap the gate (router) weight, returning the old weight.
    pub fn swap_gate(&mut self, new_weight: Tensor) -> Tensor {
        let old = self.gate.weight().clone();
        self.gate = Linear::new(new_weight, None);
        old
    }
}

// ---------------------------------------------------------------------------
// 4. Rotary Embedding
// ---------------------------------------------------------------------------

/// Precomputed cos/sin tables for Rotary Position Embeddings.
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> anyhow::Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;

        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    /// Apply RoPE to Q and K tensors. Both should be [B, H, S, D].
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ---------------------------------------------------------------------------
// 5. Attention
// ---------------------------------------------------------------------------

/// Multi-head attention with Qwen3-specific per-head Q/K RMSNorm and GQA.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Attention {
    /// Swap the o_proj weight, returning the old weight.
    /// IMPORTANT: Preserves existing bias (Codex review #2).
    pub fn swap_o_proj(&mut self, new_weight: Tensor) -> Tensor {
        let old = self.o_proj.weight().clone();
        let bias = self.o_proj.bias().cloned(); // preserve bias if present
        self.o_proj = Linear::new(new_weight, bias);
        old
    }

    /// Get a reference to the o_proj weight tensor.
    pub fn o_proj_weight(&self) -> &Tensor {
        self.o_proj.weight()
    }

    fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let bias = cfg.attention_bias();

        let q_proj = linear_b(cfg.hidden_size, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let hidden_size = head_dim * num_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor, rope: &RotaryEmbedding, offset: usize) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;

        // 1. Project Q/K/V
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // 2. Reshape: (B, L, H*D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per-head Q/K RMSNorm (Qwen3-specific)
        // Flatten batch*heads into first dim so RmsNorm operates on [., L, D]
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_normed = self.q_norm.forward(&q_flat)?;
        let k_normed = self.k_norm.forward(&k_flat)?;
        let q = q_normed.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_normed.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // 4. Apply RoPE
        let (q, k) = rope.apply(&q, &k, offset)?;

        // 5. GQA: repeat KV heads to match Q head count
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, num_kv_groups)?;
        let v = repeat_kv(v, num_kv_groups)?;

        // 6. Scaled dot-product attention with causal mask
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.t()?)? * scale)?;

        // Causal mask: upper triangle -> -inf
        let mask = build_causal_mask(l, scores.dtype(), xs.device())?;
        let scores = scores.broadcast_add(&mask)?;

        let probs = softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

        // 7. Output projection
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    /// Forward pass with KV-cache for autoregressive generation.
    ///
    /// `layer_cache` is `None` on first call (prefill) and `Some` on subsequent
    /// calls (decode). Updated in-place with concatenated K/V.
    fn forward_cached(
        &self,
        xs: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
        layer_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, new_len, _) = xs.dims3()?;

        // 1. Project Q/K/V for new tokens
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // 2. Reshape: (B, new_len, H*D) -> (B, H, new_len, D)
        let q = q
            .reshape((b, new_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, new_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, new_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per-head Q/K RMSNorm (Qwen3-specific, BEFORE caching)
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_normed = self.q_norm.forward(&q_flat)?;
        let k_normed = self.k_norm.forward(&k_flat)?;
        let q = q_normed.reshape((b, self.num_heads, new_len, self.head_dim))?;
        let k = k_normed.reshape((b, self.num_kv_heads, new_len, self.head_dim))?;

        // 4. Apply RoPE with offset
        let (q, k) = rope.apply(&q, &k, offset)?;

        // 5. Concatenate with cached K/V (or initialize cache)
        let (k_full, v_full) = if let Some((k_cache, v_cache)) = layer_cache.take() {
            // Concat along seq_len dim (dim 2)
            (Tensor::cat(&[&k_cache, &k], 2)?, Tensor::cat(&[&v_cache, &v], 2)?)
        } else {
            (k, v)
        };

        // Store updated cache (in KV-head space, NOT expanded)
        *layer_cache = Some((k_full.clone(), v_full.clone()));

        // 6. GQA: repeat KV heads to match Q head count
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k_expanded = repeat_kv(k_full, num_kv_groups)?;
        let v_expanded = repeat_kv(v_full, num_kv_groups)?;

        // 7. Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let total_len = k_expanded.dim(2)?;
        let scores = (q.matmul(&k_expanded.t()?)? * scale)?;

        // Causal mask: [1, 1, new_len, total_len]
        let mask = build_cached_mask(new_len, total_len, scores.dtype(), xs.device())?;
        let scores = scores.broadcast_add(&mask)?;

        let probs = softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_expanded)?;

        // 8. Output projection
        ctx.transpose(1, 2)?
            .reshape((b, new_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

/// Repeat KV heads for grouped-query attention.
/// Input: (B, num_kv_heads, L, D) -> Output: (B, num_heads, L, D)
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .expand((b, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b, num_kv_heads * n_rep, seq_len, head_dim))?;
    Ok(x)
}

/// Build a causal attention mask: upper triangle filled with -inf.
fn build_causal_mask(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| if j <= i { 0.0 } else { minf })
        })
        .collect();
    // Shape: (1, 1, seq_len, seq_len) for broadcasting over (B, H, L, L)
    Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

/// Build a causal mask for cached attention: new tokens attend to all prior + themselves.
///
/// Shape: `(1, 1, new_len, total_len)` where `total_len = cached_len + new_len`.
/// For single-token decode (`new_len=1`), this is all zeros (attend to everything).
fn build_cached_mask(
    new_len: usize,
    total_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let cache_len = total_len - new_len;
    let mask: Vec<f32> = (0..new_len)
        .flat_map(|i| {
            (0..total_len).map(move |j| {
                // New token at position i can attend to:
                // - All cached positions [0..cache_len)
                // - Itself and prior new tokens [cache_len..cache_len+i]
                if j < cache_len + i + 1 { 0.0 } else { minf }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (1, 1, new_len, total_len), device)?.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// 6. Dense MLP (for non-MoE layers)
// ---------------------------------------------------------------------------

/// Standard dense MLP used in non-MoE decoder layers.
struct DenseMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DenseMlp {
    fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }
}

impl Module for DenseMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        let fused = (gate * up)?;
        self.down_proj.forward(&fused)
    }
}

// ---------------------------------------------------------------------------
// 7. FeedForward enum
// ---------------------------------------------------------------------------

/// Dispatches between dense MLP and sparse MoE depending on the layer.
enum FeedForward {
    Dense(DenseMlp),
    Moe(SparseMoeBlock),
}

impl FeedForward {
    fn forward_with_capture(
        &self,
        xs: &Tensor,
        capture: bool,
        masked_experts: Option<&std::collections::HashSet<usize>>,
    ) -> Result<(Tensor, Option<MoeLayerCapture>)> {
        match self {
            FeedForward::Dense(mlp) => Ok((mlp.forward(xs)?, None)),
            FeedForward::Moe(moe) => moe.forward_with_capture(xs, capture, masked_experts),
        }
    }
}

// ---------------------------------------------------------------------------
// 8. DecoderLayer
// ---------------------------------------------------------------------------

/// A single transformer decoder layer with attention + feed-forward.
struct DecoderLayer {
    self_attn: Attention,
    feed_forward: FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    layer_idx: usize,
}

impl DecoderLayer {
    fn new(cfg: &Qwen3MoeConfig, layer_idx: usize, vb: VarBuilder) -> anyhow::Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;

        let feed_forward = if cfg.is_moe_layer(layer_idx) {
            FeedForward::Moe(SparseMoeBlock::new(cfg, vb.pp("mlp"))?)
        } else {
            FeedForward::Dense(DenseMlp::new(cfg, vb.pp("mlp"))?)
        };

        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
            layer_idx,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
        capture: bool,
        masked_experts: Option<&std::collections::HashSet<usize>>,
    ) -> Result<(Tensor, Option<Tensor>, Option<MoeLayerCapture>)> {
        // Pre-norm -> attention -> residual
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward(&normed, rope, offset)?;
        let xs = (xs + attn_out)?;

        // Pre-norm -> feed-forward -> residual
        let normed = self.post_attention_layernorm.forward(&xs)?;

        // Optionally capture the normed residual (input to MoE) for abliteration analysis
        let residual_capture = if capture { Some(normed.clone()) } else { None };

        let (ff_out, moe_capture) =
            self.feed_forward
                .forward_with_capture(&normed, capture, masked_experts)?;
        let xs = (xs + ff_out)?;

        Ok((xs, residual_capture, moe_capture))
    }

    /// Cached forward pass for autoregressive generation (no capture).
    fn forward_cached(
        &self,
        xs: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
        layer_cache: &mut Option<(Tensor, Tensor)>,
        masked_experts: Option<&std::collections::HashSet<usize>>,
    ) -> Result<Tensor> {
        let (out, _, _) = self.forward_cached_with_capture(
            xs, rope, offset, layer_cache, false, masked_experts,
        )?;
        Ok(out)
    }

    /// Cached forward pass with optional routing capture for decode-time analysis.
    fn forward_cached_with_capture(
        &self,
        xs: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
        layer_cache: &mut Option<(Tensor, Tensor)>,
        capture: bool,
        masked_experts: Option<&std::collections::HashSet<usize>>,
    ) -> Result<(Tensor, Option<Tensor>, Option<MoeLayerCapture>)> {
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward_cached(&normed, rope, offset, layer_cache)?;
        let xs = (xs + attn_out)?;

        let normed = self.post_attention_layernorm.forward(&xs)?;
        let residual_capture = if capture { Some(normed.clone()) } else { None };
        let (ff_out, moe_capture) = self
            .feed_forward
            .forward_with_capture(&normed, capture, masked_experts)?;
        let xs = (xs + ff_out)?;
        Ok((xs, residual_capture, moe_capture))
    }

    /// Get the MoE block, if this is a MoE layer.
    pub fn moe_block(&self) -> Option<&SparseMoeBlock> {
        match &self.feed_forward {
            FeedForward::Moe(moe) => Some(moe),
            FeedForward::Dense(_) => None,
        }
    }

    /// Get a mutable reference to the MoE block.
    pub fn moe_block_mut(&mut self) -> Option<&mut SparseMoeBlock> {
        match &mut self.feed_forward {
            FeedForward::Moe(moe) => Some(moe),
            FeedForward::Dense(_) => None,
        }
    }

    /// Get the attention module's o_proj weight.
    pub fn o_proj_weight(&self) -> &Tensor {
        self.self_attn.o_proj_weight()
    }

    /// Swap the attention module's o_proj weight, returning old.
    pub fn swap_o_proj(&mut self, new_weight: Tensor) -> Tensor {
        self.self_attn.swap_o_proj(new_weight)
    }
}

// ---------------------------------------------------------------------------
// 9. Qwen3MoeModel
// ---------------------------------------------------------------------------

/// Full Qwen3 MoE model with dual forward paths:
/// - `forward()`: Single-pass prefill with activation capture for abliteration
/// - `forward_cached()`: KV-cached autoregressive generation for eval
pub struct Qwen3MoeModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    config: Qwen3MoeConfig,
    expert_mask: ExpertMask,
}

impl Qwen3MoeModel {
    pub fn new(cfg: &Qwen3MoeConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let vb_model = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let vb_layers = vb_model.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, i, vb_layers.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let rope = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.device(),
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            config: cfg.clone(),
            expert_mask: ExpertMask::new(),
        })
    }

    /// Run a single-pass forward through the model (internal implementation).
    ///
    /// When `capture` is true, collects:
    /// - Normed residual states at each MoE layer (for computing refusal directions)
    /// - Per-expert routing decisions and raw expert outputs
    fn forward_impl(&self, input_ids: &Tensor, capture: bool) -> Result<ModelOutput> {
        let (_b, _l) = input_ids.dims2()?;

        let mut xs = self.embed_tokens.forward(input_ids)?;

        let mut residual_states = Vec::new();
        let mut moe_captures = Vec::new();

        for layer in &self.layers {
            let masked_experts = self.masked_experts_for_layer(layer.layer_idx);
            let (out, residual, moe_capture) =
                layer.forward(&xs, &self.rope, 0, capture, masked_experts.as_ref())?;
            xs = out;

            if let Some(res) = residual {
                residual_states.push((layer.layer_idx, res));
            }
            if let Some(cap) = moe_capture {
                moe_captures.push((layer.layer_idx, cap));
            }
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs)?;

        Ok(ModelOutput {
            logits,
            residual_states,
            moe_captures,
        })
    }

    /// Cached forward pass with optional routing capture for decode-time analysis.
    ///
    /// Snapshots `cache.seq_len` once at start for consistent RoPE offset,
    /// then advances the cache after all layers complete.
    ///
    /// When `capture=false`, takes the fast path (no capture allocations).
    /// When `capture=true`, collects per-layer routing indices and weights,
    /// clearing expert_outputs to minimize memory overhead.
    fn forward_cached_capture_impl(
        &self,
        input_ids: &Tensor,
        cache: &mut crate::model::arch::KVCache,
        capture: bool,
    ) -> Result<ModelOutput> {
        let (b, new_len) = input_ids.dims2()?;

        // Guard: batch size must be 1 for cached generation
        if b != 1 {
            candle_core::bail!("forward_cached requires batch_size=1, got {b}");
        }
        // Guard: cache must match model layer count
        if cache.entries.len() != self.layers.len() {
            candle_core::bail!(
                "KVCache has {} entries but model has {} layers",
                cache.entries.len(),
                self.layers.len(),
            );
        }

        let offset = cache.seq_len();

        let mut xs = self.embed_tokens.forward(input_ids)?;
        let mut moe_captures = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let masked_experts = self.masked_experts_for_layer(layer.layer_idx);
            if capture {
                let (out, _residual, moe_capture) = layer.forward_cached_with_capture(
                    &xs, &self.rope, offset, &mut cache.entries[i],
                    true, masked_experts.as_ref(),
                )?;
                xs = out;
                if let Some(mut cap) = moe_capture {
                    // Clear expert_outputs to save memory — we only need routing indices/weights
                    cap.expert_outputs.clear();
                    moe_captures.push((layer.layer_idx, cap));
                }
            } else {
                xs = layer.forward_cached(
                    &xs, &self.rope, offset, &mut cache.entries[i],
                    masked_experts.as_ref(),
                )?;
            }
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs)?;
        cache.advance(new_len);

        Ok(ModelOutput { logits, residual_states: vec![], moe_captures })
    }

    /// Get the set of masked expert indices for a specific decoder layer, if any.
    fn masked_experts_for_layer(&self, layer_idx: usize) -> Option<std::collections::HashSet<usize>> {
        if self.expert_mask.is_empty() {
            return None;
        }
        let experts = self.expert_mask.experts_in_layer(layer_idx);
        if experts.is_empty() { None } else { Some(experts) }
    }

    /// Access the model configuration.
    pub fn config(&self) -> &Qwen3MoeConfig {
        &self.config
    }

    /// Get the MoE block for a given layer index, if it is an MoE layer.
    pub fn moe_block(&self, layer_idx: usize) -> Option<&SparseMoeBlock> {
        self.layers.get(layer_idx).and_then(|l| l.moe_block())
    }

    /// Get a mutable MoE block for a given layer index.
    pub fn moe_block_mut(&mut self, layer_idx: usize) -> Option<&mut SparseMoeBlock> {
        self.layers.get_mut(layer_idx).and_then(|l| l.moe_block_mut())
    }

    /// Total number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Swap an expert's down_proj weight in-place, returning the old weight.
    ///
    /// `layer_idx` is the decoder layer index (not MoE position).
    /// Returns an error if the layer is not a MoE layer or the expert index is out of bounds.
    fn swap_expert_weight_impl(
        &mut self,
        layer_idx: usize,
        expert_idx: usize,
        new_weight: Tensor,
    ) -> anyhow::Result<Tensor> {
        let moe = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("Layer index {layer_idx} out of bounds"))?
            .moe_block_mut()
            .ok_or_else(|| anyhow::anyhow!("Layer {layer_idx} is not a MoE layer"))?;
        Ok(moe.swap_expert_weight(expert_idx, new_weight))
    }
}

impl MoeModel for Qwen3MoeModel {
    fn forward(&self, input_ids: &Tensor, capture: bool) -> anyhow::Result<ModelOutput> {
        self.forward_impl(input_ids, capture).map_err(Into::into)
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    fn moe_layer_indices(&self) -> Vec<usize> {
        (0..self.num_layers())
            .filter(|&i| self.config.is_moe_layer(i))
            .collect()
    }

    fn weight_key(&self, layer_idx: usize, expert_idx: usize) -> String {
        format!(
            "model.layers.{}.mlp.experts.{}.down_proj.weight",
            layer_idx, expert_idx,
        )
    }

    fn swap_expert_weight(
        &mut self,
        layer_idx: usize,
        expert_idx: usize,
        new_weight: &Tensor,
    ) -> anyhow::Result<Tensor> {
        self.set_weight(
            &WeightKey::MoeDownProj {
                layer: layer_idx,
                expert: expert_idx,
            },
            new_weight,
        )
    }

    fn num_decoder_layers(&self) -> usize {
        self.layers.len()
    }

    fn get_weight(&self, key: &WeightKey) -> anyhow::Result<Tensor> {
        match key {
            WeightKey::AttnOProj { layer } => {
                let l = self
                    .layers
                    .get(*layer)
                    .ok_or_else(|| anyhow::anyhow!("Layer {layer} out of bounds"))?;
                Ok(l.o_proj_weight().clone())
            }
            WeightKey::MoeDownProj { layer, expert } => {
                let moe = self
                    .layers
                    .get(*layer)
                    .ok_or_else(|| anyhow::anyhow!("Layer {layer} out of bounds"))?
                    .moe_block()
                    .ok_or_else(|| anyhow::anyhow!("Layer {layer} is not MoE"))?;
                Ok(moe.expert(*expert).down_proj_weight().clone())
            }
            WeightKey::MoeGate { layer } => {
                let moe = self
                    .layers
                    .get(*layer)
                    .ok_or_else(|| anyhow::anyhow!("Layer {layer} out of bounds"))?
                    .moe_block()
                    .ok_or_else(|| anyhow::anyhow!("Layer {layer} is not MoE"))?;
                Ok(moe.gate_weight().clone())
            }
        }
    }

    fn set_weight(&mut self, key: &WeightKey, tensor: &Tensor) -> anyhow::Result<Tensor> {
        match key {
            WeightKey::AttnOProj { layer } => {
                let l = self
                    .layers
                    .get_mut(*layer)
                    .ok_or_else(|| anyhow::anyhow!("Layer {layer} out of bounds"))?;
                Ok(l.swap_o_proj(tensor.clone()))
            }
            WeightKey::MoeDownProj { layer, expert } => {
                self.swap_expert_weight_impl(*layer, *expert, tensor.clone())
            }
            WeightKey::MoeGate { layer } => {
                let moe = self
                    .layers
                    .get_mut(*layer)
                    .ok_or_else(|| anyhow::anyhow!("Layer {layer} out of bounds"))?
                    .moe_block_mut()
                    .ok_or_else(|| anyhow::anyhow!("Layer {layer} is not MoE"))?;
                Ok(moe.swap_gate(tensor.clone()))
            }
        }
    }

    fn format_chat_prompt(&self, prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn forward_cached(
        &self,
        input_ids: &Tensor,
        cache: &mut crate::model::arch::KVCache,
    ) -> anyhow::Result<Tensor> {
        let output = self.forward_cached_capture_impl(input_ids, cache, false)?;
        Ok(output.logits)
    }

    fn forward_cached_with_capture(
        &self,
        input_ids: &Tensor,
        cache: &mut crate::model::arch::KVCache,
        capture: bool,
    ) -> anyhow::Result<ModelOutput> {
        self.forward_cached_capture_impl(input_ids, cache, capture).map_err(Into::into)
    }

    fn set_expert_mask(&mut self, mask: ExpertMask) {
        self.expert_mask = mask;
    }

    fn clear_expert_mask(&mut self) {
        self.expert_mask = ExpertMask::new();
    }
}
