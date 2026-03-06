// Qwen3.5 hybrid GDN/attention model for candle.
//
// Architecture: 3:1 ratio of GDN (linear attention) to standard attention layers.
// Standard attention uses output gating: o_proj(attn * sigmoid(gate)).
// All norms use shifted weight: (1 + weight) * rms_norm(x).
//
// Weight prefix: model.language_model.layers.{i}.{linear_attn|self_attn|mlp}.*

use anyhow::Result;
use candle_core::{Device, Module, Tensor, D};
use candle_nn::{
    embedding, linear_no_bias, ops::sigmoid, ops::softmax_last_dim, Embedding, Linear, RmsNorm,
    VarBuilder,
};

use super::gated_delta::{GatedDeltaNet, GdnState};
use super::ops::PartialRotaryEmbedding;
use super::qwen35_config::{LayerType, Qwen35Config};
use super::qwen35_steering::{HookPoint, RuntimeCtx};

// ---------------------------------------------------------------------------
// RmsNorm with shifted weight loading
// ---------------------------------------------------------------------------

/// Load an RmsNorm whose stored weights need +1 shift.
/// HF Qwen3.5 stores raw weight; the model computes (1 + w) * rms_norm(x).
fn load_shifted_rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let raw_weight = vb.get(size, "weight")?;
    let shifted = (raw_weight.ones_like()? + &raw_weight)?;
    Ok(RmsNorm::new(shifted, eps))
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

// ---------------------------------------------------------------------------
// Gated Attention (standard attention with output gating)
// ---------------------------------------------------------------------------

struct GatedAttention {
    q_proj: Linear, // outputs 2x head_dim (Q + gate)
    k_proj: Linear,
    v_proj: Linear,
    pub o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: PartialRotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_groups: usize,
}

impl GatedAttention {
    fn new(config: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        let h = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        // q_proj outputs Q AND gate: 2 * num_heads * head_dim
        let q_proj = linear_no_bias(h, num_heads * head_dim * 2, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(h, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(h, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, h, vb.pp("o_proj"))?;

        let q_norm = load_shifted_rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = load_shifted_rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope: PartialRotaryEmbedding::new(
                config.rotary_dim(),
                8192, // max_seq_len — will recreate if needed
                config.rope_theta,
                vb.device(),
            )?,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_groups: num_heads / num_kv_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let h = self.num_heads;
        let kv_h = self.num_kv_heads;
        let d = self.head_dim;

        // Q projection: [B, S, 2*H*D] -> split into queries [B, S, H, D] and gate [B, S, H*D]
        let q_proj_out = self.q_proj.forward(xs)?;
        let q_proj_reshaped = q_proj_out.reshape((b, s, h, 2 * d))?;
        let queries = q_proj_reshaped.narrow(D::Minus1, 0, d)?;
        let gate = q_proj_reshaped.narrow(D::Minus1, d, d)?.reshape((b, s, h * d))?;

        let keys = self.k_proj.forward(xs)?.reshape((b, s, kv_h, d))?;
        let values = self.v_proj.forward(xs)?.reshape((b, s, kv_h, d))?;

        // Apply per-head norms
        let queries = self.q_norm.forward(&queries)?; // [B, S, H, D]
        let keys = self.k_norm.forward(&keys)?;

        // Transpose to [B, H, S, D] for attention
        let queries = queries.transpose(1, 2)?;
        let keys = keys.transpose(1, 2)?;
        let values = values.transpose(1, 2)?;

        // RoPE
        let cache_offset = match kv_cache {
            Some((k, _)) => k.dim(2)?,
            None => 0,
        };
        let queries = self.rope.apply(&queries, cache_offset)?;
        let keys = self.rope.apply(&keys, cache_offset)?;

        // KV cache update
        let (keys, values) = match kv_cache.take() {
            Some((prev_k, prev_v)) => {
                let keys = Tensor::cat(&[prev_k, keys], 2)?;
                let values = Tensor::cat(&[prev_v, values], 2)?;
                (keys, values)
            }
            None => (keys, values),
        };
        *kv_cache = Some((keys.clone(), values.clone()));

        // GQA: expand KV heads (contiguous for Metal matmul compatibility)
        let keys = if self.kv_groups > 1 {
            let (b, kv_h, seq, d) = keys.dims4()?;
            keys.unsqueeze(2)?
                .expand((b, kv_h, self.kv_groups, seq, d))?
                .reshape((b, h, seq, d))?
                .contiguous()?
        } else {
            keys
        };
        let values = if self.kv_groups > 1 {
            let (b, kv_h, seq, d) = values.dims4()?;
            values.unsqueeze(2)?
                .expand((b, kv_h, self.kv_groups, seq, d))?
                .reshape((b, h, seq, d))?
                .contiguous()?
        } else {
            values
        };

        // Scaled dot-product attention
        // Ensure contiguous layout for Metal matmul (RoPE cat produces non-contiguous tensors)
        let queries = queries.contiguous()?;
        let keys = keys.contiguous()?;
        let scale = (d as f64).powf(-0.5);
        let attn_weights = (queries.matmul(&keys.transpose(2, 3)?)? * scale)?;

        // Causal mask
        let kv_len = attn_weights.dim(D::Minus1)?;
        if s > 1 {
            let mask = create_causal_mask(s, kv_len, xs.dtype(), xs.device())?;
            let attn_weights = attn_weights.broadcast_add(&mask)?;
            let attn_weights = softmax_last_dim(&attn_weights)?;
            let attn_out = attn_weights.matmul(&values)?; // [B, H, S, D]
            let attn_out = attn_out.transpose(1, 2)?.reshape((b, s, h * d))?;

            // Output gating: o_proj(attn * sigmoid(gate))
            let gated = (attn_out * sigmoid(&gate)?)?;
            Ok(self.o_proj.forward(&gated)?)
        } else {
            let attn_weights = softmax_last_dim(&attn_weights)?;
            let attn_out = attn_weights.matmul(&values)?;
            let attn_out = attn_out.transpose(1, 2)?.reshape((b, s, h * d))?;
            let gated = (attn_out * sigmoid(&gate)?)?;
            Ok(self.o_proj.forward(&gated)?)
        }
    }
}

/// Create a causal attention mask: positions where q can attend to k.
/// Returns [1, 1, q_len, kv_len] with 0 for valid and -inf for masked.
fn create_causal_mask(q_len: usize, kv_len: usize, dtype: candle_core::DType, device: &Device) -> candle_core::Result<Tensor> {
    let offset = kv_len - q_len;
    // Build mask values on CPU then transfer
    let mut mask_data = vec![0f32; q_len * kv_len];
    for q in 0..q_len {
        for k in 0..kv_len {
            if k > q + offset {
                mask_data[q * kv_len + k] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, (q_len, kv_len), device)?;
    mask.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

enum AttentionLayer {
    Gdn(GatedDeltaNet),
    FullAttn(GatedAttention),
}

/// Cache for a single decoder layer.
pub enum LayerCache {
    /// GDN layer: (conv_state, recurrent_state)
    Gdn(Option<Tensor>, GdnState),
    /// Full attention: KV cache (K, V)
    FullAttn(Option<(Tensor, Tensor)>),
}

impl LayerCache {
    pub fn new_gdn() -> Self {
        LayerCache::Gdn(None, None)
    }
    pub fn new_full_attn() -> Self {
        LayerCache::FullAttn(None)
    }
}

struct DecoderLayer {
    attn: AttentionLayer,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn new(
        config: &Qwen35Config,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_gdn = config.is_gdn_layer(layer_idx);

        let attn = if is_gdn {
            AttentionLayer::Gdn(GatedDeltaNet::new(config, vb.pp("linear_attn"))?)
        } else {
            AttentionLayer::FullAttn(GatedAttention::new(config, vb.pp("self_attn"))?)
        };

        let input_layernorm = load_shifted_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = load_shifted_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::new(config.hidden_size, config.intermediate_size, vb.pp("mlp"))?;

        Ok(Self {
            attn,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(&self, xs: &Tensor, cache: &mut LayerCache) -> Result<Tensor> {
        let mut ctx = RuntimeCtx::noop();
        self.forward_with_ctx(xs, cache, 0, 0, &mut ctx)
    }

    fn forward_with_ctx(
        &self,
        xs: &Tensor,
        cache: &mut LayerCache,
        layer_idx: usize,
        num_layers: usize,
        ctx: &mut RuntimeCtx,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(xs)?;
        let residual = xs;

        let attn_out = match (&self.attn, cache) {
            (AttentionLayer::Gdn(gdn), LayerCache::Gdn(conv_state, rec_state)) => {
                let (out, new_conv, new_rec) =
                    gdn.forward(&normed, conv_state.as_ref(), rec_state.take(), None)?;
                *conv_state = Some(new_conv);
                *rec_state = Some(new_rec);
                out
            }
            (AttentionLayer::FullAttn(attn), LayerCache::FullAttn(kv_cache)) => {
                attn.forward(&normed, kv_cache)?
            }
            _ => anyhow::bail!("Cache type mismatch for layer"),
        };

        // Hook: TokenMixerOut (capture then steer before residual add)
        ctx.maybe_capture(layer_idx, HookPoint::TokenMixerOut, &attn_out, num_layers)?;
        let attn_out = ctx.maybe_steer(layer_idx, HookPoint::TokenMixerOut, attn_out)?;

        let h = (residual + attn_out)?;
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let out = (h + mlp_out)?;

        // Hook: ResidualPostMlp (capture then steer)
        ctx.maybe_capture(layer_idx, HookPoint::ResidualPostMlp, &out, num_layers)?;
        let out = ctx.maybe_steer(layer_idx, HookPoint::ResidualPostMlp, out)?;

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

pub struct Qwen35Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    config: Qwen35Config,
}

impl Qwen35Model {
    pub fn new(config: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        // Weight prefix: model.language_model.*
        let lm_vb = vb.pp("model.language_model");

        let embed_tokens = embedding(config.vocab_size, config.hidden_size, lm_vb.pp("embed_tokens"))?;
        let norm = load_shifted_rms_norm(config.hidden_size, config.rms_norm_eps, lm_vb.pp("norm"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = DecoderLayer::new(
                config,
                i,
                lm_vb.pp(format!("layers.{i}")),
            )?;
            layers.push(layer);
        }

        // lm_head is at the top level (not under model.language_model)
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: config.clone(),
        })
    }

    /// Create a fresh cache for all layers.
    pub fn make_cache(&self) -> Vec<LayerCache> {
        self.config
            .layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::LinearAttention => LayerCache::new_gdn(),
                LayerType::FullAttention => LayerCache::new_full_attn(),
            })
            .collect()
    }

    /// Forward pass. Returns logits [B, S, vocab_size].
    pub fn forward(&self, input_ids: &Tensor, cache: &mut [LayerCache]) -> Result<Tensor> {
        let mut ctx = RuntimeCtx::noop();
        self.forward_with_ctx(input_ids, cache, &mut ctx)
    }

    /// Forward pass with capture/steering context.
    pub fn forward_with_ctx(
        &self,
        input_ids: &Tensor,
        cache: &mut [LayerCache],
        ctx: &mut RuntimeCtx,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            cache.len() == self.layers.len(),
            "Cache length ({}) != layer count ({})",
            cache.len(),
            self.layers.len()
        );
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let num_layers = self.layers.len();

        for (i, (layer, layer_cache)) in
            self.layers.iter().zip(cache.iter_mut()).enumerate()
        {
            hidden = layer.forward_with_ctx(&hidden, layer_cache, i, num_layers, ctx)?;
        }

        hidden = self.norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;
        Ok(logits)
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}
