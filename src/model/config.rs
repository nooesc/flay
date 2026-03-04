use anyhow::{bail, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub attention_bias: Option<bool>,
}

impl Qwen3MoeConfig {
    /// Validate MoE geometry and attention parameters. Call after deserialization.
    pub fn validate(&self) -> Result<()> {
        if self.num_experts > 0 && self.decoder_sparse_step == 0 {
            bail!(
                "decoder_sparse_step must be > 0 when num_experts > 0 (got {})",
                self.decoder_sparse_step,
            );
        }
        if self.num_key_value_heads == 0 {
            bail!("num_key_value_heads must be > 0");
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            bail!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads,
                self.num_key_value_heads,
            );
        }
        if self.num_experts > 0 {
            if self.num_experts_per_tok == 0 {
                bail!("num_experts_per_tok must be > 0 when num_experts > 0");
            }
            if self.num_experts_per_tok > self.num_experts {
                bail!(
                    "num_experts_per_tok ({}) must be <= num_experts ({})",
                    self.num_experts_per_tok,
                    self.num_experts,
                );
            }
        }
        Ok(())
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.num_experts > 0 && (layer_idx + 1) % self.decoder_sparse_step == 0
    }

    pub fn attention_bias(&self) -> bool {
        self.attention_bias.unwrap_or(false)
    }
}
