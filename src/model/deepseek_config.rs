use anyhow::{bail, Result};
use serde::Deserialize;

/// Configuration for DeepSeek-V3 / V3.1 / R1 models.
///
/// Field names match the actual HuggingFace config.json for `model_type: "deepseek_v3"`.
/// Notable differences from Qwen3-MoE:
///   - MLA (Multi-head Latent Attention) instead of standard GQA
///   - 256 routed experts + 1 shared expert, top-8 routing
///   - First `first_k_dense_replace` layers are dense (no MoE)
///   - `moe_layer_freq` controls which subsequent layers have MoE
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,

    // -- MoE geometry (field names match HF config.json) --
    #[serde(default = "default_n_routed_experts")]
    pub n_routed_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_n_shared_experts")]
    pub n_shared_experts: usize,
    #[serde(default = "default_first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub n_group: usize,
    #[serde(default)]
    pub topk_group: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub routed_scaling_factor: f64,

    // -- MLA (Multi-head Latent Attention) --
    #[serde(default)]
    pub q_lora_rank: usize,
    #[serde(default)]
    pub kv_lora_rank: usize,
    #[serde(default)]
    pub qk_nope_head_dim: usize,
    #[serde(default)]
    pub qk_rope_head_dim: usize,
    #[serde(default)]
    pub v_head_dim: usize,

    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_n_routed_experts() -> usize {
    256
}
fn default_num_experts_per_tok() -> usize {
    8
}
fn default_n_shared_experts() -> usize {
    1
}
fn default_first_k_dense_replace() -> usize {
    3
}
fn default_moe_layer_freq() -> usize {
    1
}

impl DeepSeekV3Config {
    /// Validate MoE geometry and MLA parameters. Call after deserialization.
    pub fn validate(&self) -> Result<()> {
        if self.n_routed_experts == 0 {
            bail!("n_routed_experts must be > 0 for DeepSeek-V3");
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.n_routed_experts {
            bail!(
                "num_experts_per_tok ({}) must be in [1, {}]",
                self.num_experts_per_tok,
                self.n_routed_experts
            );
        }
        if self.first_k_dense_replace > self.num_hidden_layers {
            bail!(
                "first_k_dense_replace ({}) exceeds num_hidden_layers ({})",
                self.first_k_dense_replace,
                self.num_hidden_layers
            );
        }
        if self.moe_layer_freq == 0 {
            bail!("moe_layer_freq must be > 0");
        }
        Ok(())
    }

    /// Returns true if the given layer index hosts an MoE block.
    ///
    /// DeepSeek-V3 uses dense layers for the first `first_k_dense_replace` layers,
    /// then MoE layers at every `moe_layer_freq` interval thereafter.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if layer_idx < self.first_k_dense_replace {
            return false;
        }
        (layer_idx - self.first_k_dense_replace).is_multiple_of(self.moe_layer_freq)
    }

    /// List all MoE layer indices.
    pub fn moe_layer_indices(&self) -> Vec<usize> {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_moe_layer(i))
            .collect()
    }
}
