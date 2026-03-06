use anyhow::{bail, Result};
use serde::Deserialize;

/// Layer type in the hybrid GDN/attention architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    LinearAttention, // GDN (Gated DeltaNet)
    FullAttention,   // Standard multi-head attention
}

/// RoPE configuration nested inside text_config.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

/// Inner text_config from the Qwen3.5 config.json.
#[derive(Debug, Clone, Deserialize)]
struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub attn_output_gate: Option<bool>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub attention_bias: Option<bool>,
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
    #[serde(default)]
    pub mamba_ssm_dtype: Option<String>,
}

/// Top-level config.json wrapper.
#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    text_config: TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
}

/// Parsed and validated Qwen3.5 configuration.
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    // Full attention params
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    // GDN (linear attention) params
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    // Norms
    pub rms_norm_eps: f64,
    // RoPE
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    // Flags
    pub attn_output_gate: bool,
    pub tie_word_embeddings: bool,
    // Per-layer type assignment
    pub layer_types: Vec<LayerType>,
}

impl Qwen35Config {
    /// Parse from config.json string, unwrapping text_config.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(json)?;
        let tc = raw.text_config;

        let layer_types: Vec<LayerType> = tc
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "linear_attention" => Ok(LayerType::LinearAttention),
                "full_attention" => Ok(LayerType::FullAttention),
                other => bail!("Unknown layer_type: {other}"),
            })
            .collect::<Result<Vec<_>>>()?;

        let config = Self {
            vocab_size: tc.vocab_size,
            hidden_size: tc.hidden_size,
            intermediate_size: tc.intermediate_size,
            num_hidden_layers: tc.num_hidden_layers,
            num_attention_heads: tc.num_attention_heads,
            num_key_value_heads: tc.num_key_value_heads,
            head_dim: tc.head_dim,
            linear_conv_kernel_dim: tc.linear_conv_kernel_dim,
            linear_key_head_dim: tc.linear_key_head_dim,
            linear_num_key_heads: tc.linear_num_key_heads,
            linear_value_head_dim: tc.linear_value_head_dim,
            linear_num_value_heads: tc.linear_num_value_heads,
            rms_norm_eps: tc.rms_norm_eps,
            rope_theta: tc.rope_parameters.rope_theta,
            partial_rotary_factor: tc.rope_parameters.partial_rotary_factor,
            attn_output_gate: tc.attn_output_gate.unwrap_or(true),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
            layer_types,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            bail!(
                "layer_types length ({}) != num_hidden_layers ({})",
                self.layer_types.len(),
                self.num_hidden_layers,
            );
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            bail!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads,
                self.num_key_value_heads,
            );
        }
        if self.head_dim == 0 || self.linear_key_head_dim == 0 || self.linear_value_head_dim == 0 {
            bail!("Head dimensions must be > 0");
        }
        if !(0.0..=1.0).contains(&self.partial_rotary_factor) {
            bail!(
                "partial_rotary_factor ({}) must be in [0.0, 1.0]",
                self.partial_rotary_factor,
            );
        }
        Ok(())
    }

    /// Number of head dims that receive rotary embeddings.
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    /// Count of GDN (linear attention) layers.
    pub fn num_gdn_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count()
    }

    /// Count of full attention layers.
    pub fn num_full_attn_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count()
    }

    /// Whether layer `idx` is a GDN layer.
    pub fn is_gdn_layer(&self, idx: usize) -> bool {
        self.layer_types.get(idx) == Some(&LayerType::LinearAttention)
    }
}
