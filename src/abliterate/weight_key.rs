// Typed key for identifying model weights (attention o_proj or MoE expert down_proj).

use std::fmt;

/// Typed key identifying a specific weight matrix in the model.
///
/// Used by `AbliteratedWeight` and the `get_weight`/`set_weight` API on `MoeModel`
/// to refer to weights without string-based key construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WeightKey {
    /// Attention output projection: `model.layers.{layer}.self_attn.o_proj.weight`
    AttnOProj { layer: usize },
    /// MoE expert down projection: `model.layers.{layer}.mlp.experts.{expert}.down_proj.weight`
    MoeDownProj { layer: usize, expert: usize },
    /// MoE router gate: `model.layers.{layer}.mlp.gate.weight`
    MoeGate { layer: usize },
}

impl WeightKey {
    /// Generate the safetensors tensor name for this weight.
    pub fn safetensors_key(&self) -> String {
        match self {
            Self::AttnOProj { layer } => {
                format!("model.layers.{layer}.self_attn.o_proj.weight")
            }
            Self::MoeDownProj { layer, expert } => {
                format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight")
            }
            Self::MoeGate { layer } => {
                format!("model.layers.{layer}.mlp.gate.weight")
            }
        }
    }

    /// The decoder layer index this weight belongs to.
    pub fn layer(&self) -> usize {
        match self {
            Self::AttnOProj { layer }
            | Self::MoeDownProj { layer, .. }
            | Self::MoeGate { layer } => *layer,
        }
    }
}

impl fmt::Display for WeightKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AttnOProj { layer } => write!(f, "layer {layer} o_proj"),
            Self::MoeDownProj { layer, expert } => {
                write!(f, "layer {layer} expert {expert} down_proj")
            }
            Self::MoeGate { layer } => write!(f, "layer {layer} gate"),
        }
    }
}
