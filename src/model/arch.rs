use anyhow::Result;
use candle_core::Tensor;

use crate::abliterate::weight_key::WeightKey;

/// Per-layer KV cache for autoregressive generation.
///
/// Each layer's cache is lazily initialized on first use. The cache stores
/// K and V in KV-head space (`[B=1, num_kv_heads, seq_len, head_dim]`).
pub struct KVCache {
    /// Per-layer (K, V) pairs, lazily initialized on first use.
    pub entries: Vec<Option<(Tensor, Tensor)>>,
    /// Current cached sequence length (same for all layers).
    seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            entries: vec![None; num_layers],
            seq_len: 0,
        }
    }

    /// Current cached sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Advance the cached sequence length by `n` tokens.
    /// Called once per model forward after all layers complete.
    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }
}

/// Captured routing and activation data from a single MoE layer forward pass.
#[derive(Debug, Clone)]
pub struct MoeLayerCapture {
    /// Per-token top-k expert indices: outer len = n_tokens, inner len = top_k
    pub expert_indices: Vec<Vec<u32>>,
    /// Per-token top-k routing weights (post-softmax, optionally normalized)
    pub routing_weights: Vec<Vec<f32>>,
    /// Per-expert list of (token_index, expert_output_tensor) pairs
    pub expert_outputs: Vec<Vec<(usize, Tensor)>>,
}

/// Output from a model forward pass, with optional MoE capture data.
pub struct ModelOutput {
    pub logits: Tensor,
    pub residual_states: Vec<(usize, Tensor)>,
    pub moe_captures: Vec<(usize, MoeLayerCapture)>,
}

/// Tracks which experts should be masked (gate logit = -inf) during inference.
/// Used as a diagnostic tool to validate expert identification before weight modification.
#[derive(Debug, Clone, Default)]
pub struct ExpertMask {
    /// Set of (layer_index, expert_index) pairs to mask.
    masked: std::collections::HashSet<(usize, usize)>,
}

impl ExpertMask {
    pub fn new() -> Self {
        Self {
            masked: std::collections::HashSet::new(),
        }
    }

    pub fn add(&mut self, layer: usize, expert: usize) {
        self.masked.insert((layer, expert));
    }

    pub fn is_masked(&self, layer: usize, expert: usize) -> bool {
        self.masked.contains(&(layer, expert))
    }

    pub fn clear(&mut self) {
        self.masked.clear();
    }

    pub fn masked_in_layer(&self, layer: usize) -> usize {
        self.masked.iter().filter(|(l, _)| *l == layer).count()
    }

    pub fn is_empty(&self) -> bool {
        self.masked.is_empty()
    }

    pub fn len(&self) -> usize {
        self.masked.len()
    }

    /// Get the set of expert indices masked in a specific layer.
    pub fn experts_in_layer(&self, layer: usize) -> std::collections::HashSet<usize> {
        self.masked
            .iter()
            .filter(|(l, _)| *l == layer)
            .map(|(_, e)| *e)
            .collect()
    }
}

/// Trait abstracting over MoE model architectures (Qwen3, DeepSeek-V3, etc.).
pub trait MoeModel {
    fn forward(&self, input_ids: &Tensor, capture: bool) -> Result<ModelOutput>;
    fn num_moe_layers(&self) -> usize {
        self.moe_layer_indices().len()
    }
    fn num_experts(&self) -> usize;
    fn moe_layer_indices(&self) -> Vec<usize>;
    fn weight_key(&self, layer_idx: usize, expert_idx: usize) -> String;
    fn shared_expert_indices(&self) -> Vec<usize> {
        vec![]
    }
    fn swap_expert_weight(
        &mut self,
        layer_idx: usize,
        expert_idx: usize,
        new_weight: &Tensor,
    ) -> Result<Tensor>;

    /// Total number of decoder layers (MoE + dense).
    fn num_decoder_layers(&self) -> usize;

    /// Get a clone of the weight tensor identified by `key`.
    fn get_weight(&self, key: &WeightKey) -> Result<Tensor>;

    /// Set the weight tensor identified by `key`, returning the old value.
    fn set_weight(&mut self, key: &WeightKey, tensor: &Tensor) -> Result<Tensor>;

    /// Wrap a user prompt in the model's chat template so tokenization
    /// produces the same token sequence seen during real chat inference.
    /// Returns the formatted string ready for `tokenizer.encode(..., false)`.
    fn format_chat_prompt(&self, prompt: &str) -> String;

    /// Forward pass with KV-cache for autoregressive generation.
    ///
    /// On first call (empty cache), processes the full input and populates the cache.
    /// On subsequent calls, processes only new tokens using cached K/V.
    /// Returns logits only (no capture data needed during generation).
    ///
    /// The caller must NOT call `cache.advance()` — the model does it internally
    /// after all layers complete, ensuring consistent RoPE offset.
    fn forward_cached(&self, input_ids: &Tensor, cache: &mut KVCache) -> Result<Tensor>;

    /// Cached forward pass with optional routing capture for decode-time analysis.
    ///
    /// Default implementation bails if `capture` is true; models that support
    /// decode-time capture override this.
    fn forward_cached_with_capture(
        &self,
        input_ids: &Tensor,
        cache: &mut KVCache,
        capture: bool,
    ) -> Result<ModelOutput> {
        if capture {
            anyhow::bail!("forward_cached_with_capture not implemented for this model");
        }
        let logits = self.forward_cached(input_ids, cache)?;
        Ok(ModelOutput {
            logits,
            residual_states: vec![],
            moe_captures: vec![],
        })
    }

    /// Set experts to mask during inference (gate logits set to -inf).
    fn set_expert_mask(&mut self, mask: ExpertMask);

    /// Clear all expert masks.
    fn clear_expert_mask(&mut self);
}
