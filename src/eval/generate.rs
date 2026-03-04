// Greedy token generation with KV-cache for eval purposes.
// Uses forward_cached for O(1) per-token inference after prefill.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::model::arch::{KVCache, MoeLayerCapture, MoeModel};

/// Stop tokens for Qwen3 models.
const QWEN3_EOS_IDS: &[u32] = &[
    151643, // <|endoftext|>
    151645, // <|im_end|>
];

/// Generate tokens greedily from the model using KV-cache.
///
/// Prefills the cache with the full prompt, then decodes one token at a time.
/// Returns the generated token IDs (excluding the original input).
pub fn generate_greedy(
    model: &dyn MoeModel,
    input_ids: &[u32],
    max_new_tokens: usize,
    device: &Device,
) -> Result<Vec<u32>> {
    if max_new_tokens == 0 {
        return Ok(Vec::new());
    }

    let num_layers = model.num_decoder_layers();
    let mut cache = KVCache::new(num_layers);
    let mut generated = Vec::with_capacity(max_new_tokens);

    // Prefill: process full prompt, populate cache
    let input = Tensor::new(input_ids, device)?.unsqueeze(0)?;
    let logits = model.forward_cached(&input, &mut cache)?;
    let seq_len = logits.dim(1)?;
    let last_logits = logits
        .narrow(1, seq_len - 1, 1)?
        .squeeze(0)?
        .squeeze(0)?
        .to_dtype(DType::F32)?;
    let mut next_id: u32 = last_logits.argmax(0)?.to_scalar()?;

    if QWEN3_EOS_IDS.contains(&next_id) {
        return Ok(generated);
    }
    generated.push(next_id);

    // Decode: single token at a time using cached K/V
    for _ in 1..max_new_tokens {
        let input = Tensor::new(&[next_id], device)?.unsqueeze(0)?;
        let logits = model.forward_cached(&input, &mut cache)?;
        let last_logits = logits
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        next_id = last_logits.argmax(0)?.to_scalar()?;

        if QWEN3_EOS_IDS.contains(&next_id) {
            break;
        }
        generated.push(next_id);
    }

    Ok(generated)
}

/// Generate tokens and decode to text.
pub fn generate_text(
    model: &dyn MoeModel,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_new_tokens: usize,
    device: &Device,
) -> Result<String> {
    let token_ids = generate_greedy(model, input_ids, max_new_tokens, device)?;
    let text = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
    Ok(text)
}

/// Routing-only captures from decode steps.
pub struct DecodeCaptures {
    /// Per-step, per-layer routing captures. steps[0] = prefill (all prompt tokens).
    pub steps: Vec<Vec<(usize, MoeLayerCapture)>>,
}

/// Generate tokens with routing capture on prefill + first N decode steps.
pub fn generate_greedy_with_capture(
    model: &dyn MoeModel,
    input_ids: &[u32],
    max_new_tokens: usize,
    capture_steps: usize,
    device: &Device,
) -> Result<(Vec<u32>, DecodeCaptures)> {
    if max_new_tokens == 0 {
        return Ok((Vec::new(), DecodeCaptures { steps: vec![] }));
    }

    let num_layers = model.num_decoder_layers();
    let mut cache = KVCache::new(num_layers);
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut decode_captures = DecodeCaptures { steps: Vec::new() };

    // Prefill WITH capture (covers first-token decision)
    let input = Tensor::new(input_ids, device)?.unsqueeze(0)?;
    let output = model.forward_cached_with_capture(&input, &mut cache, true)?;
    decode_captures.steps.push(output.moe_captures);

    let seq_len = output.logits.dim(1)?;
    let last_logits = output.logits
        .narrow(1, seq_len - 1, 1)?
        .squeeze(0)?.squeeze(0)?
        .to_dtype(DType::F32)?;
    let mut next_id: u32 = last_logits.argmax(0)?.to_scalar()?;

    if QWEN3_EOS_IDS.contains(&next_id) {
        return Ok((generated, decode_captures));
    }
    generated.push(next_id);

    // Decode loop: capture first N steps, then normal
    for step in 1..max_new_tokens {
        let input = Tensor::new(&[next_id], device)?.unsqueeze(0)?;
        let should_capture = step <= capture_steps;

        let logits = if should_capture {
            let output = model.forward_cached_with_capture(&input, &mut cache, true)?;
            decode_captures.steps.push(output.moe_captures);
            output.logits
        } else {
            model.forward_cached(&input, &mut cache)?
        };

        let last_logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        next_id = last_logits.argmax(0)?.to_scalar()?;

        if QWEN3_EOS_IDS.contains(&next_id) {
            break;
        }
        generated.push(next_id);
    }

    Ok((generated, decode_captures))
}
