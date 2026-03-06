// Autoregressive generation for Qwen3.5 hybrid GDN/attention model.
//
// Supports greedy and temperature-based sampling with chat template formatting.
// Cache-aware: prefill processes full prompt, decode generates one token at a time.
// Optional capture/steering via RuntimeCtx.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use serde::Serialize;

use super::qwen35::Qwen35Model;
use super::qwen35_steering::{CaptureTrace, RuntimeCtx, SteeringPlan};

/// Rich generation result with metadata for eval.
#[derive(Debug, Clone, Serialize)]
pub struct GenerationResult {
    /// Full generated text (including think tags).
    pub full_text: String,
    /// Text with <think>...</think> content removed.
    pub answer_text: String,
    /// Content inside <think> tags (empty if none).
    pub think_text: String,
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Whether generation hit the max token limit (vs EOS).
    pub hit_limit: bool,
}

/// Strip `<think>...</think>` blocks from text.
/// Returns (answer_text, think_text).
pub fn strip_think_tags(text: &str) -> (String, String) {
    let mut answer = String::new();
    let mut think = String::new();
    let mut remaining = text;

    while let Some(start) = remaining.find("<think>") {
        answer.push_str(&remaining[..start]);
        let after_tag = &remaining[start + 7..]; // skip "<think>"
        if let Some(end) = after_tag.find("</think>") {
            think.push_str(after_tag[..end].trim());
            think.push('\n');
            remaining = &after_tag[end + 8..]; // skip "</think>"
        } else {
            // Unclosed think tag — rest is think content
            think.push_str(after_tag.trim());
            remaining = "";
            break;
        }
    }
    answer.push_str(remaining);
    (answer.trim().to_string(), think.trim().to_string())
}

/// Stop tokens for Qwen3.5 models.
const QWEN35_EOS_IDS: &[u32] = &[
    248044, // <|endoftext|>
    248046, // <|im_end|>
];

/// Sampling strategy for token generation.
pub enum Sampling {
    Greedy,
    Temperature(f64),
}

/// Format a user prompt with the Qwen3.5 chat template.
/// Seeds assistant turn with closed think tags to skip the think phase.
pub fn format_chat_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n")
}

/// Sample a token ID from logits.
fn sample_token(logits: &Tensor, sampling: &Sampling) -> Result<u32> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    match sampling {
        Sampling::Greedy => {
            let id = logits_f32.argmax(D::Minus1)?.to_scalar::<u32>()?;
            Ok(id)
        }
        Sampling::Temperature(temp) => {
            let scaled = (logits_f32 / *temp)?;
            let probs = candle_nn::ops::softmax_last_dim(&scaled)?;
            let probs_vec: Vec<f32> = probs.to_vec1()?;
            let r: f32 = rand::random();
            let mut cumsum = 0.0;
            for (i, &p) in probs_vec.iter().enumerate() {
                cumsum += p;
                if cumsum >= r {
                    return Ok(i as u32);
                }
            }
            Ok((probs_vec.len() - 1) as u32)
        }
    }
}

/// Generate tokens autoregressively from a Qwen3.5 model.
///
/// Returns generated token IDs (excluding the input prompt).
pub fn generate(
    model: &Qwen35Model,
    input_ids: &[u32],
    max_new_tokens: usize,
    sampling: &Sampling,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut ctx = RuntimeCtx::noop();
    generate_with_ctx(model, input_ids, max_new_tokens, sampling, device, &mut ctx)
}

/// Generate tokens with capture/steering context.
pub fn generate_with_ctx(
    model: &Qwen35Model,
    input_ids: &[u32],
    max_new_tokens: usize,
    sampling: &Sampling,
    device: &Device,
    ctx: &mut RuntimeCtx,
) -> Result<Vec<u32>> {
    if max_new_tokens == 0 {
        return Ok(Vec::new());
    }

    let mut cache = model.make_cache();
    let mut generated = Vec::with_capacity(max_new_tokens);
    let prompt_len = input_ids.len();

    // Prefill: process full prompt
    ctx.is_prefill = true;
    ctx.seq_offset = 0;
    ctx.capture_pos = Some(prompt_len - 1); // capture last token of prompt
    let input = Tensor::new(input_ids, device)?.unsqueeze(0)?;
    let logits = model.forward_with_ctx(&input, &mut cache, ctx)?;
    let seq_len = logits.dim(1)?;
    let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(0)?.squeeze(0)?;
    let mut next_id = sample_token(&last_logits, sampling)?;

    if QWEN35_EOS_IDS.contains(&next_id) {
        return Ok(generated);
    }
    generated.push(next_id);

    // Decode: one token at a time with cache
    ctx.is_prefill = false;
    for step in 1..max_new_tokens {
        ctx.decode_step = step - 1;
        ctx.seq_offset = prompt_len + step - 1;
        ctx.capture_pos = Some(0); // decode input is always 1 token
        let input = Tensor::new(&[next_id], device)?.unsqueeze(0)?;
        let logits = model.forward_with_ctx(&input, &mut cache, ctx)?;
        let last_logits = logits.squeeze(0)?.squeeze(0)?;
        next_id = sample_token(&last_logits, sampling)?;

        if QWEN35_EOS_IDS.contains(&next_id) {
            break;
        }
        generated.push(next_id);
    }

    Ok(generated)
}

/// Generate text from a prompt using the chat template.
pub fn generate_text(
    model: &Qwen35Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &Sampling,
    device: &Device,
) -> Result<String> {
    let formatted = format_chat_prompt(prompt);
    let encoding = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let token_ids = generate(model, encoding.get_ids(), max_new_tokens, sampling, device)?;
    let text = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
    Ok(text)
}

/// Generate text with capture enabled. Returns (text, trace).
pub fn generate_text_with_capture(
    model: &Qwen35Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &Sampling,
    device: &Device,
) -> Result<(String, CaptureTrace)> {
    let formatted = format_chat_prompt(prompt);
    let encoding = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

    let mut trace = CaptureTrace::default();
    let mut ctx = RuntimeCtx {
        capture: Some(&mut trace),
        steering: None,
        is_prefill: true,
        decode_step: 0,
        seq_offset: 0,
        capture_pos: None,
    };

    let token_ids = generate_with_ctx(
        model,
        encoding.get_ids(),
        max_new_tokens,
        sampling,
        device,
        &mut ctx,
    )?;
    let text = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
    Ok((text, trace))
}

/// Generate text with steering applied. Returns text.
pub fn generate_text_with_steering(
    model: &Qwen35Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &Sampling,
    device: &Device,
    plan: &SteeringPlan,
) -> Result<String> {
    let result = generate_result_with_steering(model, tokenizer, prompt, max_new_tokens, sampling, device, plan)?;
    Ok(result.full_text)
}

/// Generate with steering, returning rich result with think stripping and metadata.
pub fn generate_result_with_steering(
    model: &Qwen35Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &Sampling,
    device: &Device,
    plan: &SteeringPlan,
) -> Result<GenerationResult> {
    let formatted = format_chat_prompt(prompt);
    let encoding = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

    let mut ctx = RuntimeCtx {
        capture: None,
        steering: Some(plan),
        is_prefill: true,
        decode_step: 0,
        seq_offset: 0,
        capture_pos: None,
    };

    let token_ids = generate_with_ctx(
        model,
        encoding.get_ids(),
        max_new_tokens,
        sampling,
        device,
        &mut ctx,
    )?;
    let tokens_generated = token_ids.len();
    let hit_limit = tokens_generated >= max_new_tokens;
    let full_text = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
    let (answer_text, think_text) = strip_think_tags(&full_text);

    Ok(GenerationResult {
        full_text,
        answer_text,
        think_text,
        tokens_generated,
        hit_limit,
    })
}

/// Generate without steering, returning rich result.
pub fn generate_result(
    model: &Qwen35Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &Sampling,
    device: &Device,
) -> Result<GenerationResult> {
    let formatted = format_chat_prompt(prompt);
    let encoding = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let token_ids = generate(model, encoding.get_ids(), max_new_tokens, sampling, device)?;
    let tokens_generated = token_ids.len();
    let hit_limit = tokens_generated >= max_new_tokens;
    let full_text = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
    let (answer_text, think_text) = strip_think_tags(&full_text);

    Ok(GenerationResult {
        full_text,
        answer_text,
        think_text,
        tokens_generated,
        hit_limit,
    })
}
