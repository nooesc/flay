use anyhow::Result;
use candle_core::Device;

use crate::eval::DomainRefusalRate;
use crate::eval::refusal::detect_refusal_generated;
use crate::model::arch::MoeModel;

/// Evaluate refusal rates on domain-specific eval prompts.
///
/// Generates tokens for each prompt and checks for refusal phrases in the
/// generated text. More accurate than the previous first-token heuristic.
pub fn evaluate_domain_refusals(
    model: &dyn MoeModel,
    domain_name: &str,
    eval_prompts: &[String],
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
) -> Result<DomainRefusalRate> {
    let mut refused = 0;

    for prompt in eval_prompts {
        let formatted = model.format_chat_prompt(prompt);
        let encoding = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {e}"))?;

        let result = detect_refusal_generated(model, tokenizer, encoding.get_ids(), device)?;
        if result.refused {
            refused += 1;
        }
    }

    let total = eval_prompts.len();
    let rate = if total > 0 {
        refused as f32 / total as f32
    } else {
        0.0
    };

    Ok(DomainRefusalRate {
        domain: domain_name.to_string(),
        refused,
        total,
        rate,
    })
}
