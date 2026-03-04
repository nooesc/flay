// KL divergence evaluation between original and abliterated model outputs
// Used to verify abliteration quality: low KL = minimal capability loss

use candle_core::{DType, Tensor, D};
use candle_nn::ops::log_softmax;

use crate::model::arch::MoeModel;

/// Compute the KL divergence D_KL(P || Q) between original and abliterated logit distributions.
///
/// Uses the numerically stable log-sum-exp formulation:
/// ```text
/// KL(P || Q) = sum_i P(i) * (log P(i) - log Q(i))
/// ```
///
/// Both inputs should be raw (unnormalized) logits of shape `[vocab_size]` or
/// `[batch, vocab_size]`. Softmax is applied internally in log-space.
///
/// Returns the mean KL divergence across all prompts as a scalar f32.
pub fn kl_divergence(original_logits: &Tensor, abliterated_logits: &Tensor) -> anyhow::Result<f32> {
    // Cast to F32 for numerical stability
    let log_p = log_softmax(&original_logits.to_dtype(DType::F32)?, D::Minus1)?;
    let log_q = log_softmax(&abliterated_logits.to_dtype(DType::F32)?, D::Minus1)?;

    // P = exp(log_p)
    let p = log_p.exp()?;

    // KL = sum(P * (log_p - log_q), dim=-1) per sample
    let kl_per_sample = (p * (log_p - log_q)?)?.sum(D::Minus1)?;

    // Return mean across all samples
    let mean_kl = kl_per_sample.mean_all()?.to_scalar::<f32>()?;

    Ok(mean_kl)
}

/// Collect the last-token logits from each prompt by running the model in non-capture mode.
///
/// For each prompt token tensor (1D, shape `[seq_len]`), runs a forward pass and
/// extracts the logits at the final sequence position. Returns a vec of 1D logit
/// tensors, each of shape `[vocab_size]`.
pub fn collect_logits(model: &dyn MoeModel, tokens: &[Tensor]) -> anyhow::Result<Vec<Tensor>> {
    let mut logits_vec = Vec::with_capacity(tokens.len());

    for (i, prompt_tokens) in tokens.iter().enumerate() {
        // [seq_len] -> [1, seq_len]
        let input = prompt_tokens.unsqueeze(0)?;
        let output = model.forward(&input, false)?;

        // output.logits: [1, seq_len, vocab_size]
        let seq_len = output.logits.dim(1)?;
        let last_logits = output
            .logits
            .narrow(1, seq_len - 1, 1)? // [1, 1, vocab_size]
            .squeeze(0)?                  // [1, vocab_size]
            .squeeze(0)?;                 // [vocab_size]

        logits_vec.push(last_logits);

        if (i + 1) % 10 == 0 {
            tracing::debug!("Collected logits for {}/{} prompts", i + 1, tokens.len());
        }
    }

    Ok(logits_vec)
}
