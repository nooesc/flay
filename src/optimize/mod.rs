// Optimization: grid search over threshold to minimize KL divergence
// Swaps expert weights in-place on the model to measure KL per trial

pub mod bayesian;

use anyhow::Result;

/// Optimization strategy for hyperparameter search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OptimizeMode {
    /// Uniform grid search over threshold values.
    Grid,
    /// Tree-structured Parzen Estimator (multi-objective Bayesian optimization).
    Bayesian,
}
use candle_core::Tensor;
use indicatif::{ProgressBar, ProgressStyle};

use crate::abliterate::direction_set::DirectionSet;
use crate::abliterate::orthogonalize::abliterate_experts;
use crate::abliterate::scoring::{select_guilty_experts, ExpertScore};
use crate::abliterate::weight_key::WeightKey;
use crate::analysis::kl_divergence::collect_logits;
use crate::pipeline::compute_mean_kl;
use crate::model::arch::MoeModel;

pub struct OptimizationResult {
    pub best_threshold: f32,
    pub best_kl: f32,
    pub num_abliterated: usize,
    pub trial_results: Vec<TrialResult>,
}

pub struct TrialResult {
    pub threshold: f32,
    pub kl_divergence: f32,
    pub num_abliterated: usize,
}

/// Grid search over threshold values with actual KL divergence measurement.
///
/// For each threshold candidate:
/// 1. Select guilty experts at that threshold
/// 2. Compute abliterated weights
/// 3. Swap them into the model
/// 4. Collect logits and measure KL divergence vs original
/// 5. Swap original weights back
///
/// Picks the threshold that abliterates the most experts while keeping
/// KL divergence below `max_kl` (default: 0.1). If all trials exceed
/// the KL budget, picks the one with the lowest KL.
pub fn optimize_threshold(
    model: &mut dyn MoeModel,
    scores: &[ExpertScore],
    directions: &DirectionSet,
    original_weights: &[(usize, usize, Tensor)],
    eval_tokens: &[Tensor],
    trials: usize,
    strength_min: f32,
) -> Result<OptimizationResult> {
    anyhow::ensure!(trials > 0, "optimization trials must be > 0");

    let max_score = scores
        .iter()
        .map(|s| s.combined_score)
        .fold(0.0f32, f32::max);
    if max_score <= 0.0 {
        return Ok(OptimizationResult {
            best_threshold: 0.0,
            best_kl: 0.0,
            num_abliterated: 0,
            trial_results: vec![],
        });
    }

    // Collect original model logits once
    println!("       Collecting baseline logits for KL measurement...");
    let original_logits = collect_logits(model, eval_tokens)?;

    let step = max_score / trials as f32;
    let mut trial_results = Vec::new();

    let pb = ProgressBar::new(trials as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("       Optimizing [{bar:30}] {pos}/{len} trials ({eta})")
            .unwrap()
            .progress_chars("=> "),
    );

    for trial in 0..trials {
        let threshold = step * (trial as f32 + 0.5);
        let guilty = select_guilty_experts(scores, Some(threshold));
        if guilty.is_empty() {
            pb.inc(1);
            continue;
        }

        // Abliterate at this threshold
        let abliterated = abliterate_experts(&guilty, directions, original_weights, strength_min, 1.0)?;

        // Swap abliterated weights into model
        let mut old_weights: Vec<(usize, usize, Tensor)> = Vec::new();
        for aw in &abliterated {
            let (layer_idx, expert_idx) = match &aw.key {
                WeightKey::MoeDownProj { layer, expert } => (*layer, *expert),
                other => anyhow::bail!("Unexpected weight key in optimize swap: {other}"),
            };
            let old = model.swap_expert_weight(layer_idx, expert_idx, &aw.new_weight)?;
            old_weights.push((layer_idx, expert_idx, old));
        }

        // Measure KL divergence
        let abliterated_logits = collect_logits(model, eval_tokens)?;
        let kl = compute_mean_kl(&original_logits, &abliterated_logits)?;

        // Swap original weights back
        for (layer_idx, expert_idx, old_weight) in old_weights {
            model.swap_expert_weight(layer_idx, expert_idx, &old_weight)?;
        }

        tracing::info!(
            "Trial {}/{}: threshold={:.4}, experts={}, KL={:.6}",
            trial + 1,
            trials,
            threshold,
            guilty.len(),
            kl,
        );

        trial_results.push(TrialResult {
            threshold,
            kl_divergence: kl,
            num_abliterated: guilty.len(),
        });

        pb.inc(1);
    }
    pb.finish_and_clear();

    // Pick best threshold: most experts abliterated with KL < 0.1,
    // or lowest KL if all exceed the budget
    let max_kl = 0.1;
    let best = trial_results
        .iter()
        .filter(|t| t.num_abliterated > 0)
        .filter(|t| t.kl_divergence < max_kl)
        .max_by_key(|t| t.num_abliterated)
        .or_else(|| {
            // All exceeded budget — pick lowest KL
            trial_results
                .iter()
                .filter(|t| t.num_abliterated > 0)
                .min_by(|a, b| a.kl_divergence.total_cmp(&b.kl_divergence))
        });

    let (best_threshold, best_kl, num_abliterated) = match best {
        Some(t) => (t.threshold, t.kl_divergence, t.num_abliterated),
        None => (max_score, 0.0, 0),
    };

    Ok(OptimizationResult {
        best_threshold,
        best_kl,
        num_abliterated,
        trial_results,
    })
}

