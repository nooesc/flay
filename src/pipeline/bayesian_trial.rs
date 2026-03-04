// Bayesian optimization inner loop for multi-objective abliteration tuning.
//
// Each trial: suggest HyperParams via TPE -> recompute SVD directions ->
// score experts -> select guilty -> abliterate -> measure KL + refusal rate ->
// report observation back to the optimizer.

use anyhow::Result;
use candle_core::Tensor;
use indicatif::{ProgressBar, ProgressStyle};

use crate::abliterate::orthogonalize::{abliterate_experts, AbliterationMode};
use crate::abliterate::scoring::{score_experts_dispatch, select_guilty_experts};
use crate::abliterate::weight_key::WeightKey;
use crate::analysis::activations::ExpertStats;
use crate::analysis::kl_divergence::collect_logits;
use crate::eval::refusal::detect_refusal_logits;
use crate::model::arch::MoeModel;
use crate::optimize::bayesian::{HyperParams, Observation, ParetoEntry, SearchSpace, TpeOptimizer};
use crate::pipeline::{build_directions, compute_mean_kl};

/// Result of Bayesian optimization.
pub struct BayesianResult {
    pub best_params: HyperParams,
    pub best_kl: f32,
    pub best_refusal_rate: f32,
    pub pareto_front: Vec<ParetoEntry>,
    pub all_trials: Vec<TrialSummary>,
}

/// Summary of a single Bayesian trial.
pub struct TrialSummary {
    pub params: HyperParams,
    pub kl: f32,
    pub refusal_rate: f32,
    pub num_abliterated: usize,
}

/// Run a single Bayesian trial.
///
/// 1. Build DirectionSet from params (recomputes SVD for this trial's num_directions/energy)
/// 2. Score experts
/// 3. Select guilty at params.threshold
/// 4. Abliterate with params.strength_min
/// 5. Swap in, measure KL + refusal rate, swap back
fn run_trial(
    model: &mut dyn MoeModel,
    stats: &ExpertStats,
    all_expert_weights: &[(usize, usize, Tensor)],
    eval_tokens: &[Tensor],
    harmful_sample: &[Tensor],
    tokenizer: &tokenizers::Tokenizer,
    original_logits: &[Tensor],
    params: &HyperParams,
    min_activations: usize,
) -> Result<(f32, f32, usize)> {
    let mode = AbliterationMode::from_index(params.abliteration_mode);

    // Recompute directions for this trial's hyperparams
    let directions = build_directions(
        stats,
        mode,
        min_activations,
        params.num_directions,
        params.direction_energy,
    )?;

    let scores = score_experts_dispatch(stats, &directions)?;
    let guilty = select_guilty_experts(&scores, Some(params.threshold));

    if guilty.is_empty() {
        return Ok((0.0, 1.0, 0));
    }

    // Filter weights to only the guilty set
    let guilty_weights: Vec<(usize, usize, Tensor)> = guilty
        .iter()
        .filter_map(|expert| {
            all_expert_weights
                .iter()
                .find(|(l, e, _)| *l == expert.layer_idx && *e == expert.expert_idx)
                .map(|(l, e, w)| (*l, *e, w.clone()))
        })
        .collect();

    let abliterated = abliterate_experts(
        &guilty,
        &directions,
        &guilty_weights,
        params.strength_min,
        1.0,
    )?;

    let num_abliterated = abliterated.len();
    if num_abliterated == 0 {
        return Ok((0.0, 1.0, 0));
    }

    // Swap abliterated weights in
    let mut old_weights: Vec<(usize, usize, Tensor)> = Vec::new();
    for aw in &abliterated {
        let (layer_idx, expert_idx) = match &aw.key {
            WeightKey::MoeDownProj { layer, expert } => (*layer, *expert),
            other => anyhow::bail!("Unexpected weight key in Bayesian trial swap: {other}"),
        };
        let old = model.swap_expert_weight(layer_idx, expert_idx, &aw.new_weight)?;
        old_weights.push((layer_idx, expert_idx, old));
    }

    // Measure KL divergence
    let trial_logits = collect_logits(model, eval_tokens)?;
    let kl = compute_mean_kl(original_logits, &trial_logits)?;

    // Measure refusal rate on harmful sample
    let mut refused = 0usize;
    for toks in harmful_sample {
        let input = toks.unsqueeze(0)?;
        let out = model.forward(&input, false)?;
        let seq_len = out.logits.dim(1)?;
        let last = out
            .logits
            .narrow(1, seq_len - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?;
        if detect_refusal_logits(&last, tokenizer)?.refused {
            refused += 1;
        }
    }
    let refusal_rate = if harmful_sample.is_empty() {
        0.0
    } else {
        refused as f32 / harmful_sample.len() as f32
    };

    // Swap back
    for (l, e, w) in old_weights {
        model.swap_expert_weight(l, e, &w)?;
    }

    Ok((kl, refusal_rate, num_abliterated))
}

/// Drive the full Bayesian optimization loop.
///
/// Uses Tree-structured Parzen Estimation (TPE) to search over:
/// - threshold, num_directions, direction_energy, strength_min, abliteration_mode
///
/// Each trial recomputes SVD directions with the trial's parameters,
/// allowing the optimizer to explore the full hyperparameter space.
pub fn optimize_bayesian(
    model: &mut dyn MoeModel,
    stats: &ExpertStats,
    all_expert_weights: &[(usize, usize, Tensor)],
    eval_tokens: &[Tensor],
    harmful_sample: &[Tensor],
    tokenizer: &tokenizers::Tokenizer,
    trials: usize,
    min_activations: usize,
) -> Result<BayesianResult> {
    anyhow::ensure!(trials > 0, "Bayesian optimization requires > 0 trials");

    // Collect baseline logits once
    println!("       Collecting baseline logits for Bayesian optimization...");
    let original_logits = collect_logits(model, eval_tokens)?;

    let mut optimizer = TpeOptimizer::new(SearchSpace::default());
    let mut all_trials = Vec::new();

    let pb = ProgressBar::new(trials as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("       Bayesian  [{bar:30}] {pos}/{len} trials ({eta})")
            .unwrap()
            .progress_chars("=> "),
    );

    for trial_idx in 0..trials {
        let params = optimizer.suggest();

        let (kl, refusal_rate, num_abliterated) = run_trial(
            model,
            stats,
            all_expert_weights,
            eval_tokens,
            harmful_sample,
            tokenizer,
            &original_logits,
            &params,
            min_activations,
        )?;

        tracing::info!(
            "Bayesian trial {}/{}: mode={}, dirs={}, energy={:.2}, threshold={:.4}, \
             strength_min={:.2} -> KL={:.6}, refusal={:.1}%, experts={}",
            trial_idx + 1,
            trials,
            AbliterationMode::from_index(params.abliteration_mode),
            params.num_directions,
            params.direction_energy,
            params.threshold,
            params.strength_min,
            kl,
            refusal_rate * 100.0,
            num_abliterated,
        );

        all_trials.push(TrialSummary {
            params: params.clone(),
            kl,
            refusal_rate,
            num_abliterated,
        });

        optimizer.tell(Observation {
            params,
            kl_divergence: kl,
            refusal_rate,
        });

        pb.inc(1);
    }
    pb.finish_and_clear();

    // Select best trial
    let best_obs = optimizer
        .best_trial(0.2)
        .or_else(|| optimizer.best_unconstrained())
        .ok_or_else(|| anyhow::anyhow!("No valid Bayesian trials completed"))?;

    let pareto_front = optimizer.pareto_front();

    Ok(BayesianResult {
        best_params: best_obs.params.clone(),
        best_kl: best_obs.kl_divergence,
        best_refusal_rate: best_obs.refusal_rate,
        pareto_front,
        all_trials,
    })
}
