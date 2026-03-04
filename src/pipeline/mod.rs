// Pipeline orchestration: direction building, threshold selection, abliteration,
// KL measurement, and evaluation — extracted from main.rs for reuse by both
// the main CLI and the Bayesian optimization inner loop.

pub mod bayesian_trial;

use anyhow::Result;
use candle_core::{Device, Tensor};
use indicatif::{ProgressBar, ProgressStyle};

use crate::abliterate::direction_set::DirectionSet;
use crate::abliterate::directions::{compute_multi_refusal_directions, compute_refusal_directions};
use crate::abliterate::orthogonalize::{abliterate_experts, AbliteratedWeight, AbliterationMode};
use crate::abliterate::routing;
use crate::abliterate::scoring::{score_experts_dispatch, select_guilty_experts, select_experts_stable, decompose_hcdg_hrcg, ExpertScore};
use crate::abliterate::weight_key::WeightKey;
use crate::analysis::activations::{collect_expert_stats, ExpertStats};
use crate::analysis::decode_analysis::DecodeRoutingStats;
use crate::analysis::kl_divergence::kl_divergence;
use crate::analysis::prompts::NamedDomain;
use crate::eval::domain::evaluate_domain_refusals;
use crate::eval::generate::generate_greedy_with_capture;
use crate::eval::reasoning::run_reasoning_canary;
use crate::eval::refusal::detect_refusal_generated;
use crate::eval::utility::{self, UtilityQuestion};
use crate::eval::{DomainRefusalRate, EvalResults, OverRefusalRate, ReasoningResult, RefusalRate};
use crate::model::arch::{MoeModel, ExpertMask};
use crate::optimize::{optimize_threshold, OptimizeMode};

/// Configuration for the abliteration pipeline.
pub struct PipelineConfig {
    pub abliteration_mode: AbliterationMode,
    pub min_activations: usize,
    pub num_directions: usize,
    pub direction_energy: f32,
    pub threshold: Option<f32>,
    pub strength_min: f32,
    pub optimize: bool,
    pub optimize_mode: OptimizeMode,
    pub trials: usize,
    pub kl_eval_size: usize,
    pub run_eval: bool,
    /// Enable residual-stream (o_proj) abliteration across all decoder layers.
    pub residual: bool,
    /// Global strength multiplier for o_proj abliteration (default 1.0).
    pub residual_strength: f32,
    /// MoE expert strength cap when --residual is active (default 0.3).
    pub moe_strength: f32,
    /// Number of bootstrap samples for SES (1 = standard selection).
    pub stable_k: usize,
    /// Top-N experts per bootstrap sample.
    pub stable_top_n: usize,
    /// Mask-only diagnostic mode (no weight modification).
    pub mask_only: bool,
    /// Path to jailbreak prompts for HCDG/HRCG decomposition.
    pub jailbreak_data: Option<String>,
    /// Maximum number of abliteration passes (1 = single pass, default).
    pub passes: usize,
    /// Target refusal rate — stop iterating when refusal drops below this.
    pub target_refusal: f32,
    /// Gate weight attenuation strength for safety-critical experts (0.0 = disabled).
    pub route_strength: f32,
    /// Capture expert routing during prefill + first N decode steps (diagnostic).
    pub capture_decode: Option<usize>,
}

/// Inputs to the pipeline (model, data, and tokens).
pub struct PipelineInputs {
    pub model: Box<dyn MoeModel>,
    pub stats: ExpertStats,
    pub all_expert_weights: Vec<(usize, usize, Tensor)>,
    pub harmful_tokens: Vec<Tensor>,
    pub harmless_tokens: Vec<Tensor>,
    pub domain_datasets: Vec<NamedDomain>,
    pub eval_prompts: Vec<String>,
    pub utility_questions: Vec<UtilityQuestion>,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    /// Tokenized jailbreak harmful prompts for HCDG/HRCG decomposition.
    pub jailbreak_tokens: Option<Vec<Tensor>>,
    /// Experts to exclude from MoE abliteration (already processed in prior passes).
    /// These experts will be removed from the guilty list before abliteration, KL, and
    /// eval so that guardrails accurately reflect only the current pass's changes.
    pub exclude_experts: std::collections::HashSet<(usize, usize)>,
}

/// Results from the pipeline.
pub struct PipelineResult {
    pub abliterated: Vec<AbliteratedWeight>,
    pub scores: Vec<ExpertScore>,
    pub threshold_used: f32,
    pub threshold_method: String,
    pub kl_divergence: Option<f32>,
    pub eval_results: Option<EvalResults>,
    pub num_abliterated: usize,
}

/// Build a DirectionSet for the given abliteration mode.
///
/// Dispatches between single-direction (mean-difference) and multi-directional
/// (SVD) extraction, and applies projected decomposition when the mode requires it.
pub fn build_directions(
    stats: &ExpertStats,
    mode: AbliterationMode,
    min_activations: usize,
    num_directions: usize,
    direction_energy: f32,
) -> Result<DirectionSet> {
    let base = if mode.is_multi() {
        DirectionSet::Multi(compute_multi_refusal_directions(
            stats,
            min_activations,
            num_directions,
            direction_energy,
        )?)
    } else {
        DirectionSet::Single(compute_refusal_directions(stats, min_activations)?)
    };

    if mode.is_projected() {
        base.project(stats)
    } else {
        Ok(base)
    }
}

/// Swap abliterated weights into the model, run a closure, then swap back.
pub fn swap_measure_swap<F, T>(
    model: &mut dyn MoeModel,
    abliterated: &[AbliteratedWeight],
    f: F,
) -> Result<T>
where
    F: FnOnce(&dyn MoeModel) -> Result<T>,
{
    // Guard: bail on duplicate keys to prevent restore corruption
    let mut seen_keys = std::collections::HashSet::new();
    for aw in abliterated {
        if !seen_keys.insert(aw.key.clone()) {
            anyhow::bail!("Duplicate weight key in abliterated set: {}", aw.key);
        }
    }

    let mut old_weights: Vec<(WeightKey, Tensor)> = Vec::new();
    for aw in abliterated {
        let old = model.set_weight(&aw.key, &aw.new_weight)?;
        old_weights.push((aw.key.clone(), old));
    }

    let result = f(model);

    // Restore in reverse order for safety
    for (key, old_weight) in old_weights.into_iter().rev() {
        model.set_weight(&key, &old_weight)?;
    }

    result
}

/// Compute mean KL divergence between two sets of per-prompt logits.
pub fn compute_mean_kl(original: &[Tensor], abliterated: &[Tensor]) -> Result<f32> {
    let n = original.len().min(abliterated.len());
    if n == 0 {
        return Ok(0.0);
    }
    let mut total_kl = 0.0f32;
    for i in 0..n {
        let orig = original[i].unsqueeze(0)?;
        let abli = abliterated[i].unsqueeze(0)?;
        total_kl += kl_divergence(&orig, &abli)?;
    }
    Ok(total_kl / n as f32)
}

/// Collect logits with a progress bar.
pub fn collect_logits_with_progress(
    model: &dyn MoeModel,
    tokens: &[Tensor],
    pb: &ProgressBar,
) -> Result<Vec<Tensor>> {
    let mut logits_vec = Vec::with_capacity(tokens.len());
    for prompt_tokens in tokens {
        let input = prompt_tokens.unsqueeze(0)?;
        let output = model.forward(&input, false)?;
        let seq_len = output.logits.dim(1)?;
        let last_logits = output
            .logits
            .narrow(1, seq_len - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?;
        logits_vec.push(last_logits);
        pb.inc(1);
    }
    Ok(logits_vec)
}

/// Run the full abliteration pipeline.
pub fn run_pipeline(
    config: &PipelineConfig,
    inputs: &mut PipelineInputs,
) -> Result<PipelineResult> {
    if config.residual && config.optimize {
        anyhow::bail!(
            "Residual-stream mode is incompatible with optimization. \
             Disable --optimize when using --residual."
        );
    }

    // -----------------------------------------------------------------------
    // Phase 1: Build directions
    // -----------------------------------------------------------------------
    println!(
        "[6/9] Phase 2: Computing {} refusal directions and scoring experts...",
        config.abliteration_mode,
    );
    let mut directions = build_directions(
        &inputs.stats,
        config.abliteration_mode,
        config.min_activations,
        config.num_directions,
        config.direction_energy,
    )?;

    // -----------------------------------------------------------------------
    // Phase 1.5: Compute residual-stream directions (if --residual)
    // -----------------------------------------------------------------------
    let residual_abliterated: Vec<AbliteratedWeight> = if config.residual {
        use crate::abliterate::residual_directions::compute_residual_directions;
        use crate::abliterate::residual_stream::abliterate_residual_stream;

        println!("       Computing per-layer residual directions...");
        let residual_dirs = compute_residual_directions(&inputs.stats)?;
        println!(
            "       Reference layer: {} (lambda range: {:.2}..{:.2})",
            residual_dirs.reference_layer,
            residual_dirs.lambda.iter().cloned().fold(f32::INFINITY, f32::min),
            residual_dirs.lambda.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );

        println!("       Abliterating o_proj across all decoder layers...");
        let o_proj_weights = abliterate_residual_stream(
            &*inputs.model,
            &residual_dirs,
            config.residual_strength,
        )?;
        println!("       {} o_proj weights abliterated", o_proj_weights.len());
        o_proj_weights
    } else {
        Vec::new()
    };

    // -----------------------------------------------------------------------
    // Phase 2: Score experts
    // -----------------------------------------------------------------------
    let mut scores = score_experts_dispatch(&inputs.stats, &directions)?;

    // -----------------------------------------------------------------------
    // Phase 3: Threshold selection
    // -----------------------------------------------------------------------
    let eval_size = config.kl_eval_size.min(inputs.harmless_tokens.len());
    let eval_tokens = &inputs.harmless_tokens[..eval_size];

    let (threshold_used, threshold_method, strength_min_override) = if config.optimize {
        match config.optimize_mode {
            OptimizeMode::Grid => {
                println!(
                    "[7/9] Optimizing threshold (grid, {} trials)...",
                    config.trials
                );
                let opt_result = optimize_threshold(
                    &mut *inputs.model,
                    &scores,
                    &directions,
                    &inputs.all_expert_weights,
                    eval_tokens,
                    config.trials,
                    config.strength_min,
                )?;

                println!(
                    "       Best threshold: {:.4} ({} experts, KL={:.6})",
                    opt_result.best_threshold, opt_result.num_abliterated, opt_result.best_kl,
                );
                for trial in &opt_result.trial_results {
                    println!(
                        "         threshold={:.4}  experts={}  KL={:.6}",
                        trial.threshold, trial.num_abliterated, trial.kl_divergence,
                    );
                }

                (
                    opt_result.best_threshold,
                    "optimized (grid search)".to_string(),
                    None,
                )
            }
            OptimizeMode::Bayesian => {
                println!(
                    "[7/9] Optimizing hyperparameters (Bayesian TPE, {} trials)...",
                    config.trials
                );
                let harmful_sample_size = 10.min(inputs.harmful_tokens.len());
                let harmful_sample = &inputs.harmful_tokens[..harmful_sample_size];

                let bayes_result = bayesian_trial::optimize_bayesian(
                    &mut *inputs.model,
                    &inputs.stats,
                    &inputs.all_expert_weights,
                    eval_tokens,
                    harmful_sample,
                    &inputs.tokenizer,
                    config.trials,
                    config.min_activations,
                )?;

                println!(
                    "       Best: threshold={:.4}, mode={}, dirs={}, energy={:.2}, KL={:.6}, refusal={:.1}%",
                    bayes_result.best_params.threshold,
                    AbliterationMode::from_index(bayes_result.best_params.abliteration_mode),
                    bayes_result.best_params.num_directions,
                    bayes_result.best_params.direction_energy,
                    bayes_result.best_kl,
                    bayes_result.best_refusal_rate * 100.0,
                );

                if !bayes_result.pareto_front.is_empty() {
                    println!("       Pareto front ({} points):", bayes_result.pareto_front.len());
                    for p in &bayes_result.pareto_front {
                        println!(
                            "         KL={:.6}  refusal={:.1}%",
                            p.kl_divergence,
                            p.refusal_rate * 100.0,
                        );
                    }
                }

                // Rebuild directions with the best params for Phase 4+.
                // The Bayesian loop built them internally per trial, but we need
                // the optimal ones for the final abliteration pass through Phase 4-7.
                let best_mode =
                    AbliterationMode::from_index(bayes_result.best_params.abliteration_mode);
                directions = build_directions(
                    &inputs.stats,
                    best_mode,
                    config.min_activations,
                    bayes_result.best_params.num_directions,
                    bayes_result.best_params.direction_energy,
                )?;
                scores = score_experts_dispatch(&inputs.stats, &directions)?;

                // Fall through to Phase 4+ so SES, HCDG/HRCG decomposition,
                // router attenuation, and mask-only diagnostics all apply.
                // KL is always recomputed in Phase 6 with the actual expert set
                // (which may differ from the Bayesian trial due to SES/HCDG filtering).
                (
                    bayes_result.best_params.threshold,
                    format!(
                        "optimized (Bayesian TPE, {} trials)",
                        config.trials
                    ),
                    Some(bayes_result.best_params.strength_min),
                )
            }
        }
    } else if let Some(t) = config.threshold {
        (t, "manual".to_string(), None)
    } else {
        let guilty = select_guilty_experts(&scores, None);
        let t = guilty.last().map(|s| s.combined_score).unwrap_or(0.0);
        (t, "auto (elbow method)".to_string(), None)
    };

    // -----------------------------------------------------------------------
    // Phase 4: Select guilty experts
    // -----------------------------------------------------------------------

    // Warn if jailbreak data is provided but SES is not enabled
    if config.jailbreak_data.is_some() && config.stable_k <= 1 {
        println!(
            "  WARNING: --jailbreak-data requires --stable-k > 1 for HCDG/HRCG decomposition. \
             Ignoring jailbreak data."
        );
    }

    let guilty = if config.stable_k > 1 {
        tracing::info!(
            "Using stability-based selection (K={}, top_n={})",
            config.stable_k, config.stable_top_n,
        );
        let regular_stable = select_experts_stable(&scores, config.stable_k, config.stable_top_n, 42);

        // HCDG/HRCG decomposition when jailbreak data is available
        if let (Some(_), Some(jailbreak_tokens)) = (&config.jailbreak_data, &inputs.jailbreak_tokens) {
            println!("       Running HCDG/HRCG decomposition...");
            println!(
                "       Capturing jailbreak activations ({} prompts)...",
                jailbreak_tokens.len(),
            );

            // 1. Capture jailbreak expert stats (jailbreak as harmful, same harmless set)
            let jailbreak_stats = collect_expert_stats(
                &*inputs.model,
                jailbreak_tokens,
                &inputs.harmless_tokens,
                &inputs.device,
            )?;

            // 2. Build directions for jailbreak stats
            let jailbreak_directions = build_directions(
                &jailbreak_stats,
                config.abliteration_mode,
                config.min_activations,
                config.num_directions,
                config.direction_energy,
            )?;

            // 3. Score jailbreak experts
            let jailbreak_scores = score_experts_dispatch(&jailbreak_stats, &jailbreak_directions)?;

            // 4. Run SES on jailbreak scores (different seed to avoid correlation
            //    with regular SES — same seed would produce identical permutations)
            let jailbreak_stable = select_experts_stable(
                &jailbreak_scores,
                config.stable_k,
                config.stable_top_n,
                43,
            );

            println!(
                "       Regular SES: {} experts, Jailbreak SES: {} experts",
                regular_stable.len(),
                jailbreak_stable.len(),
            );

            // 5. Decompose into HCDG (detection) and HRCG (control)
            let (hcdg, hrcg) = decompose_hcdg_hrcg(&regular_stable, &jailbreak_stable);

            println!(
                "       HCDG (detection, preserved): {} experts",
                hcdg.len(),
            );
            println!(
                "       HRCG (control, abliteration targets): {} experts",
                hrcg.len(),
            );

            if hrcg.is_empty() {
                println!(
                    "  WARNING: HRCG is empty — all regular stable experts also appear in \
                     jailbreak stable set. Falling back to regular SES selection."
                );
                regular_stable
            } else {
                // 6. Use HRCG as the guilty set — map back to references into `scores`
                // (hrcg borrows from jailbreak_scores which will be dropped, so we need
                // to find the matching experts in the original scores vec)
                let hrcg_keys: std::collections::HashSet<(usize, usize)> = hrcg
                    .iter()
                    .map(|s| (s.layer_idx, s.expert_idx))
                    .collect();
                let mut guilty_from_hrcg: Vec<&ExpertScore> = scores
                    .iter()
                    .filter(|s| hrcg_keys.contains(&(s.layer_idx, s.expert_idx)))
                    .collect();
                guilty_from_hrcg.sort_by(|a, b| b.combined_score.total_cmp(&a.combined_score));
                guilty_from_hrcg
            }
        } else {
            regular_stable
        }
    } else {
        select_guilty_experts(&scores, Some(threshold_used))
    };

    // Filter out experts already processed in prior multi-pass iterations.
    // These had MoeDownProj abliterated previously; re-abliterating would
    // over-subtract the refusal direction. Filtering here (before abliteration,
    // KL, and eval) ensures guardrails accurately reflect this pass's changes.
    let guilty: Vec<&ExpertScore> = if inputs.exclude_experts.is_empty() {
        guilty
    } else {
        let pre_exclude = guilty.len();
        let filtered: Vec<&ExpertScore> = guilty
            .into_iter()
            .filter(|s| !inputs.exclude_experts.contains(&(s.layer_idx, s.expert_idx)))
            .collect();
        if filtered.len() < pre_exclude {
            tracing::info!(
                "Excluded {} already-abliterated experts from guilty list",
                pre_exclude - filtered.len(),
            );
        }
        filtered
    };

    println!(
        "       {} / {} experts selected for abliteration (threshold: {:.4}, {})",
        guilty.len(),
        scores.len(),
        threshold_used,
        threshold_method,
    );

    if guilty.is_empty() {
        println!("\nNo experts exceeded the refusal threshold. Model appears clean.");
        return Ok(PipelineResult {
            abliterated: vec![],
            scores,
            threshold_used,
            threshold_method,
            kl_divergence: None,
            eval_results: None,
            num_abliterated: 0,
        });
    }

    // Display selected experts
    let display_limit = 20;
    for (i, expert) in guilty.iter().enumerate().take(display_limit) {
        println!(
            "       #{}: layer {} expert {} (score={:.4}, projection={:.4}, bias={:.2})",
            i + 1,
            expert.layer_idx,
            expert.expert_idx,
            expert.combined_score,
            expert.refusal_projection,
            expert.routing_bias,
        );
    }
    if guilty.len() > display_limit {
        println!(
            "       ... and {} more (see --report for full list)",
            guilty.len() - display_limit
        );
    }

    // -----------------------------------------------------------------------
    // Decode capture diagnostic (must come before mask-only early return)
    // -----------------------------------------------------------------------
    if let Some(capture_steps) = config.capture_decode {
        let num_experts = inputs.model.num_experts();
        let moe_indices = inputs.model.moe_layer_indices();
        let mut decode_stats = DecodeRoutingStats::new(&moe_indices, num_experts);

        println!(
            "\n[Decode Capture] Capturing routing: prefill + {} decode steps...",
            capture_steps
        );

        // Convert Tensor tokens to u32 slices and run capture
        println!("  Harmful prompts ({})...", inputs.harmful_tokens.len());
        for tok_tensor in &inputs.harmful_tokens {
            let ids: Vec<u32> = tok_tensor.squeeze(0)?.to_vec1::<u32>()?;
            let (_, captures) = generate_greedy_with_capture(
                &*inputs.model,
                &ids,
                capture_steps + 1,
                capture_steps,
                &inputs.device,
            )?;
            decode_stats.accumulate(&captures, true);
        }

        println!("  Harmless prompts ({})...", inputs.harmless_tokens.len());
        for tok_tensor in &inputs.harmless_tokens {
            let ids: Vec<u32> = tok_tensor.squeeze(0)?.to_vec1::<u32>()?;
            let (_, captures) = generate_greedy_with_capture(
                &*inputs.model,
                &ids,
                capture_steps + 1,
                capture_steps,
                &inputs.device,
            )?;
            decode_stats.accumulate(&captures, false);
        }

        let decode_scores = decode_stats.score_experts(3);
        let prefill_experts: Vec<(usize, usize)> = guilty
            .iter()
            .map(|g| (g.layer_idx, g.expert_idx))
            .collect();
        let jaccard = decode_stats.jaccard_vs_prefill(&prefill_experts, 10, 3);

        println!("\n[Decode vs Prefill] Jaccard similarity (top-10 per layer):");
        let mut total_j = 0.0;
        let mut n_layers = 0;
        for (layer, j) in &jaccard {
            if *j > 0.0 || prefill_experts.iter().any(|(l, _)| l == layer) {
                println!("    Layer {:2}: {:.3}", layer, j);
                total_j += j;
                n_layers += 1;
            }
        }
        if n_layers > 0 {
            println!("    Mean: {:.3}", total_j / n_layers as f32);
        }

        println!("\n[Decode Capture] Top 15 decode-scored experts:");
        for (i, (layer, expert, score)) in decode_scores.iter().take(15).enumerate() {
            let pfx = if prefill_experts.contains(&(*layer, *expert)) {
                " [P]"
            } else {
                ""
            };
            println!(
                "    #{:2}: L{} E{} (score={:.4}){}",
                i + 1,
                layer,
                expert,
                score,
                pfx
            );
        }

        // Mask decode-identified experts and run eval
        let top_n = guilty.len().max(10);
        let mut mask = ExpertMask::new();
        for &(layer, expert, _) in decode_scores.iter().take(top_n) {
            mask.add(layer, expert);
        }
        println!(
            "\n[Decode Mask] Masking top {} decode-scored experts",
            mask.len()
        );
        inputs.model.set_expert_mask(mask);

        let eval_results = if config.run_eval {
            println!("\nRunning evaluation suite (decode-targeted mask-only)...");
            Some(run_eval_core(
                &*inputs.model,
                &inputs.harmful_tokens,
                &inputs.harmless_tokens,
                &inputs.domain_datasets,
                &inputs.eval_prompts,
                &inputs.utility_questions,
                &inputs.tokenizer,
                &inputs.device,
                0.0,
            )?)
        } else {
            None
        };

        inputs.model.clear_expert_mask();
        let num_guilty = guilty.len();
        drop(guilty);

        return Ok(PipelineResult {
            abliterated: vec![],
            scores,
            threshold_used,
            threshold_method,
            kl_divergence: None,
            eval_results,
            num_abliterated: num_guilty,
        });
    }

    // -----------------------------------------------------------------------
    // Mask-only diagnostic path
    // -----------------------------------------------------------------------
    if config.mask_only {
        let mut mask = ExpertMask::new();
        for g in &guilty {
            mask.add(g.layer_idx, g.expert_idx);
        }
        tracing::info!("Masking {} experts for diagnostic eval", mask.len());
        println!("       Masking {} experts (diagnostic mode, no weight changes)", mask.len());
        inputs.model.set_expert_mask(mask);

        let eval_results = if config.run_eval {
            println!("\nRunning evaluation suite (mask-only)...");
            Some(run_eval_core(
                &*inputs.model,
                &inputs.harmful_tokens,
                &inputs.harmless_tokens,
                &inputs.domain_datasets,
                &inputs.eval_prompts,
                &inputs.utility_questions,
                &inputs.tokenizer,
                &inputs.device,
                0.0, // no KL divergence in mask-only mode
            )?)
        } else {
            None
        };

        inputs.model.clear_expert_mask();

        // Drop guilty before moving scores
        let num_guilty = guilty.len();
        drop(guilty);

        return Ok(PipelineResult {
            abliterated: vec![],
            scores,
            threshold_used,
            threshold_method,
            kl_divergence: None,
            eval_results,
            num_abliterated: num_guilty,
        });
    }

    // -----------------------------------------------------------------------
    // Phase 5: Abliterate
    // -----------------------------------------------------------------------
    println!("[8/9] Phase 3: Abliterating guilty experts...");

    let skip_moe = config.residual && config.moe_strength == 0.0;
    let abliterated_experts = if skip_moe {
        println!("       Skipping MoE abliteration (--moe-strength 0.0)");
        Vec::new()
    } else {
        let guilty_weights: Vec<(usize, usize, Tensor)> = guilty
            .iter()
            .filter_map(|expert| {
                inputs
                    .all_expert_weights
                    .iter()
                    .find(|(l, e, _)| *l == expert.layer_idx && *e == expert.expert_idx)
                    .map(|(l, e, w)| (*l, *e, w.clone()))
            })
            .collect();

        let (expert_strength_min, expert_strength_max) = if config.residual {
            (config.moe_strength * 0.5, config.moe_strength)
        } else {
            (strength_min_override.unwrap_or(config.strength_min), 1.0)
        };

        let results = abliterate_experts(
            &guilty,
            &directions,
            &guilty_weights,
            expert_strength_min,
            expert_strength_max,
        )?;
        println!(
            "       Abliterated {} expert weight matrices",
            results.len(),
        );
        results
    };

    // Router weight attenuation (optional, controlled by --route-strength)
    //
    // attenuate_router_weights modifies the model in-place and returns (key, old_weight).
    // We capture the new weight, then restore the original so that swap_measure_swap
    // can manage the weight lifecycle consistently (swap in for eval, then restore).
    let routing_abliterated: Vec<AbliteratedWeight> = if config.route_strength > 0.0 {
        let expert_targets: Vec<(usize, usize)> = guilty
            .iter()
            .map(|g| (g.layer_idx, g.expert_idx))
            .collect();
        let routing_modified = routing::attenuate_router_weights(
            &mut *inputs.model,
            &expert_targets,
            config.route_strength,
        )?;
        println!(
            "       Attenuated router weights for {} gate matrices (strength={:.2})",
            routing_modified.len(),
            config.route_strength,
        );
        // Capture new weights, then restore originals
        let mut result = Vec::with_capacity(routing_modified.len());
        for (key, old_weight) in routing_modified {
            let new_weight = inputs.model.get_weight(&key)?;
            // Restore original weight so swap_measure_swap handles the lifecycle
            inputs.model.set_weight(&key, &old_weight)?;
            result.push(AbliteratedWeight {
                key,
                new_weight,
                strength: config.route_strength,
            });
        }
        result
    } else {
        Vec::new()
    };

    // Free the pre-abliteration expert weights — no longer needed after abliteration.
    // This reclaims significant GPU/system memory before KL/eval forward passes.
    inputs.all_expert_weights.clear();
    inputs.all_expert_weights.shrink_to_fit();

    // Combine o_proj, MoE, and routing abliterated weights into one vector
    let mut abliterated = residual_abliterated;
    abliterated.extend(abliterated_experts);
    abliterated.extend(routing_abliterated);

    // -----------------------------------------------------------------------
    // Phase 6: KL divergence measurement
    // -----------------------------------------------------------------------
    // KL is always freshly computed here (never reused from optimization trials)
    // because SES/HCDG/router filtering may change the expert set post-optimization.
    let kl_divergence_value = if !abliterated.is_empty() {
        println!(
            "       Measuring KL divergence ({} eval prompts)...",
            eval_size
        );

        let pb = ProgressBar::new(eval_size as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("       Baseline  [{bar:30}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );
        let original_logits =
            collect_logits_with_progress(&*inputs.model, eval_tokens, &pb)?;
        pb.finish_and_clear();

        let pb = ProgressBar::new(eval_size as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("       Abliterated [{bar:30}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );
        let kl = swap_measure_swap(&mut *inputs.model, &abliterated, |model| {
            let abliterated_logits = collect_logits_with_progress(model, eval_tokens, &pb)?;
            pb.finish_and_clear();
            compute_mean_kl(&original_logits, &abliterated_logits)
        })?;

        println!("       KL divergence: {:.6}", kl);
        Some(kl)
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Phase 7: Evaluation suite
    // -----------------------------------------------------------------------
    let eval_results = if config.run_eval {
        Some(run_eval_suite(
            &mut *inputs.model,
            &abliterated,
            &inputs.harmful_tokens,
            &inputs.harmless_tokens,
            &inputs.domain_datasets,
            &inputs.eval_prompts,
            &inputs.utility_questions,
            &inputs.tokenizer,
            &inputs.device,
            kl_divergence_value.unwrap_or(0.0),
        )?)
    } else {
        None
    };

    Ok(PipelineResult {
        num_abliterated: abliterated.len(),
        abliterated,
        scores,
        threshold_used,
        threshold_method,
        kl_divergence: kl_divergence_value,
        eval_results,
    })
}

/// Run the abliteration pipeline with optional multi-pass iteration.
///
/// When `config.passes` is 1 (default), this delegates directly to `run_pipeline()`.
/// When `config.passes` > 1, it runs multiple abliteration passes, committing weights
/// permanently after each pass and re-extracting refusal directions from the modified
/// model. Iteration stops early when:
/// - Target refusal rate is reached
/// - Reasoning capability drops below guardrail threshold
/// - KL divergence exceeds guardrail threshold
/// - Diminishing returns (refusal delta < 2% between passes)
///
/// Note: multi-pass requires `config.run_eval = true` for stopping conditions to work
/// (except the pass count limit).
pub fn run_pipeline_iterative(
    config: &PipelineConfig,
    inputs: &mut PipelineInputs,
) -> Result<PipelineResult> {
    const REASONING_MIN_PASS_RATE: f32 = 0.80;
    const KL_DIVERGENCE_CEILING: f32 = 0.15;
    const MIN_REFUSAL_DELTA: f32 = 0.02;

    if config.passes == 0 {
        anyhow::bail!("--passes must be at least 1");
    }

    if config.passes <= 1 {
        return run_pipeline(config, inputs);
    }

    if !config.run_eval {
        println!(
            "WARNING: Multi-pass abliteration (--passes {}) without --eval. \
             Only the pass count limit will be used as a stopping condition.",
            config.passes,
        );
    }

    // Measure baseline reasoning canary before any abliteration so multi-pass
    // guardrails use a relative threshold instead of a hardcoded 80%.
    let baseline_reasoning_rate: Option<f32> = if config.run_eval {
        tracing::info!("Measuring baseline reasoning canary (pre-abliteration)...");
        println!("Measuring baseline reasoning canary...");
        let (passed, total) = crate::eval::reasoning::run_reasoning_canary(
            inputs.model.as_ref(),
            &inputs.tokenizer,
            &inputs.device,
        )?;
        let rate = passed as f32 / total.max(1) as f32;
        println!(
            "  Baseline reasoning: {} / {} ({:.1}%)",
            passed, total, rate * 100.0,
        );
        Some(rate)
    } else {
        None
    };

    let mut all_abliterated: Vec<AbliteratedWeight> = Vec::new();
    let mut already_abliterated: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    let mut last_refusal_rate: Option<f32> = None;
    let mut cumulative_kl: f32 = 0.0;
    let mut final_scores = Vec::new();
    let mut final_threshold = 0.0f32;
    let mut final_threshold_method = String::new();
    let mut final_kl: Option<f32> = None;
    let mut final_eval: Option<EvalResults> = None;

    for pass in 0..config.passes {
        println!("\n{}", "=".repeat(60));
        println!("  Pass {}/{}", pass + 1, config.passes);
        println!("{}\n", "=".repeat(60));

        // Tell the pipeline which experts to exclude from MoE abliteration
        // so that guardrails (KL, reasoning, refusal) accurately reflect
        // only this pass's actual changes.
        inputs.exclude_experts = already_abliterated.clone();

        let result = run_pipeline(config, inputs)?;

        // Safety net: filter any already-abliterated experts that slipped through.
        // With exclude_experts set above, run_pipeline should already exclude them,
        // so this should be a no-op in practice.
        let pre_filter_count = result.abliterated.len();
        let new_abliterated: Vec<AbliteratedWeight> = result.abliterated.into_iter()
            .filter(|aw| {
                if let WeightKey::MoeDownProj { layer, expert } = &aw.key {
                    !already_abliterated.contains(&(*layer, *expert))
                } else {
                    true
                }
            })
            .collect();

        let skipped = pre_filter_count - new_abliterated.len();
        if skipped > 0 {
            tracing::warn!(
                "Pass {}: safety net filtered {} experts that should have been excluded upstream",
                pass + 1, skipped,
            );
        }

        // Commit abliterated weights permanently to the model
        for aw in &new_abliterated {
            inputs.model.set_weight(&aw.key, &aw.new_weight)?;
            if let WeightKey::MoeDownProj { layer, expert } = &aw.key {
                already_abliterated.insert((*layer, *expert));
            }
        }

        let pass_abliterated_count = new_abliterated.len();
        all_abliterated.extend(new_abliterated);

        // Capture result metadata from this pass (move eval_results, not clone)
        final_scores = result.scores;
        final_threshold = result.threshold_used;
        final_threshold_method = result.threshold_method;
        final_kl = result.kl_divergence;
        final_eval = result.eval_results;

        if pass_abliterated_count == 0 {
            println!("[Multi-pass] No experts abliterated in pass {}. Stopping.", pass + 1);
            break;
        }

        // Check stopping conditions from eval results
        if let Some(ref eval) = final_eval {
            let current_refusal = eval.refusal_rate.rate;

            // Target refusal rate reached
            if current_refusal <= config.target_refusal {
                println!(
                    "[Multi-pass] Target refusal rate {:.1}% reached (current: {:.1}%). Stopping.",
                    config.target_refusal * 100.0,
                    current_refusal * 100.0,
                );
                break;
            }

            // Reasoning guardrail — uses baseline-relative threshold.
            // Stop if reasoning drops more than 10% from the model's own baseline,
            // or below the absolute floor of 40% (whichever is higher).
            if let Some(ref reasoning) = eval.reasoning_canary {
                let pass_rate = reasoning.passed as f32 / reasoning.total.max(1) as f32;
                let threshold = if let Some(baseline) = baseline_reasoning_rate {
                    // 90% of baseline, but at least 40% absolute
                    (0.90 * baseline).max(0.40).min(REASONING_MIN_PASS_RATE)
                } else {
                    REASONING_MIN_PASS_RATE
                };
                if pass_rate < threshold {
                    println!(
                        "[Multi-pass] WARNING: Reasoning pass rate {:.1}% below {:.1}% threshold ({}/{}{}). \
                         Stopping to preserve capabilities.",
                        pass_rate * 100.0,
                        threshold * 100.0,
                        reasoning.passed, reasoning.total,
                        baseline_reasoning_rate
                            .map(|b| format!(", baseline: {:.1}%", b * 100.0))
                            .unwrap_or_default(),
                    );
                    break;
                }
            }

            // KL divergence guardrail (approximate cumulative across all passes).
            // NOTE: This sums per-pass marginal KL values, not the true KL from the
            // original unmodified model. KL divergence is not strictly additive, so this
            // is a conservative approximation. For exact cumulative KL, we'd need to
            // store the original pre-abliteration logits, which is memory-prohibitive.
            if let Some(kl) = final_kl {
                cumulative_kl += kl;
                if cumulative_kl > KL_DIVERGENCE_CEILING {
                    println!(
                        "[Multi-pass] WARNING: Cumulative KL divergence {:.4} exceeds {:.2} threshold. Stopping.",
                        cumulative_kl,
                        KL_DIVERGENCE_CEILING,
                    );
                    break;
                }
            }

            // Diminishing returns
            if let Some(last) = last_refusal_rate {
                let delta = last - current_refusal;
                if delta < MIN_REFUSAL_DELTA {
                    println!(
                        "[Multi-pass] Diminishing returns: refusal delta {:.1}% < {:.0}%. Stopping.",
                        delta * 100.0,
                        MIN_REFUSAL_DELTA * 100.0,
                    );
                    break;
                }
            }

            last_refusal_rate = Some(current_refusal);
        }

        // Prepare for next pass (unless this is the last one)
        if pass + 1 < config.passes {
            println!(
                "\n[Multi-pass] Re-collecting activation statistics for pass {}...",
                pass + 2,
            );

            // Re-collect stats from the now-modified model
            inputs.stats = collect_expert_stats(
                &*inputs.model,
                &inputs.harmful_tokens,
                &inputs.harmless_tokens,
                &inputs.device,
            )?;

            // Rebuild all_expert_weights from the modified model.
            // run_pipeline() clears this vec to free memory, so we must repopulate
            // it by scoring with a quick single-direction pass and extracting weights
            // from the model for any expert with positive routing bias.
            let quick_dirs = build_directions(
                &inputs.stats,
                config.abliteration_mode,
                config.min_activations,
                config.num_directions,
                config.direction_energy,
            )?;
            let quick_scores = score_experts_dispatch(&inputs.stats, &quick_dirs)?;

            let dtype = match &inputs.device {
                candle_core::Device::Cpu => candle_core::DType::F32,
                _ => candle_core::DType::BF16,
            };

            inputs.all_expert_weights = quick_scores
                .iter()
                .filter(|s| s.routing_bias > 0.0)
                .filter_map(|s| {
                    let key = WeightKey::MoeDownProj {
                        layer: s.layer_idx,
                        expert: s.expert_idx,
                    };
                    inputs.model.get_weight(&key).ok().map(|w| {
                        let w = w.to_dtype(dtype).unwrap_or(w);
                        (s.layer_idx, s.expert_idx, w)
                    })
                })
                .collect();

            println!(
                "       Loaded {} expert weights for pass {}",
                inputs.all_expert_weights.len(),
                pass + 2,
            );
        }
    }

    // Deduplicate abliterated weights: if the same key was abliterated in multiple
    // passes, keep only the last version (which represents the cumulative effect).
    let mut seen = std::collections::HashSet::new();
    let mut deduped: Vec<AbliteratedWeight> = Vec::new();
    for aw in all_abliterated.into_iter().rev() {
        if seen.insert(aw.key.clone()) {
            deduped.push(aw);
        }
    }
    deduped.reverse();

    Ok(PipelineResult {
        num_abliterated: deduped.len(),
        abliterated: deduped,
        scores: final_scores,
        threshold_used: final_threshold,
        threshold_method: final_threshold_method,
        kl_divergence: final_kl,
        eval_results: final_eval,
    })
}

/// Core evaluation logic shared by both weight-swap and mask-only paths.
///
/// Takes a `&dyn MoeModel` (already configured — either with weights swapped
/// in or with an expert mask active) and runs the full eval battery.
fn run_eval_core(
    model: &dyn MoeModel,
    harmful_tokens: &[Tensor],
    harmless_tokens: &[Tensor],
    domain_datasets: &[NamedDomain],
    eval_prompts: &[String],
    utility_questions: &[UtilityQuestion],
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    kl_divergence: f32,
) -> Result<EvalResults> {
    // Refusal rate on harmful prompts (generation-based)
    println!("  Measuring refusal rate on harmful prompts...");
    let mut refused = 0;
    for prompt_tokens in harmful_tokens {
        let ids: Vec<u32> = prompt_tokens.to_vec1()?;
        let result = detect_refusal_generated(model, tokenizer, &ids, device)?;
        if result.refused {
            refused += 1;
        }
    }
    let refusal_rate = RefusalRate {
        refused,
        total: harmful_tokens.len(),
        rate: if harmful_tokens.is_empty() {
            0.0
        } else {
            refused as f32 / harmful_tokens.len() as f32
        },
    };
    println!(
        "  Refusal rate: {} / {} ({:.1}%)",
        refusal_rate.refused,
        refusal_rate.total,
        refusal_rate.rate * 100.0,
    );

    // Over-refusal on harmless prompts
    let over_refusal = if !harmless_tokens.is_empty() {
        println!("  Measuring over-refusal on harmless prompts...");
        let mut over_refused = 0;
        for prompt_tokens in harmless_tokens {
            let ids: Vec<u32> = prompt_tokens.to_vec1()?;
            let result = detect_refusal_generated(model, tokenizer, &ids, device)?;
            if result.refused {
                over_refused += 1;
            }
        }
        let rate = over_refused as f32 / harmless_tokens.len() as f32;
        println!(
            "  Over-refusal: {} / {} ({:.1}%)",
            over_refused,
            harmless_tokens.len(),
            rate * 100.0,
        );
        Some(OverRefusalRate {
            refused: over_refused,
            total: harmless_tokens.len(),
            rate,
        })
    } else {
        None
    };

    // Reasoning canary
    println!("  Running reasoning canary...");
    let (canary_passed, canary_total) = run_reasoning_canary(model, tokenizer, device)?;
    let reasoning_canary = Some(ReasoningResult {
        passed: canary_passed,
        total: canary_total,
    });
    println!(
        "  Reasoning canary: {} / {} passed",
        canary_passed, canary_total,
    );

    // Per-domain refusal matrix
    let mut domain_refusal: Vec<DomainRefusalRate> = Vec::new();

    if !domain_datasets.is_empty() {
        for domain in domain_datasets {
            if domain.eval.is_empty() {
                continue;
            }
            println!(
                "  Evaluating domain '{}' ({} eval prompts)...",
                domain.name,
                domain.eval.len()
            );
            let result = evaluate_domain_refusals(
                model,
                &domain.name,
                &domain.eval,
                tokenizer,
                device,
            )?;
            println!(
                "  Domain '{}': {} / {} refused ({:.1}%)",
                result.domain, result.refused, result.total,
                result.rate * 100.0,
            );
            domain_refusal.push(result);
        }
    } else if !eval_prompts.is_empty() {
        println!(
            "  Evaluating domain refusal rates ({} eval prompts)...",
            eval_prompts.len()
        );
        let result =
            evaluate_domain_refusals(model, "default", eval_prompts, tokenizer, device)?;
        println!(
            "  Domain '{}': {} / {} refused ({:.1}%)",
            result.domain, result.refused, result.total, result.rate * 100.0,
        );
        domain_refusal.push(result);
    }

    // Utility benchmark
    let utility_results = if !utility_questions.is_empty() {
        println!(
            "  Running utility benchmark ({} questions)...",
            utility_questions.len(),
        );
        let results = utility::run_utility_benchmark(model, tokenizer, utility_questions, device)?;
        println!(
            "  Utility: {} / {} ({:.1}%)",
            results.passed, results.total, results.rate * 100.0,
        );
        for cat in &results.per_category {
            println!(
                "    {}: {} / {} ({:.1}%)",
                cat.category, cat.passed, cat.total, cat.rate * 100.0,
            );
        }
        Some(results)
    } else {
        None
    };

    Ok(EvalResults {
        refusal_rate,
        kl_divergence,
        reasoning_canary,
        domain_refusal,
        over_refusal,
        utility: utility_results,
    })
}

/// Run the full evaluation suite with abliterated weights swapped in.
///
/// Delegates to `swap_measure_swap` for weight management — this ensures
/// weights are always restored even if evaluation fails mid-way, and inherits
/// the duplicate-key guard.
fn run_eval_suite(
    model: &mut dyn MoeModel,
    abliterated: &[AbliteratedWeight],
    harmful_tokens: &[Tensor],
    harmless_tokens: &[Tensor],
    domain_datasets: &[NamedDomain],
    eval_prompts: &[String],
    utility_questions: &[UtilityQuestion],
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    kl_divergence: f32,
) -> Result<EvalResults> {
    println!("\nRunning evaluation suite...");

    swap_measure_swap(model, abliterated, |model| {
        run_eval_core(
            model,
            harmful_tokens,
            harmless_tokens,
            domain_datasets,
            eval_prompts,
            utility_questions,
            tokenizer,
            device,
            kl_divergence,
        )
    })
}
