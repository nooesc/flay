use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;

use flay::abliterate::orthogonalize::AbliterationMode;
use flay::analysis::activations::collect_expert_stats;
use flay::analysis::prompts::{load_datasets, load_domains_with_names, load_prompts_from_file};
use flay::io::report::{FlayReport, ReproducibilityMetadata};
use flay::io::safetensors::{copy_model_files, load_tensors_selective, save_model};
use flay::model::arch::MoeModel;
use flay::model::config::Qwen3MoeConfig;
use flay::model::deepseek_config::DeepSeekV3Config;
use flay::model::qwen3_moe::Qwen3MoeModel;
use flay::optimize::OptimizeMode;
use flay::pipeline::{run_pipeline_iterative, PipelineConfig, PipelineInputs};

#[derive(Parser)]
#[command(name = "flay")]
#[command(about = "Per-expert abliteration for Mixture-of-Experts language models")]
struct Cli {
    /// Model path or HuggingFace model ID
    model: String,

    /// Output directory (default: {model}-flay/)
    #[arg(short, long)]
    output: Option<String>,

    /// Device: metal, cuda, cpu (auto-detect by default)
    #[arg(short, long, default_value = "auto")]
    device: String,

    /// Expert refusal score threshold (default: auto)
    #[arg(short, long)]
    threshold: Option<f32>,

    /// Enable few-trial optimization for best quality
    #[arg(long)]
    optimize: bool,

    /// Optimization strategy (grid or bayesian)
    #[arg(long, default_value = "grid")]
    optimize_mode: OptimizeMode,

    /// Number of optimization trials (default: 15)
    #[arg(long, default_value = "15")]
    trials: usize,

    /// Save detailed per-expert analysis report
    #[arg(long)]
    report: bool,

    /// Path to harmful prompts dataset
    #[arg(long)]
    harmful_dataset: Option<String>,

    /// Path to harmless prompts dataset
    #[arg(long)]
    harmless_dataset: Option<String>,

    /// Domain prompt directory (contains harmful.txt, harmless.txt, eval.txt).
    /// Can be specified multiple times for multi-domain abliteration.
    #[arg(long = "domain", value_name = "DIR")]
    domains: Vec<String>,

    /// HuggingFace model revision (branch, tag, or commit hash)
    #[arg(long, default_value = "main")]
    revision: String,

    /// Minimum activation count per expert for per-expert refusal directions (default: 3)
    #[arg(long, default_value = "3")]
    min_activations: usize,

    /// Number of harmless prompts to use for KL divergence evaluation (default: 20)
    #[arg(long, default_value = "20")]
    kl_eval_size: usize,

    /// Run full evaluation suite after abliteration
    #[arg(long, conflicts_with = "skip_eval")]
    eval: bool,

    /// Skip evaluation (faster iteration)
    #[arg(long, conflicts_with = "eval")]
    skip_eval: bool,

    /// Abliteration mode: single, multi, projected, multi-projected
    #[arg(long, default_value = "multi-projected")]
    abliteration_mode: AbliterationMode,

    /// Number of SVD directions to extract (multi/multi-projected modes)
    #[arg(long, default_value = "5")]
    num_directions: usize,

    /// Energy threshold for SVD direction filtering (0.01-0.5, default: 0.1)
    #[arg(long, default_value = "0.1")]
    direction_energy: f32,

    /// Minimum abliteration strength (default: 0.5, range 0.0-1.0)
    #[arg(long, default_value = "0.5")]
    strength_min: f32,

    /// Enable residual-stream (o_proj) abliteration across all decoder layers
    #[arg(long)]
    residual: bool,

    /// Global strength multiplier for residual-stream o_proj abliteration (default: 1.0)
    #[arg(long, default_value = "1.0", value_parser = parse_strength)]
    residual_strength: f32,

    /// MoE expert strength multiplier when used with --residual (default: 0.3)
    /// Without --residual, expert strength is controlled by --strength-min
    #[arg(long, default_value = "0.3", value_parser = parse_strength)]
    moe_strength: f32,

    /// Skip saving the abliterated model (eval-only mode for benchmarking)
    #[arg(long)]
    no_save: bool,

    /// Number of bootstrap samples for stability-based expert selection.
    /// When >1, uses K-fold SES instead of single-pass scoring.
    #[arg(long, default_value_t = 1)]
    stable_k: usize,

    /// Top-N experts per bootstrap sample before intersection.
    #[arg(long, default_value_t = 20)]
    stable_top_n: usize,

    /// Mask identified experts instead of abliterating weights.
    /// Runs eval with masked experts for diagnostic comparison.
    #[arg(long, requires = "eval")]
    mask_only: bool,

    /// Capture expert routing during prefill + first N decode steps.
    /// Diagnostic mode: requires --mask-only and --eval.
    #[arg(long, requires_all = ["mask_only", "eval"])]
    capture_decode: Option<usize>,

    /// Path to jailbreak prompt directory for HCDG/HRCG decomposition.
    /// Must contain harmful.txt with jailbreak-reformulated harmful prompts.
    /// When provided with --stable-k > 1, decomposes stable experts into
    /// detection (HCDG) and control (HRCG) groups, targeting only HRCG.
    #[arg(long, value_name = "DIR")]
    jailbreak_data: Option<String>,

    /// Path to utility benchmark JSONL file (auto-gradable capability tests)
    #[arg(long, value_name = "FILE")]
    utility_benchmark: Option<String>,

    /// Maximum number of abliteration passes. Each pass re-extracts directions
    /// from the modified model and abliterates newly-identified experts.
    #[arg(long, default_value_t = 1, value_parser = parse_passes)]
    passes: usize,

    /// Target refusal rate — stop iterating when refusal drops below this.
    #[arg(long, default_value_t = 0.5)]
    target_refusal: f32,

    /// Gate weight attenuation strength for safety-critical experts.
    /// 0.0 = disabled, 0.5 = halve gate magnitude, 1.0 = zero gate row.
    #[arg(long, default_value_t = 0.0, value_parser = parse_strength)]
    route_strength: f32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    // Guard: --residual + --optimize is not supported
    if cli.residual && cli.optimize {
        anyhow::bail!(
            "--residual and --optimize cannot be used together. \
             Optimization for the hybrid approach is not yet implemented."
        );
    }

    println!("flay v{}", env!("CARGO_PKG_VERSION"));
    println!("Per-expert MoE abliteration\n");
    println!("Model:  {}", cli.model);
    println!("Device: {}", cli.device);
    println!("Mode:   {}", cli.abliteration_mode);
    if cli.optimize {
        println!("Optimize: {:?} ({} trials)", cli.optimize_mode, cli.trials);
    } else {
        println!("Optimize: off (single-pass)");
    }
    if cli.residual {
        println!(
            "Residual: o_proj strength={:.2}, MoE strength={:.2}",
            cli.residual_strength, cli.moe_strength
        );
    } else {
        println!("Residual: off (MoE-only mode)");
    }
    if cli.stable_k > 1 {
        println!("SES:      K={}, top_n={}", cli.stable_k, cli.stable_top_n);
    }
    if cli.mask_only {
        println!("Mode:     mask-only (diagnostic, no weight changes)");
    }
    if cli.route_strength > 0.0 {
        println!("Routing:  gate attenuation strength={:.2}", cli.route_strength);
    }
    if cli.passes > 1 {
        println!(
            "Passes:   {} (target refusal: {:.1}%)",
            cli.passes,
            cli.target_refusal * 100.0,
        );
    }
    println!();

    // 1. Select device + dtype
    let device = flay::device::select_device(&cli.device)?;
    let dtype = match &device {
        Device::Cpu => DType::F32,
        _ => DType::BF16,
    };
    println!("[1/9] Device: {:?}, dtype: {:?}", device, dtype);

    // 2. Resolve model files
    println!("[2/9] Resolving model files...");
    let model_files = flay::io::hub::resolve_model(&cli.model, &cli.revision)?;
    println!("       Config:    {}", model_files.config_path.display());
    println!("       Tokenizer: {}", model_files.tokenizer_path.display());
    println!("       Weights:   {} shard(s)", model_files.weight_paths.len());

    // 3. Parse config — detect architecture
    let config_str = std::fs::read_to_string(&model_files.config_path)
        .context("Failed to read config.json")?;
    let config_raw: serde_json::Value =
        serde_json::from_str(&config_str).context("Failed to parse config.json")?;
    let model_type = config_raw["model_type"].as_str().unwrap_or("unknown");
    println!("       model_type: {model_type}");

    // 4. Load model
    println!("[3/9] Loading model weights (mmap)...");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&model_files.weight_paths, dtype, &device)?
    };

    let model: Box<dyn MoeModel> = match model_type {
        "qwen3_moe" | "qwen2_moe" => {
            let config: Qwen3MoeConfig =
                serde_json::from_str(&config_str).context("Failed to parse Qwen3 config")?;
            config.validate().context("Invalid Qwen3 model config")?;
            println!(
                "       Architecture: {} layers, {} MoE experts, top-{}",
                config.num_hidden_layers, config.num_experts, config.num_experts_per_tok,
            );
            Box::new(Qwen3MoeModel::new(&config, vb)?)
        }
        "deepseek_v3" => {
            let config: DeepSeekV3Config =
                serde_json::from_str(&config_str).context("Failed to parse DeepSeek-V3 config")?;
            config.validate().context("Invalid DeepSeek-V3 model config")?;
            println!(
                "       Architecture: DeepSeek-V3, {} layers, {} routed experts + {} shared, top-{}",
                config.num_hidden_layers,
                config.n_routed_experts,
                config.n_shared_experts,
                config.num_experts_per_tok,
            );
            anyhow::bail!(
                "DeepSeek-V3 architecture detected and config parsed successfully, \
                 but the full model implementation (MLA attention + shared experts) \
                 is not yet complete."
            );
        }
        other => {
            anyhow::bail!("Unsupported model architecture: {other}");
        }
    };

    let moe_indices = model.moe_layer_indices();
    println!(
        "       Model loaded: {} MoE layers at {:?}",
        moe_indices.len(),
        moe_indices,
    );

    // 5. Load and tokenize prompt datasets
    println!("[4/9] Loading prompt datasets...");
    let (datasets, domain_datasets) = if !cli.domains.is_empty() {
        let (ds, named) = load_domains_with_names(&cli.domains)?;
        (ds, named)
    } else {
        let ds = load_datasets(cli.harmful_dataset.as_deref(), cli.harmless_dataset.as_deref())?;
        (ds, Vec::new())
    };
    println!(
        "       {} harmful, {} harmless, {} eval prompts",
        datasets.harmful.len(),
        datasets.harmless.len(),
        datasets.eval.len(),
    );

    // Compute prompt hash for reproducibility before datasets are partially moved.
    let prompt_hash = hash_prompts(&datasets.harmful, &datasets.harmless, &datasets.eval);

    let tokenizer = tokenizers::Tokenizer::from_file(&model_files.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    let harmful_tokens = tokenize_prompts(&datasets.harmful, &*model, &tokenizer, &device)?;
    let harmless_tokens = tokenize_prompts(&datasets.harmless, &*model, &tokenizer, &device)?;
    println!(
        "       Tokenized: {} harmful, {} harmless sequences",
        harmful_tokens.len(),
        harmless_tokens.len(),
    );

    // Load and tokenize jailbreak prompts if --jailbreak-data is provided
    let jailbreak_tokens = if let Some(ref jailbreak_dir) = cli.jailbreak_data {
        let jailbreak_path = std::path::Path::new(jailbreak_dir).join("harmful.txt");
        println!("       Loading jailbreak prompts from {}...", jailbreak_path.display());
        let jailbreak_prompts = load_prompts_from_file(&jailbreak_path)
            .with_context(|| format!("Failed to load jailbreak prompts from {}", jailbreak_path.display()))?;
        let tokens = tokenize_prompts(&jailbreak_prompts, &*model, &tokenizer, &device)?;
        println!(
            "       Tokenized: {} jailbreak harmful sequences",
            tokens.len(),
        );
        Some(tokens)
    } else {
        None
    };

    // 6. Collect expert activation statistics
    println!("[5/9] Phase 1: Collecting expert activation statistics...");
    let stats = collect_expert_stats(&*model, &harmful_tokens, &harmless_tokens, &device)?;
    println!(
        "       Collected stats across {} MoE layers",
        stats.moe_layer_indices.len(),
    );

    // Build the set of expert weight keys (only down_proj, not full model).
    // Use routing_bias > 0 as the filter: any expert activated more by harmful
    // than harmless prompts could be flagged by any scoring mode (single or multi).
    // This avoids under-loading weights when multi-direction scoring detects
    // experts that single-direction scoring would miss.
    let scores_for_keys =
        flay::abliterate::scoring::score_experts_dispatch(
            &stats,
            &flay::abliterate::direction_set::DirectionSet::Single(
                flay::abliterate::directions::compute_refusal_directions(&stats, cli.min_activations)?,
            ),
        )?;

    let needed_keys: std::collections::HashSet<String> = scores_for_keys
        .iter()
        .filter(|s| s.routing_bias > 0.0)
        .map(|s| model.weight_key(s.layer_idx, s.expert_idx))
        .collect();

    println!("       Loading {} expert weights for abliteration...", needed_keys.len());
    let expert_tensors =
        load_tensors_selective(&model_files.weight_paths, &needed_keys, &device)?;

    let all_expert_weights: Vec<(usize, usize, Tensor)> = scores_for_keys
        .iter()
        .filter(|s| s.routing_bias > 0.0)
        .filter_map(|s| {
            let key = model.weight_key(s.layer_idx, s.expert_idx);
            expert_tensors
                .get(&key)
                .map(|t| (s.layer_idx, s.expert_idx, t.to_dtype(dtype).unwrap_or_else(|_| t.clone())))
        })
        .collect();

    // 7. Run the pipeline
    let pipeline_config = PipelineConfig {
        abliteration_mode: cli.abliteration_mode,
        min_activations: cli.min_activations,
        num_directions: cli.num_directions,
        direction_energy: cli.direction_energy,
        threshold: cli.threshold,
        strength_min: cli.strength_min,
        optimize: cli.optimize,
        optimize_mode: cli.optimize_mode,
        trials: cli.trials,
        kl_eval_size: cli.kl_eval_size,
        run_eval: cli.eval && !cli.skip_eval,
        residual: cli.residual,
        residual_strength: cli.residual_strength,
        moe_strength: cli.moe_strength,
        stable_k: cli.stable_k,
        stable_top_n: cli.stable_top_n,
        mask_only: cli.mask_only,
        jailbreak_data: cli.jailbreak_data.clone(),
        passes: cli.passes,
        target_refusal: cli.target_refusal,
        route_strength: cli.route_strength,
        capture_decode: cli.capture_decode,
    };

    let utility_questions = if let Some(ref path) = cli.utility_benchmark {
        println!("       Loading utility benchmark: {path}");
        flay::eval::utility::load_utility_benchmark(path)?
    } else {
        Vec::new()
    };

    let mut pipeline_inputs = PipelineInputs {
        model,
        stats,
        all_expert_weights,
        harmful_tokens,
        harmless_tokens,
        domain_datasets,
        eval_prompts: datasets.eval,
        utility_questions,
        tokenizer,
        device,
        jailbreak_tokens,
        exclude_experts: std::collections::HashSet::new(),
    };

    let result = run_pipeline_iterative(&pipeline_config, &mut pipeline_inputs)?;

    if result.abliterated.is_empty() {
        return Ok(());
    }

    // Extract values we need, then drop the model to free memory before saving
    let save_device = pipeline_inputs.device.clone();
    let num_experts = pipeline_inputs.model.num_experts();
    let total_experts = num_experts * pipeline_inputs.model.moe_layer_indices().len();
    drop(pipeline_inputs);

    // 8. Save model (unless --no-save)
    let output_dir = match &cli.output {
        Some(dir) => PathBuf::from(dir),
        None => {
            let basename = cli.model.rsplit('/').next().unwrap_or(&cli.model);
            PathBuf::from(format!("{basename}-flay"))
        }
    };

    if cli.no_save {
        println!("[9/9] Skipping model save (--no-save)");
    } else {
        println!("[9/9] Saving abliterated model to {}...", output_dir.display());
        save_model(
            &model_files.weight_paths,
            &result.abliterated,
            &output_dir,
            &save_device,
        )?;
        copy_model_files(&model_files.model_dir, &output_dir)?;
        println!("       Model saved successfully.");
    }

    // 9. Generate report
    if cli.report {
        println!("\nGenerating reports...");
        let reproducibility = Some(ReproducibilityMetadata {
            git_commit: git_commit_hash(),
            model_revision: model_files.revision.clone(),
            prompt_hash: prompt_hash.clone(),
            generation_strategy: "greedy (argmax)".to_string(),
        });
        let mut report = FlayReport::new(
            cli.model.clone(),
            model_files.revision.clone(),
            total_experts,
            &result.scores,
            &result.abliterated,
            result.threshold_used,
            result.threshold_method.clone(),
            result.kl_divergence,
            cli.abliteration_mode.to_string(),
            reproducibility,
        );
        report.eval = result.eval_results.clone();
        report.save_json(&output_dir)?;
        report.save_markdown(&output_dir)?;
        report.save_model_card(&output_dir)?;
        println!("       Reports saved to {}", output_dir.display());
    }

    // 10. Print summary
    println!("\n{}", "=".repeat(60));
    println!("  Flay abliteration complete");
    println!("{}", "=".repeat(60));
    println!("  Model:             {}", cli.model);
    println!("  Mode:              {}", cli.abliteration_mode);
    if cli.residual {
        let o_proj_count = result
            .abliterated
            .iter()
            .filter(|aw| matches!(&aw.key, flay::abliterate::weight_key::WeightKey::AttnOProj { .. }))
            .count();
        let expert_count = result
            .abliterated
            .iter()
            .filter(|aw| {
                matches!(&aw.key, flay::abliterate::weight_key::WeightKey::MoeDownProj { .. })
            })
            .count();
        println!("  o_proj abliterated: {} layers", o_proj_count);
        println!("  Experts abliterated: {} / {}", expert_count, total_experts);
        println!("  Residual strength: {:.2}", cli.residual_strength);
        println!("  MoE strength:      {:.2}", cli.moe_strength);
    } else {
        println!(
            "  Experts abliterated: {} / {}",
            result.abliterated.len(),
            total_experts,
        );
    }
    println!(
        "  Threshold:         {:.4} ({})",
        result.threshold_used, result.threshold_method
    );
    if let Some(kl) = result.kl_divergence {
        println!("  KL divergence:     {:.6}", kl);
    }
    println!("  Output:            {}", output_dir.display());
    println!("{}", "=".repeat(60));

    Ok(())
}

/// Parse --passes value, requiring at least 1.
fn parse_passes(s: &str) -> Result<usize, String> {
    let v: usize = s.parse().map_err(|e| format!("{e}"))?;
    if v == 0 {
        return Err("--passes must be at least 1".to_string());
    }
    Ok(v)
}

/// Parse a strength value in the range [0.0, 1.0].
fn parse_strength(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("value {v} not in range 0.0..=1.0"));
    }
    Ok(v)
}

/// Compute a stable hash of the prompt content for reproducibility.
///
/// Uses FNV-1a (64-bit) which is deterministic across Rust versions and platforms,
/// unlike `DefaultHasher` (SipHash) whose algorithm is not guaranteed stable.
fn hash_prompts(harmful: &[String], harmless: &[String], eval: &[String]) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut h = FNV_OFFSET;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    };

    for p in harmful {
        feed(p.as_bytes());
        feed(b"\x00"); // string separator
    }
    feed(b"\xff"); // list boundary
    for p in harmless {
        feed(p.as_bytes());
        feed(b"\x00");
    }
    feed(b"\xfe"); // list boundary
    for p in eval {
        feed(p.as_bytes());
        feed(b"\x00");
    }

    format!("{:016x}", h)
}

/// Retrieve the short git commit hash of the current working tree (if in a git repo).
/// Appends "-dirty" when uncommitted changes are present.
fn git_commit_hash() -> Option<String> {
    let short = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())?;

    let dirty = std::process::Command::new("git")
        .args(["diff", "--quiet", "HEAD"])
        .status()
        .map(|s| !s.success())
        .unwrap_or(false);

    Some(if dirty {
        format!("{short}-dirty")
    } else {
        short
    })
}

/// Tokenize a list of prompts into tensors, wrapping each in the model's chat template.
fn tokenize_prompts(
    prompts: &[String],
    model: &dyn MoeModel,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
) -> Result<Vec<Tensor>> {
    prompts
        .iter()
        .map(|prompt| {
            let formatted = model.format_chat_prompt(prompt);
            let encoding = tokenizer
                .encode(formatted.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
            Tensor::new(encoding.get_ids(), device).map_err(Into::into)
        })
        .collect()
}
