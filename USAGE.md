# Flay — Usage Guide

## Quick Start

```bash
# Basic — auto-detect GPU, auto threshold, multi-projected mode
flay Qwen/Qwen3-30B-A3B-Instruct-2507

# With evaluation and reporting
flay Qwen/Qwen3-30B-A3B-Instruct-2507 --eval --report

# Stability-based selection + multi-pass iteration
flay Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --stable-k 5 --passes 3 --target-refusal 0.1 --eval

# Bayesian optimization (searches directions, thresholds, modes, and strengths)
flay Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --optimize --optimize-mode bayesian --trials 30 --eval
```

## Features

- **Per-expert scoring** — routing bias + refusal projection identify guilty experts
- **Four abliteration modes** — single, multi, projected, multi-projected (`--abliteration-mode`)
- **Stability-based expert selection (SES)** — K-fold bootstrap for robust identification (`--stable-k`, `--stable-top-n`)
- **HCDG/HRCG decomposition** — separate detection from enforcement experts (`--jailbreak-data`)
- **Multi-pass iteration** — re-extract and abliterate until target refusal rate with guardrails (`--passes`, `--target-refusal`)
- **Router weight attenuation** — reduce expert activation via gate scaling (`--route-strength`)
- **Residual-stream abliteration** — o_proj abliteration across all decoder layers (`--residual`)
- **Expert masking diagnostics** — test selection without modifying weights (`--mask-only`)
- **Decode-time expert capture diagnostics** — compare prefill vs decode experts and run decode-targeted masking (`--capture-decode`)
- **Grid search and Bayesian TPE optimization** — automatic hyperparameter tuning (`--optimize`, `--optimize-mode`)
- **Comprehensive eval suite** — refusal rate, over-refusal, reasoning canary, KL divergence, domain breakdown
- **Utility benchmark** — auto-graded capability tests (`--utility-benchmark`)
- **Reproducibility metadata** — git hash, prompt hash, generation strategy in reports (`--report`)
- **Multi-domain abliteration** — target specific harm categories (`--domain`)
- **Eval-only mode** — benchmark without saving model files (`--no-save`)

## CLI Reference

| Option | Default | Description |
|:-------|:--------|:------------|
| `MODEL` | (required) | HuggingFace model ID or local path |
| `--output`, `-o` | `{model}-flay/` | Output directory |
| `--device`, `-d` | `auto` | Compute device: `metal`, `cuda`, `cpu` |
| **Abliteration** | | |
| `--abliteration-mode` | `multi-projected` | `single`, `multi`, `projected`, `multi-projected` |
| `--threshold`, `-t` | auto (elbow) | Expert refusal score threshold |
| `--strength-min` | `0.5` | Minimum abliteration strength |
| `--num-directions` | `5` | SVD directions to extract (multi modes) |
| `--direction-energy` | `0.1` | Energy threshold for SVD filtering |
| `--min-activations` | `3` | Minimum activation count for per-expert directions |
| **Advanced Selection** | | |
| `--stable-k` | `1` | Bootstrap samples for SES (>1 enables stability selection) |
| `--stable-top-n` | `20` | Top-N experts per bootstrap sample |
| `--jailbreak-data` | — | Jailbreak prompt directory for HCDG/HRCG decomposition |
| `--mask-only` | off | Mask experts instead of abliterating (diagnostic, requires `--eval`) |
| `--capture-decode` | — | Capture first N decode steps for decode-vs-prefill analysis (requires `--mask-only --eval`) |
| **Multi-Pass** | | |
| `--passes` | `1` | Maximum abliteration passes (re-extracts directions each pass) |
| `--target-refusal` | `0.5` | Stop when refusal rate drops below this |
| **Router & Residual** | | |
| `--route-strength` | `0.0` | Gate weight attenuation (0.0=off, 1.0=zero gate row) |
| `--residual` | off | Enable residual-stream o_proj abliteration |
| `--residual-strength` | `1.0` | o_proj abliteration strength |
| `--moe-strength` | `0.3` | MoE expert strength when combined with `--residual` |
| **Optimization** | | |
| `--optimize` | off | Enable optimization trials |
| `--optimize-mode` | `grid` | `grid` or `bayesian` |
| `--trials` | `15` | Number of optimization trials |
| **Evaluation** | | |
| `--eval` | off | Run full evaluation suite after abliteration |
| `--skip-eval` | off | Skip evaluation |
| `--utility-benchmark` | — | Path to utility benchmark JSONL file |
| `--kl-eval-size` | `20` | Harmless prompts for KL divergence measurement |
| **Data & Output** | | |
| `--domain` | — | Domain prompt directory (repeatable for multi-domain) |
| `--harmful-dataset` | embedded | Path to harmful prompts |
| `--harmless-dataset` | embedded | Path to harmless prompts |
| `--revision` | `main` | HuggingFace model revision |
| `--report` | off | Save JSON, markdown, and model card reports |
| `--no-save` | off | Skip saving model (eval-only benchmarking) |

## Output

Flay writes a drop-in replacement model to the output directory:

```
{model}-flay/
  model-00001-of-NNNNN.safetensors  # modified weight shards
  model.safetensors.index.json       # shard index (if sharded)
  config.json                        # copied from original
  tokenizer.json                     # copied from original
  tokenizer_config.json              # copied from original
  flay-report.json                   # (with --report)
  flay-report.md                     # (with --report)
  README.md                          # HuggingFace model card (with --report)
```

The output model is compatible with any framework that supports the original architecture (transformers, vLLM, llama.cpp, etc.).

## Advanced Strategies

### Stability-Based Expert Selection (SES)

Single-pass scoring can be noisy — an expert's ranking may shift with different prompt subsets. SES runs K bootstrap resamples of the expert scores, selects the top-N in each sample, and intersects the results. Only experts that consistently rank in the top-N across all K folds are selected. Falls back to frequency-based selection (appearing in at least K-1 folds) if strict intersection yields too few experts.

```bash
flay model --stable-k 5 --stable-top-n 20 --eval
```

### HCDG/HRCG Decomposition

Not all safety-related experts serve the same function. Some detect harmful content (active on both regular and jailbreak-reformulated prompts), while others enforce refusal (active only on regular harmful prompts, bypassed by jailbreaks).

- **HCDG** (Harmful Content Detection Group) — experts in both regular and jailbreak stable sets
- **HRCG** (Harmful Response Control Group) — experts only in the regular stable set

Flay targets only HRCG for abliteration, preserving the model's ability to recognize harmful content while removing its refusal enforcement.

Requires `--stable-k > 1` and a jailbreak prompt directory containing `harmful.txt`.

```bash
flay model --stable-k 5 --jailbreak-data ./data/eval/jailbreak --eval
```

### Multi-Pass Iteration

A single abliteration pass may not remove all refusal behavior — the model's remaining experts can compensate. Multi-pass mode commits weights after each pass, re-collects activation statistics from the modified model, and identifies newly-emerged guilty experts.

Guardrail stops prevent over-abliteration:

- **Target refusal** — stop when refusal rate drops below threshold
- **Reasoning** — stop if reasoning pass rate drops below 80%
- **KL divergence** — stop if cumulative KL exceeds 0.15
- **Diminishing returns** — stop if refusal delta < 2% between passes

Requires `--eval` for guardrails to function.

```bash
flay model --passes 5 --target-refusal 0.1 --eval
```

### Router Weight Attenuation

Complementary to `down_proj` abliteration. Scales the router gate weight row for each guilty expert by `(1 - strength)`, reducing the probability the router selects that expert. At strength 1.0 the gate row is zeroed (expert never activated). Can be combined with abliteration for a two-pronged approach.

```bash
flay model --route-strength 0.5 --eval
```

### Residual-Stream Abliteration

Extends abliteration beyond MoE experts to the attention output projection (`o_proj`) across all decoder layers. Uses per-layer refusal directions from the residual stream with a band-based strength schedule (early layers get lighter treatment, peak refusal layers 24–39 get full strength). When `--residual` is active, MoE expert strength defaults to 0.3 for a balanced hybrid approach.

Cannot be combined with `--optimize`.

```bash
flay model --residual --residual-strength 1.0 --moe-strength 0.3 --eval
```

### Expert Masking (Diagnostic)

Sets gate logits to `-inf` for identified experts during eval, preventing their activation without permanent weight changes. Useful for validating that the right experts were identified before committing to abliteration.

```bash
flay model --mask-only --eval
```

## Evaluation

With `--eval`, Flay runs a comprehensive post-abliteration evaluation:

- **Refusal rate** — generation-based refusal detection on harmful prompts
- **Over-refusal rate** — false positive refusal on harmless prompts
- **Reasoning canary** — math/logic questions to verify capability preservation
- **KL divergence** — distributional shift from the original model
- **Domain refusal rates** — per-domain breakdown when using `--domain`
- **Utility benchmark** — auto-graded capability tests (`--utility-benchmark`)

Reports (`--report`) include reproducibility metadata: git commit hash (with dirty flag), model revision, FNV-1a prompt content hash, and generation strategy.

## Building

```bash
# CPU only
cargo build --release

# With Metal (Apple Silicon)
cargo build --release --features metal

# With CUDA
cargo build --release --features cuda
```

Requires Rust 2024 edition (1.85+).

## Supported Architectures

- Qwen3 MoE (Qwen3-30B-A3B, Qwen3-235B-A22B)

The model implementation is built from scratch using [candle](https://github.com/huggingface/candle) to support per-expert activation capture hooks that the standard candle-transformers model doesn't expose. Adding new MoE architectures requires implementing the model forward pass with capture support.
