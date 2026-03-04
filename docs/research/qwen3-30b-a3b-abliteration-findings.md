# Abliteration Research Findings — Qwen3-30B-A3B-Instruct-2507

**Date:** 2026-03-03
**Model:** Qwen/Qwen3-30B-A3B-Instruct-2507 (48 MoE layers, 128 experts/layer, top-8 routing, 6,144 total experts)
**System:** Mac Studio M3 Ultra, 256GB RAM, Metal (BF16)
**Goal:** Minimize chat-mode refusal while preserving agent functionality

## Executive Summary

Phase 2A (Runs 1-8) systematically tested MoE expert abliteration and residual-stream abliteration on Qwen3-30B-A3B. **MoE expert surgery is ineffective for this model** — no combination of modes, strengths, selection strategies, or capture phases reduced chat refusal. Residual `o_proj` abliteration was the only lever that moved refusal, with a ceiling around 75%. A mask-only causality test revealed that the experts identified as "refusal control" were actually **compliance pathways** — blocking them increased refusal. Run 8 confirmed a strong prefill/decode mismatch (Jaccard 0.10), but decode-targeted masking still failed.

Phase 2B (Run 9) then moved to attention-only LoRA (`o_proj` + `v_proj`, last 24 layers) and reduced refusal from 89.0% to 9.9% with 0% over-refusal on the MLX eval harness. **Conclusion: MoE surgery is closed for this model, attention fine-tuning is viable.**

## Complete Results

| Run | Mode | Target | Experts | Refusal | Over-ref | Reasoning | KL |
|-----|------|--------|---------|---------|----------|-----------|-----|
| Baseline (unmodified) | — | — | — | ~100% | 0% | 54.3% | — |
| 1 | projected + SES | MoE down_proj | 3 | 90.1% | — | 54.3% | 0.0003 |
| 2 | multi-proj + router atten | MoE down_proj + gate | 71 | 89.1% | 0% | 57.1% | 0.0123 |
| 3 | multi (aggressive) | MoE down_proj + gate | 83 | 89.1% | 0% | 60.0% | 0.0522 |
| 4 | residual-only 0.35 | 48 o_proj layers | 0 | 84.2% | 0% | 54.3% | 0.0056 |
| 5 | residual-only 0.60 | 48 o_proj layers | 0 | 79.2% | 0% | 54.3% | 0.0068 |
| 6 | residual-only 1.00 | 48 o_proj layers | 0 | 75.2% | 0% | 54.3% | 0.0156 |
| 7 | mask-only HRCG | 10 experts masked | 10 | **95.0%** | 0% | 54.3% | — |
| 8 | decode-capture mask | 83 decode-scored | 83 | **94.1%** | 0% | 51.4% | — |

## Phase 2B Outcome (Run 9)

| Checkpoint | Refusal | Over-refusal | Reasoning | Val Loss |
|-----------|---------|--------------|-----------|----------|
| Baseline (MLX eval harness) | 89.0% | 0.0% | 54.3% | - |
| Iter 500 | 51.5% | 0.0% | 80.0% | 0.748 |
| Iter 1750 | 28.7% | 0.0% | 97.1% | 0.515 |
| Iter 2500 | **9.9%** | **0.0%** | **97.1%** | 0.654 |

- LoRA targets: `self_attn.o_proj`, `self_attn.v_proj` (last 24 layers), rank 16, scale 2.0, dropout 0.05
- Trainable params: 3.3M / 30.5B (0.011%)
- Data mix: 4520 examples (xLAM + FineTome + Hermes + anti-refusal)

Note: Phase 2A and Phase 2B used different evaluation wrappers. Interpret absolute baseline values within-phase.

## Key Findings

### 1. MoE Expert Surgery Does Not Reduce Chat Refusal

Across four distinct MoE approaches (projected, multi-projected, multi-unprojected, mask-only) with varying strengths, direction counts, energy thresholds, router attenuation, and selection strategies (elbow, SES, HCDG/HRCG), refusal never dropped below ~87-89%. KL divergence confirms the model IS modified (up to 0.052), but the modification does not affect refusal behavior.

### 2. HRCG "Control" Experts Are Actually Compliance Pathways

Run 7's mask-only diagnostic is the most revealing result. Masking the 10 highest-scoring HRCG experts (identified via jailbreak-based decomposition as "refusal enforcement" experts) **increased** refusal to 95.0%. These experts help the model *comply* with requests, not refuse them. Blocking them removes the model's ability to override its refusal instinct.

### 3. Residual o_proj Is Part of the Refusal Circuit (But Has a Ceiling)

Attention o_proj abliteration across all 48 decoder layers reduced refusal monotonically:
- Strength 0.00 → ~100% (unmodified)
- Strength 0.35 → 84.2% (-4.9pp)
- Strength 0.60 → 79.2% (-9.9pp)
- Strength 1.00 → 75.2% (-13.9pp)

Diminishing returns are clear. No capability damage at any strength (reasoning 54.3%, over-refusal 0% throughout). The ceiling at ~70-75% suggests refusal is distributed across multiple circuit components beyond o_proj.

### 4. Prefill/Decode Mismatch Conclusively Confirmed (Run 8)

Run 8 captured expert routing during 3 decode steps across all 201 prompts, then scored and masked 83 decode-targeted experts. Jaccard similarity between prefill top-10 and decode top-10 experts per layer was mean 0.100 (range 0.000-0.300) — 90% of decode-active experts differ from prefill-active experts. ZERO overlap between the top 15 decode-scored and top 15 prefill-scored experts.

Top 15 decode-scored experts: L30 E42, L40 E9, L43 E1, L20 E115, L47 E87, L30 E124, L0 E76, L42 E57, L45 E40, L30 E47, L40 E69, L44 E120, L43 E13, L43 E57, L36 E62.

Despite targeting the correct phase, decode-targeted masking produced the same pattern as prefill-targeted masking: refusal **increased** to 94.1%. This confirms that compliance pathways are distributed across both phases — MoE routing fabric distributes compliance, not refusal.

### 5. Capability Damage: First Observed in Run 8

Across Runs 1-7, reasoning canary stayed at baseline (54.3%) and over-refusal remained at 0%. Run 8 was the first experiment to show capability damage: reasoning dropped to 51.4% (-2.9pp from baseline). This suggests decode-targeted masking of 83 experts disrupts general processing pathways, not just compliance/refusal circuits. Over-refusal remained at 0% across all 8 runs.

## Root Cause Analysis

### Why MoE Abliteration Fails

Six hypotheses were developed and documented in `overnight-shared/claude-moe-vs-residual.md`. The top three, supported by Run 7:

**H1 — Prefill/Decode Mismatch (CONFIRMED by Run 8):** Flay captures expert activations at the last prompt token during prefill (`model.forward`, `src/analysis/activations.rs:110`), but refusal is measured during decode (`model.forward_cached`, `src/eval/generate.rs:34`). These are separate code paths. Run 8 proved this: Jaccard similarity between prefill and decode top experts is 0.10, with zero overlap in the top 15. However, targeting decode experts also fails — both expert sets contain compliance pathways, not refusal pathways.

**H5 — Attention Writes Refusal Before MoE Sees It:** In each transformer layer, attention runs before MoE (`src/model/qwen3_moe.rs:693-707`). The attention `o_proj` writes refusal information into the residual stream; MoE experts then process this already-contaminated input. Abliterating o_proj cuts refusal at the source. Abliterating MoE down_proj tries to cut it downstream, after attention has already committed the signal.

**H3 — Router Routes Around Abliterated Experts:** With top-8 routing from 128 experts, the router can redistribute to unmodified experts. Even mask-only (setting gate logits to -inf) just shifts probability mass to other experts that independently produce the same refusal output. Run 7 confirmed this — masking 10 experts didn't remove refusal, it removed compliance.

### Why Residual o_proj Works (Partially)

Unlike MoE experts (sparse, routed, per-token), attention o_proj is applied to every token at every layer during both prefill and decode. It cannot be routed around. The banded strength schedule (peak at layers 24-39) targets the layers with strongest refusal signal. However, the ~75% ceiling suggests refusal is also encoded in:
- Embedding/unembedding layers (never modified)
- MoE gate_proj/up_proj (not abliterable in current codebase)
- The KV cache itself (attention patterns, not just projections)
- Token-level probabilities in the LM head

## Lessons Learned

1. **SES strict intersection is too conservative for distributed refusal** — selected only 3 experts (Run 1)
2. **Reasoning guardrail must be baseline-relative** — the 80% hardcoded threshold was above the model's 54.3% baseline (fixed in `fa945f0`)
3. **Multi-pass converges immediately** — Run 2 pass 2 found zero new experts; the elbow method is stable
4. **BF16 dtype handling needs explicit casts** — gate weights on Metal are BF16, `to_vec2::<f32>()` fails without F32 cast (fixed in `263c6c0`)
5. **Completions API bypasses refusal entirely** — chat template tokens activate the refusal pathway; raw text completion does not
6. **Run ONE flay instance at a time** — 3 parallel instances OOM on 256GB

## Recommended Next Steps

1. **Keep MoE expert surgery closed for Qwen3-30B-A3B.** Run 9 success removed the need for additional MoE ablation diagnostics on this model.
2. **Run Phase 2C agent-harness validation** on iter-1750 and iter-2500:
   - BFCL tool-call correctness (schema validity, tool selection, argument accuracy)
   - End-to-end coding agent tasks (multi-turn completion + tool sequencing)
   - Custom no-tool and ambiguous-tool cases
3. **Choose checkpoint by harness outcomes**, not refusal alone.
4. **Only revisit additional training if harness regressions appear**, otherwise freeze and package the winning checkpoint.

## File References

- Experiment reports: `runs/run{1-9}-*/`
- Detailed MoE vs residual analysis: `overnight-shared/claude-moe-vs-residual.md`
- Experiment design doc: `docs/plans/2026-03-03-abliteration-experiment-design.md`
- Codex trend analysis: `overnight-shared/codex-pane2-trend-and-next.md`
