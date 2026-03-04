# Flay

Per-expert abliteration for Mixture-of-Experts language models.

Flay identifies which MoE experts participate in refusal behavior and abliterates only those, leaving the rest untouched. In a model with 6,144 total experts, it typically targets fewer than 100. Built from scratch in Rust using [candle](https://github.com/huggingface/candle) with custom per-expert activation capture hooks.

> **Looking for CLI usage & installation?** See [USAGE.md](USAGE.md)

## The Hypothesis

MoE models route tokens through a sparse subset of experts. If refusal behavior is concentrated in specific experts, we should be able to surgically remove it without the collateral damage of uniform abliteration. Flay was built to test this — score every expert by refusal participation, select the guilty ones, and abliterate only those with score-proportional strength.

## Experiment Results — Qwen3-30B-A3B

Eight experiments on Qwen3-30B-A3B-Instruct-2507 (48 MoE layers, 128 experts/layer, top-8 routing) on a Mac Studio M3 Ultra (Metal BF16).

| Run | Approach | Target | Experts | Refusal | KL |
|-----|----------|--------|---------|---------|-----|
| baseline (unmodified chat eval) | — | — | — | ~100% | — |
| 1 | projected + SES | MoE down_proj | 3 | 90.1% | 0.0003 |
| 2 | multi-proj + router atten | MoE down_proj + gate | 71 | 89.1% | 0.0123 |
| 3 | multi (aggressive) | MoE down_proj + gate | 83 | 89.1% | 0.0522 |
| 4 | residual-only 0.35 | 48 o_proj layers | 0 | 84.2% | 0.0056 |
| 5 | residual-only 0.60 | 48 o_proj layers | 0 | 79.2% | 0.0068 |
| 6 | residual-only 1.00 | 48 o_proj layers | 0 | 75.2% | 0.0156 |
| 7 | mask-only HRCG | 10 experts masked | 10 | **95.0%** | — |
| 8 | decode-capture mask | 83 decode-scored | 83 | **94.1%** | — |

### Runs 1–3: MoE surgery doesn't move refusal

Tried projected, multi-projected, and aggressive multi-directional abliteration with router attenuation. Different selection strategies (elbow, SES, HCDG/HRCG decomposition). Different numbers of experts (3 to 83). KL divergence confirms the weights are modified — but refusal stays pinned at ~90%.

### Runs 4–6: Residual o_proj is part of the circuit

Switched to abliterating attention `o_proj` across all 48 decoder layers. This actually moved the needle — refusal dropped monotonically from ~100% to 75% as strength increased. No capability damage at any strength (reasoning canary stable, 0% over-refusal). But diminishing returns suggest a ceiling around 70-75%.

### Run 7: The inversion

The most revealing experiment. Used jailbreak-based HCDG/HRCG decomposition to identify 10 "refusal enforcement" experts. Masked them at inference time. Refusal went *up* to 95%. The experts we thought were enforcing refusal were actually **compliance pathways** — they help the model *override* its refusal instinct. Blocking them makes it refuse more.

### Run 8: Prefill/decode mismatch confirmed

Captured expert routing during decode steps (not just prefill). Jaccard similarity between prefill and decode top experts: 0.10. Zero overlap in the top 15. Despite targeting the correct phase, decode-targeted masking still increased refusal to 94.1%. Compliance pathways are distributed across both phases.

## Conclusion

**MoE expert surgery does not reduce chat refusal in Qwen3-30B-A3B.** Eight experiments across every combination of modes, strengths, selection strategies, and capture phases. The hypothesis — that refusal is concentrated in specific experts — is wrong for this architecture.

## Key Findings

- **High-scoring experts are compliance pathways, not refusal pathways.** Masking them removes the model's ability to comply, not its tendency to refuse.
- **Residual o_proj abliteration works but has a ceiling (~75%).** It's applied to every token at every layer — can't be routed around. But refusal is distributed across more components than just attention output projections.
- **Prefill and decode activate almost entirely different experts** (Jaccard 0.10). Scoring based on prefill activations tells you very little about decode behavior.
- **KL divergence confirms weights are modified but behavior doesn't change.** The router compensates — with top-8 from 128 experts, probability mass redistributes to unmodified experts.

<details>
<summary><b>Root cause analysis</b></summary>

**Why MoE abliteration fails:**

1. **Attention writes refusal before MoE sees it.** In each transformer layer, attention runs before MoE. The `o_proj` writes refusal information into the residual stream; MoE experts process already-contaminated input. Abliterating `down_proj` tries to cut refusal downstream, after attention has already committed the signal.

2. **The router routes around abliterated experts.** With top-8 from 128 experts, the router redistributes to unmodified experts. Even mask-only (gate logits → -inf) just shifts probability mass to other experts that independently produce the same output.

3. **Compliance is distributed, not localized.** Both prefill and decode expert sets contain compliance pathways. There's no clean "refusal cluster" to cut.

</details>

<details>
<summary><b>How It Works</b></summary>

### Phase 1: Activation Capture

Harmful and harmless prompt datasets are run through the model, recording residual stream states at each MoE layer, per-expert activation counts (how often each expert is selected for harmful vs. harmless inputs), and per-expert output means. This produces a detailed picture of which experts the router favors for harmful content and what those experts output.

### Phase 2: Scoring and Selection

Each expert receives a combined refusal score from two signals: **refusal projection** (how aligned the expert's behavior is with the refusal direction) and **routing bias** (how much more frequently the router selects it for harmful content). Experts are ranked by `combined_score = refusal_projection * min(routing_bias, 3.0)` and the guilty set is selected via manual threshold, the elbow method, or stability-based selection.

<details>
<summary>Scoring formulas</summary>

**Refusal projection** — In single-direction mode:

```
expert_diff = mean(expert_output | harmful) - mean(expert_output | harmless)
refusal_projection = |dot(expert_diff, refusal_direction)|
```

In multi-direction mode, a weighted RMS across all SVD directions:

```
refusal_projection = sqrt(sum(w_k * dot(expert_diff, direction_k)^2) / sum(w_k))
```

**Routing bias** — capped at 3.0 to prevent rarely-activated experts from dominating:

```
routing_bias = min(count(activated | harmful) / count(activated | harmless), 3.0)
```

**Threshold selection** — The elbow method walks down sorted scores, finds the largest relative gap (>30%), and splits there. If no clear elbow but meaningful score variance exists, a 15%-of-max-score cutoff is used. If the distribution is truly flat, only the single highest-scoring expert is selected.

</details>

### Phase 3: Variable-Strength Abliteration

Guilty experts are abliterated with score-proportional strength: the highest-scoring expert gets full removal, the lowest gets `strength_min` (default 0.5). In the default MoE-only path, only the `down_proj` weight matrix of each guilty expert is modified. With `--residual`, attention `o_proj` weights across all decoder layers are also abliterated. Embeddings and the LM head are never modified.

<details>
<summary>Orthogonalization math</summary>

Refusal direction removal via orthogonal projection:

```
W' = W - strength * r * (r^T @ W)
```

Variable strength per expert:

```
strength(expert) = strength_min + (1 - strength_min) * (score - min_score) / (max_score - min_score)
```

For multi-directional modes, each direction is orthogonalized sequentially with weight-proportional strength:

```
for each direction (r_k, w_k):
    sub_strength = strength * (w_k / sum(weights))
    W = W - sub_strength * r_k * (r_k^T @ W)
```

</details>

</details>

## Comparison with Existing Tools

| Feature | HERETIC | ErisForge | DECCP | **Flay** |
|:--------|:--------|:----------|:------|:------------|
| Approach | TPE optimization + LoRA | Uniform orthogonalization | Contrastive decoding | **Per-expert selective** |
| Granularity | Per-layer strength | Per-layer strength | Token-level | **Per-expert strength** |
| Expert awareness | No | No | No | **Yes** |
| Experts modified | All | All | N/A (inference-time) | **Only guilty subset** (+ o_proj with `--residual`) |
| Direction type | Global | Global | N/A | **Global + per-expert, single or multi (SVD)** |
| Multi-pass | No | No | No | **Yes (with guardrails)** |
| Router attenuation | No | No | No | **Yes** |
| Optimization | 200 trials (slow) | None | None | **Grid search or Bayesian TPE** |
| Output format | LoRA adapter | Safetensors | N/A | **Safetensors + model card** |

## References

- Arditi, A., Obeso, O., Syed, A., Paleka, D., & Rimsky, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Zou, A., Wang, Z., Kolter, J.Z., & Fredrikson, M. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
- Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)

## License

Apache-2.0

## Why "Flay"?

Existing tools use a sledgehammer — abliterating every expert uniformly. Flay cuts only what needs cutting.
