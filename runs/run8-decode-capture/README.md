# Run 8: Decode-Time Expert Capture Diagnostic

Diagnostic run testing whether prefill-time expert scoring misses the actual refusal experts, which might only activate during decode (token generation). Captured expert activations across prefill + 3 decode steps for all 201 eval prompts. Found 90% expert mismatch between prefill and decode, and masking decode-targeted experts also failed.

## Configuration

| Parameter | Value |
|:----------|------:|
| Base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Mode | multi-projected, mask-only |
| Decode capture | `--capture-decode 3` |
| Prompts captured | 201 (prefill + 3 decode steps each) |
| Decode-scored experts masked | 83 |

## Prefill vs Decode Expert Overlap

| Metric | Value |
|:-------|------:|
| Jaccard similarity (mean) | 0.100 |
| Jaccard similarity (range) | 0.000 - 0.300 |
| Expert mismatch rate | ~90% |
| Overlap with prefill top 15 | 0 / 15 |

## Top Decode Experts

| Layer | Expert | Decode Score |
|------:|-------:|-------------:|
| 30 | 42 | 189.12 |
| 40 | 9 | 160.25 |
| 43 | 1 | 130.45 |
| 20 | 115 | 130.31 |
| 47 | 87 | 105.20 |

## Eval Results

| Metric | Run 0 MoE baseline | Run 8 | Delta |
|:-------|:--------:|:-----:|------:|
| Refusal rate | 87.1% | 94.1% (95/101) | +7.0pp |
| Over-refusal | 0% | 0% | -- |
| Reasoning | 54.3% | 51.4% (18/35) | -2.9pp |

Reference baseline above is the earlier single-pass MoE run (83 experts), not the unmodified-chat baseline used in Phase 2A summary docs.

## Key Findings

- Decode-time experts are **90% different** from prefill-time experts (Jaccard ~0.1). Prior runs were scoring the wrong experts.
- Even after identifying and masking 83 decode-targeted experts, refusal **increased** by 7pp -- the same paradoxical result as Run 7.
- First observed capability damage: reasoning dropped 54.3% to 51.4%, indicating the 83-expert mask was too aggressive for general capability.
- Refusal behavior does not live in MoE experts at either prefill or decode time. It resides in the attention layers (o_proj), consistent with residual-stream abliteration results from runs 4-6.

## Conclusion

MoE expert surgery line of investigation **permanently closed**. Refusal is encoded in attention (o_proj), not in MoE feed-forward experts. All further work pivoted to LoRA fine-tuning targeting attention layers.
