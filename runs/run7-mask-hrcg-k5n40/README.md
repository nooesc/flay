# Run 7: HRCG Mask-Only Diagnostic

Diagnostic run testing the HRCG (Harmfulness Refusal Control Graph) decomposition hypothesis: that MoE experts can be separated into "detection" experts (HCDG) that recognize harmful content and "control" experts (HRCG) that execute the refusal response. Masking the control experts should reduce refusal while preserving detection. This run disproved that hypothesis.

## Configuration

| Parameter | Value |
|:----------|------:|
| Base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Mode | multi-projected |
| SES scoring | K=5, top_n=40 |
| Intervention | mask-only (no weight changes) |
| HCDG detection experts preserved | 3 |
| HRCG control experts masked | 10 |
| Layers affected | 23, 36, 42-46 |

## Masked HRCG Experts

| Layer | Expert | SES Score |
|------:|-------:|----------:|
| 45 | 37 | 21.97 |
| 43 | 43 | 19.43 |
| 23 | 39 | 17.96 |
| 42 | 36 | 16.27 |
| 46 | 19 | 16.26 |
| 46 | 11 | 14.50 |
| 46 | 75 | 14.43 |
| 46 | 111 | 13.60 |
| 36 | 75 | 13.58 |
| 44 | 53 | 12.29 |

## Eval Results

| Metric | Run 0 MoE baseline | Run 7 | Delta |
|:-------|:--------:|:-----:|------:|
| Refusal rate | 87.1% | 95.0% (96/101) | +7.9pp |
| Over-refusal | 0% | 0% | -- |
| Reasoning | 54.3% | 54.3% (19/35) | -- |

Reference baseline above is the earlier single-pass MoE run (83 experts), not the unmodified-chat baseline used in Phase 2A summary docs.

## Key Findings

- Masking the top-scoring "refusal control" experts **increased** refusal by ~8pp instead of reducing it.
- The HRCG-labeled experts are actually **compliance pathways** -- removing them makes the model *more* likely to refuse, not less.
- The HCDG/HRCG decomposition does not hold for this architecture. Expert scoring based on refusal-correlated activation does not distinguish cause from effect.

## Conclusion

MoE expert surgery cannot reliably target refusal behavior. This result, combined with runs 1-6 showing that residual-stream (attention o_proj) abliteration is where refusal signal lives, led to closing the MoE surgery line of investigation and pivoting to residual+eval redesign and fine-tuning.
