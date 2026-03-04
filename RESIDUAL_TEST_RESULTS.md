# Residual-Stream Abliteration Test Results

## Model: Qwen/Qwen3-30B-A3B-Instruct-2507

Tested on CPU/F32 with `--eval --domain data/cyber --report`

## Comparison: MoE-only vs Residual (strength=1.0)

| Metric | MoE-only | Residual (1.0) |
|--------|----------|----------------|
| Experts abliterated | 74 / 6144 | 74 / 6144 |
| o_proj layers abliterated | 0 | 48 |
| Expert strength range | 0.50..1.00 | 0.15..0.30 |
| KL divergence | 0.057 | 0.030 |
| **Refusal rate (harmful)** | **91%** | **88%** |
| **Reasoning canary** | **1/10** | **1/10** |
| Domain 'cyber' refusal | 40% | 38% |
| Total weight replacements | 74 | 122 |
| Runtime (CPU) | ~33 min | ~33 min |

## Key Findings

### 1. Eval methodology is flawed (reasoning canary)
The reasoning canary checks if the expected answer (e.g., "4" for "2+2?") is in the
**first predicted token's top-5**. Qwen3-Instruct models don't output bare answers as
first tokens — they use preambles like "The answer is..." or `<think>` tags. Both
abliterated and unabliterated models likely score ~1/10 on this test.

**Fix needed**: Generate multiple tokens and check for answer in full response.

### 2. Eval methodology is flawed (refusal detection)
The refusal detector checks if tokens like "I", "Sorry", "As" have >50% combined
probability as the first token. This is a reasonable heuristic for coherent models but
gives false positives when the model is incoherent (garbage starting with "I" ≠ refusal).

**Fix needed**: Generate a few tokens and check for actual refusal phrases from
`REFUSAL_MARKERS`, or validate coherence before classifying.

### 3. Residual provides marginal improvement
- Refusal: 91% → 88% (3 percentage point reduction)
- KL divergence: 0.057 → 0.030 (lower = less distortion)
- Cyber domain: 40% → 38%

The lower KL with residual is surprising — o_proj abliteration is more "surgical"
(removes refusal signal from attention output) while MoE expert abliteration at full
strength (0.5-1.0) modifies the expert function weights more heavily.

### 4. Neither approach effectively uncensors this model
91% refusal rate after MoE-only abliteration means the refusal behavior in
Qwen3-30B-A3B-Instruct-2507 is not primarily localized in MoE expert weights.
This is a newer model (July 2025) with potentially stronger refusal training
across more model components.

## Next Steps

1. **Fix eval system** — Generate tokens instead of checking first-token probabilities.
   This is blocking accurate measurement of abliteration effectiveness.
2. **Test on original Qwen3-30B-A3B** (not -Instruct-2507) which may have weaker refusal training
3. **Try lower residual_strength** (0.3, 0.5) to see if there's a sweet spot
4. **Run the saved model through vllm-mlx** to test with actual generation, bypassing
   the eval system entirely
