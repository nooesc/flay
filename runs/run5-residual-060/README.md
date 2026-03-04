---
license: apache-2.0
tags:
  - abliterated
  - uncensored
  - flay
  - moe
base_model: Qwen/Qwen3-30B-A3B-Instruct-2507
base_model_revision: main
pipeline_tag: text-generation
---

# Qwen3-30B-A3B-Instruct-2507 (Flay Abliterated)

This model was created by applying [Flay](https://github.com/nooesc/flay) per-expert abliteration to [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507).

Flay identifies individual MoE experts responsible for refusal behavior and surgically removes the refusal direction from their weight matrices, preserving the model's general capabilities while reducing refusal.

## Abliteration Summary

| Metric | Value |
|:-------|------:|
| Base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Revision | `main` |
| Total experts | 6144 |
| Experts abliterated | 0 |
| o_proj layers abliterated | 48 |
| Threshold | 9.8851 (auto (elbow method)) |
| KL divergence | 0.006831 |

## Modified Experts

| Layer | Expert | Strength | Direction Source |
|------:|-------:|---------:|:----------------|
| 0 | - | 0.15 | o_proj residual |
| 1 | - | 0.15 | o_proj residual |
| 2 | - | 0.15 | o_proj residual |
| 3 | - | 0.15 | o_proj residual |
| 4 | - | 0.15 | o_proj residual |
| 5 | - | 0.15 | o_proj residual |
| 6 | - | 0.15 | o_proj residual |
| 7 | - | 0.15 | o_proj residual |
| 8 | - | 0.27 | o_proj residual |
| 9 | - | 0.27 | o_proj residual |
| 10 | - | 0.27 | o_proj residual |
| 11 | - | 0.27 | o_proj residual |
| 12 | - | 0.27 | o_proj residual |
| 13 | - | 0.27 | o_proj residual |
| 14 | - | 0.27 | o_proj residual |
| 15 | - | 0.27 | o_proj residual |
| 16 | - | 0.42 | o_proj residual |
| 17 | - | 0.42 | o_proj residual |
| 18 | - | 0.42 | o_proj residual |
| 19 | - | 0.42 | o_proj residual |
| 20 | - | 0.42 | o_proj residual |
| 21 | - | 0.42 | o_proj residual |
| 22 | - | 0.42 | o_proj residual |
| 23 | - | 0.42 | o_proj residual |
| 24 | - | 0.60 | o_proj residual |
| 25 | - | 0.60 | o_proj residual |
| 26 | - | 0.60 | o_proj residual |
| 27 | - | 0.60 | o_proj residual |
| 28 | - | 0.60 | o_proj residual |
| 29 | - | 0.60 | o_proj residual |
| 30 | - | 0.60 | o_proj residual |
| 31 | - | 0.60 | o_proj residual |
| 32 | - | 0.60 | o_proj residual |
| 33 | - | 0.60 | o_proj residual |
| 34 | - | 0.60 | o_proj residual |
| 35 | - | 0.60 | o_proj residual |
| 36 | - | 0.60 | o_proj residual |
| 37 | - | 0.60 | o_proj residual |
| 38 | - | 0.60 | o_proj residual |
| 39 | - | 0.60 | o_proj residual |
| 40 | - | 0.51 | o_proj residual |
| 41 | - | 0.51 | o_proj residual |
| 42 | - | 0.51 | o_proj residual |
| 43 | - | 0.51 | o_proj residual |
| 44 | - | 0.51 | o_proj residual |
| 45 | - | 0.51 | o_proj residual |
| 46 | - | 0.51 | o_proj residual |
| 47 | - | 0.51 | o_proj residual |

## Usage

This model is a drop-in replacement for the base model. Use it with any framework that supports the original architecture (transformers, vLLM, llama.cpp, etc.).

## Disclaimer

This model has had safety-trained refusal behavior reduced. It may produce outputs that the original model would have refused. Use responsibly and in accordance with applicable laws and regulations.
