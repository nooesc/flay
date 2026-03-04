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
| Experts abliterated | 3 |
| Threshold | 9.8851 (auto (elbow method)) |
| KL divergence | 0.000348 |

## Modified Experts

| Layer | Expert | Strength | Direction Source |
|------:|-------:|---------:|:----------------|
| 43 | 127 | 1.00 | per-expert |
| 41 | 48 | 0.27 | per-expert |
| 42 | 64 | 0.15 | per-expert |

## Usage

This model is a drop-in replacement for the base model. Use it with any framework that supports the original architecture (transformers, vLLM, llama.cpp, etc.).

## Disclaimer

This model has had safety-trained refusal behavior reduced. It may produce outputs that the original model would have refused. Use responsibly and in accordance with applicable laws and regulations.
