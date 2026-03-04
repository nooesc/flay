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
| Experts abliterated | 71 |
| Threshold | 42.8492 (auto (elbow method)) |

## Modified Experts

| Layer | Expert | Strength | Direction Source |
|------:|-------:|---------:|:----------------|
| 47 | 85 | 1.00 | per-expert |
| 40 | 45 | 0.95 | per-expert |
| 46 | 103 | 0.93 | per-expert |
| 45 | 84 | 0.83 | per-expert |
| 43 | 44 | 0.79 | per-expert |
| 47 | 86 | 0.70 | per-expert |
| 47 | 77 | 0.69 | per-expert |
| 43 | 127 | 0.64 | per-expert |
| 44 | 98 | 0.61 | per-expert |
| 46 | 127 | 0.56 | per-expert |
| 44 | 73 | 0.55 | per-expert |
| 40 | 84 | 0.51 | per-expert |
| 43 | 77 | 0.49 | per-expert |
| 45 | 37 | 0.49 | per-expert |
| 41 | 18 | 0.43 | per-expert |
| 42 | 7 | 0.43 | per-expert |
| 24 | 97 | 0.43 | per-expert |
| 45 | 101 | 0.42 | per-expert |
| 23 | 39 | 0.42 | per-expert |
| 32 | 80 | 0.42 | per-expert |
| 43 | 43 | 0.42 | per-expert |
| 42 | 36 | 0.39 | per-expert |
| 46 | 19 | 0.39 | per-expert |
| 39 | 120 | 0.39 | per-expert |
| 44 | 83 | 0.39 | per-expert |
| 46 | 36 | 0.39 | per-expert |
| 45 | 22 | 0.37 | per-expert |
| 46 | 75 | 0.36 | per-expert |
| 39 | 70 | 0.36 | per-expert |
| 45 | 24 | 0.36 | per-expert |
| 46 | 111 | 0.34 | per-expert |
| 36 | 75 | 0.34 | per-expert |
| 46 | 11 | 0.34 | per-expert |
| 47 | 101 | 0.33 | per-expert |
| 39 | 8 | 0.33 | per-expert |
| 44 | 27 | 0.32 | per-expert |
| 44 | 53 | 0.32 | per-expert |
| 38 | 75 | 0.32 | per-expert |
| 42 | 31 | 0.32 | per-expert |
| 43 | 50 | 0.32 | per-expert |
| 43 | 20 | 0.31 | per-expert |
| 22 | 53 | 0.30 | per-expert |
| 47 | 73 | 0.30 | per-expert |
| 47 | 2 | 0.29 | per-expert |
| 26 | 75 | 0.29 | per-expert |
| 40 | 17 | 0.29 | per-expert |
| 36 | 97 | 0.28 | per-expert |
| 45 | 10 | 0.27 | per-expert |
| 47 | 20 | 0.27 | per-expert |
| 20 | 80 | 0.26 | per-expert |
| 41 | 48 | 0.26 | per-expert |
| 39 | 10 | 0.26 | per-expert |
| 27 | 8 | 0.26 | per-expert |
| 24 | 2 | 0.26 | per-expert |
| 37 | 14 | 0.26 | per-expert |
| 36 | 63 | 0.26 | per-expert |
| 45 | 30 | 0.24 | per-expert |
| 41 | 69 | 0.24 | per-expert |
| 31 | 51 | 0.23 | per-expert |
| 46 | 38 | 0.23 | per-expert |
| 47 | 125 | 0.22 | per-expert |
| 44 | 0 | 0.22 | per-expert |
| 23 | 28 | 0.22 | per-expert |
| 19 | 110 | 0.22 | per-expert |
| 46 | 60 | 0.21 | per-expert |
| 47 | 78 | 0.21 | per-expert |
| 32 | 86 | 0.21 | per-expert |
| 20 | 95 | 0.21 | per-expert |
| 32 | 116 | 0.21 | per-expert |
| 44 | 89 | 0.21 | per-expert |
| 41 | 42 | 0.20 | per-expert |
| 19 | - | 0.10 | gate attenuation |
| 20 | - | 0.10 | gate attenuation |
| 22 | - | 0.10 | gate attenuation |
| 23 | - | 0.10 | gate attenuation |
| 24 | - | 0.10 | gate attenuation |
| 26 | - | 0.10 | gate attenuation |
| 27 | - | 0.10 | gate attenuation |
| 31 | - | 0.10 | gate attenuation |
| 32 | - | 0.10 | gate attenuation |
| 36 | - | 0.10 | gate attenuation |
| 37 | - | 0.10 | gate attenuation |
| 38 | - | 0.10 | gate attenuation |
| 39 | - | 0.10 | gate attenuation |
| 40 | - | 0.10 | gate attenuation |
| 41 | - | 0.10 | gate attenuation |
| 42 | - | 0.10 | gate attenuation |
| 43 | - | 0.10 | gate attenuation |
| 44 | - | 0.10 | gate attenuation |
| 45 | - | 0.10 | gate attenuation |
| 46 | - | 0.10 | gate attenuation |
| 47 | - | 0.10 | gate attenuation |

## Usage

This model is a drop-in replacement for the base model. Use it with any framework that supports the original architecture (transformers, vLLM, llama.cpp, etc.).

## Disclaimer

This model has had safety-trained refusal behavior reduced. It may produce outputs that the original model would have refused. Use responsibly and in accordance with applicable laws and regulations.
