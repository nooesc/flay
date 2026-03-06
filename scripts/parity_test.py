"""Parity test: get top-10 logits from mlx-lm for comparison with Rust/candle."""
import sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
TEXT = "Hello, world!"

print(f"Loading model: {MODEL_PATH}")
model, tokenizer = load(MODEL_PATH)

# Tokenize without special tokens to match Rust (encode(text, false))
ids = tokenizer.encode(TEXT, add_special_tokens=False)
print(f"Input: {TEXT!r} ({len(ids)} tokens) ids={ids}")

# Forward pass
input_ids = mx.array([ids])
logits = model(input_ids)  # [1, S, vocab]
mx.eval(logits)

last_logits = logits[0, -1, :]  # [vocab]
mx.eval(last_logits)

logit_np = last_logits.tolist()

# Top-10
indexed = sorted(enumerate(logit_np), key=lambda x: -x[1])
print(f"\nTop-10 next token predictions (last position):")
for rank, (idx, logit) in enumerate(indexed[:10]):
    token = tokenizer.decode([idx])
    print(f"  #{rank+1}: {token!r} (id={idx}) logit={logit:.4f}")

import numpy as np
arr = np.array(logit_np)
print(f"Logit stats: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
