#!/usr/bin/env python3
"""Compare flay and heretic results on the same model.

Usage:
    python scripts/compare_heretic.py \
        --flay-report output/flay-report.json \
        --heretic-checkpoint checkpoints/Model.jsonl \
        --output comparison.md
"""

import argparse
import json
import sys


def load_flay_report(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: flay report not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in flay report: {e}", file=sys.stderr)
        sys.exit(1)


def load_heretic_best(path: str) -> dict:
    """Parse Heretic's Optuna JSONL and find the Pareto-best trial."""
    params_map = {}
    results = {}

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            op = rec.get("op_code")
            tid = rec.get("trial_id")

            if op == 5:  # Parameter record
                dist_raw = rec.get("distribution", "{}")
                dist = json.loads(dist_raw) if isinstance(dist_raw, str) else dist_raw
                pname = rec["param_name"]
                pval = rec["param_value_internal"]
                choices = dist.get("attributes", {}).get("choices")
                if choices:
                    pval = choices[int(pval)]
                params_map.setdefault(tid, {})[pname] = pval

            if op == 6 and rec.get("state") == 1:  # Completed trial
                vals = rec.get("values", [])
                if len(vals) == 2:
                    results[tid] = {"kl": vals[0], "refusal_rate": vals[1]}

    if not results:
        print("Error: no completed trials found in Heretic JSONL", file=sys.stderr)
        sys.exit(1)

    # Find best: lowest KL with refusal_rate < 0.5
    candidates = [
        (tid, r) for tid, r in results.items() if r["refusal_rate"] < 0.5
    ]
    if not candidates:
        candidates = sorted(results.items(), key=lambda x: x[1]["refusal_rate"])

    best_tid, best_result = min(candidates, key=lambda x: x[1]["kl"])
    best_result["params"] = params_map.get(best_tid, {})
    best_result["trial_id"] = best_tid
    best_result["total_trials"] = len(results)
    return best_result


def generate_comparison(flay: dict, heretic: dict) -> str:
    flay_kl = flay.get("kl_divergence")
    if flay_kl is None:
        flay_kl = flay.get("eval", {}).get("kl_divergence", "N/A")
    flay_refusal = flay.get("eval", {}).get("refusal_rate", {}).get("rate", "N/A")
    flay_experts = flay.get("abliterated_experts", "N/A")
    flay_total = flay.get("total_experts", "N/A")

    h_kl = heretic["kl"]
    h_refusal = heretic["refusal_rate"]
    h_trials = heretic["total_trials"]

    def fmt(val, spec):
        try:
            return format(float(val), spec)
        except (TypeError, ValueError):
            return "N/A"

    lines = [
        "# Flay vs Heretic Comparison",
        "",
        f"| Metric | Flay | Heretic |",
        f"|---|---|---|",
        f"| KL Divergence | {fmt(flay_kl, '.4f')} | {fmt(h_kl, '.4f')} |",
        f"| Refusal Rate | {fmt(flay_refusal, '.1%')} | {fmt(h_refusal, '.1%')} |",
        f"| Experts Modified | {flay_experts}/{flay_total} | all |",
        f"| Optimization Trials | N/A | {h_trials} |",
        f"| Method | Per-expert MoE | Uniform layer-range |",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare flay vs heretic results")
    parser.add_argument("--flay-report", required=True, help="Path to flay JSON report")
    parser.add_argument("--heretic-checkpoint", required=True, help="Path to heretic JSONL")
    parser.add_argument("--output", default="comparison.md", help="Output markdown file")
    args = parser.parse_args()

    flay = load_flay_report(args.flay_report)
    heretic = load_heretic_best(args.heretic_checkpoint)
    comparison = generate_comparison(flay, heretic)

    with open(args.output, "w") as f:
        f.write(comparison)

    print(comparison)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
