#!/bin/bash
# Ablation matrix: systematic comparison of abliteration configurations.
# Runs configs sequentially with --no-save to avoid OOM during model writing.
# Each config takes ~20min with KV-cache.

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
FLAY="./target/release/flay"
DOMAIN="data/cyber"
RESULTS_FILE="matrix-results.md"
COMMON_ARGS="--eval --domain $DOMAIN --no-save"

run_config() {
    local name="$1"
    local outdir="$2"
    shift 2
    local args="$@"

    local logfile="${outdir}.log"

    echo "[$(date '+%H:%M:%S')] Starting: $name"
    if $FLAY $MODEL $COMMON_ARGS -o "$outdir" $args > "$logfile" 2>&1; then
        echo "[$(date '+%H:%M:%S')] Completed: $name (exit 0)"
    else
        local rc=$?
        echo "[$(date '+%H:%M:%S')] FAILED: $name (exit $rc)"
    fi
}

parse_results() {
    local name="$1"
    local logfile="$2"

    if [ ! -f "$logfile" ]; then
        echo "| $name | MISSING | - | - | - |"
        return
    fi

    local refusal=$(grep -oP 'Refusal rate: \d+ / \d+ \(\K[0-9.]+' "$logfile" 2>/dev/null | head -1)
    local reasoning=$(grep -oP 'Reasoning canary: \K\d+/\d+' "$logfile" 2>/dev/null | head -1)
    local kl=$(grep -oP 'KL divergence:\s+\K[0-9.]+' "$logfile" 2>/dev/null | head -1)
    local cyber=$(grep -oP "Domain 'cyber': \d+ / \d+ refused \(\K[0-9.]+" "$logfile" 2>/dev/null | head -1)

    echo "| $name | ${refusal:-?}% | ${reasoning:-?} | ${kl:-?} | ${cyber:-?}% |"
}

echo "Starting ablation matrix at $(date)"
echo ""

# Run all configs sequentially (--no-save avoids OOM)
run_config "MoE-only" "matrix-moe-only"
run_config "Residual-only 0.3" "matrix-res-0.3" \
    --residual --residual-strength 0.3 --moe-strength 0.0
run_config "Residual-only 0.6" "matrix-res-0.6" \
    --residual --residual-strength 0.6 --moe-strength 0.0
run_config "Residual-only 1.0" "matrix-res-1.0" \
    --residual --residual-strength 1.0 --moe-strength 0.0
run_config "Hybrid 0.6/0.3" "matrix-hybrid-0.6-0.3" \
    --residual --residual-strength 0.6 --moe-strength 0.3
run_config "Hybrid 1.0/0.3" "matrix-hybrid-1.0-0.3" \
    --residual --residual-strength 1.0 --moe-strength 0.3

echo ""
echo "================================================================"
echo "  MATRIX COMPLETE at $(date)"
echo "================================================================"
echo ""

# Build results table
cat > "$RESULTS_FILE" <<'HEADER'
# Ablation Matrix Results

| Config | Refusal % | Reasoning | KL | Cyber Domain % |
|--------|-----------|-----------|-----|----------------|
HEADER

parse_results "MoE-only" "matrix-moe-only.log" >> "$RESULTS_FILE"
parse_results "Residual-only 0.3" "matrix-res-0.3.log" >> "$RESULTS_FILE"
parse_results "Residual-only 0.6" "matrix-res-0.6.log" >> "$RESULTS_FILE"
parse_results "Residual-only 1.0" "matrix-res-1.0.log" >> "$RESULTS_FILE"
parse_results "Hybrid 0.6/0.3" "matrix-hybrid-0.6-0.3.log" >> "$RESULTS_FILE"
parse_results "Hybrid 1.0/0.3" "matrix-hybrid-1.0-0.3.log" >> "$RESULTS_FILE"

cat "$RESULTS_FILE"
