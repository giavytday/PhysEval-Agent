#!/usr/bin/env bash
#
# PhysEval-Agent master pipeline runner.
#
# Executes the complete workflow sequentially:
#   Step 1: Task synthesis        -> tasks/benchmark_suite.jsonl
#   Step 2: Batch rollouts        -> data/trajectories.jsonl   (physeval.run_rollouts)
#   Step 3: Dataset export        -> data/{prm_steps,dpo_pairs,stats}.json     (physeval.export_dataset)
#   Step 4: Benchmark evaluation  -> eval_results.jsonl         (physeval.eval_benchmark)
#   Step 5: Visual report         -> reports/                   (physeval.generate_report)
#
# Usage:
#   ./run_pipeline.sh --smoke-test                 # hermetic: 3 tasks, mock LLM, no network
#   ./run_pipeline.sh --full                       # 600 tasks against a real model
#   ./run_pipeline.sh --smoke-test --concurrency 3 --model gpt-4o-mini
#
# Flags:
#   --smoke-test      Tiny hermetic run (3 synthesized tasks, deterministic mock LLM).
#   --full            Full-scale run (600 tasks); requires an OpenAI-compatible key.
#   --concurrency N   Parallel rollouts/evaluations (default: 3 smoke / 4 full).
#   --model NAME      Chat model id (default: OPENAI_MODEL env or gpt-4o-mini).
#   --client C        'openai' (default for --full) or 'mock' (default for --smoke-test).
#   --base-url URL    OpenAI-compatible endpoint (e.g. https://openrouter.ai/api/v1).
#   --max-turns N     Agentic repair budget (default: 2 smoke / 4 full).
#   --out-dir PATH    Root directory for all artifacts (default: <repo>/runs/pipeline).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

MODE="smoke"
CLIENT=""
CONCURRENCY=""
MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
MAX_TURNS=""
BASE_URL_ARGS=()
OUT_ROOT="$ROOT/runs/pipeline"

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | grep -E '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)  MODE="smoke"; shift ;;
        --full)        MODE="full";  shift ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --client)      CLIENT="$2"; shift 2 ;;
        --base-url)    BASE_URL="$2"; shift 2 ;;
        --max-turns)   MAX_TURNS="$2"; shift 2 ;;
        --out-dir)     OUT_ROOT="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *) echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
    esac
done

if [[ "$MODE" == "smoke" ]]; then
    SUITE_ARGS=(--per-domain 1)
    SUITE_PATH="$OUT_ROOT/tasks/benchmark_suite.jsonl"
    CLIENT="${CLIENT:-mock}"
    CONCURRENCY="${CONCURRENCY:-3}"
    MAX_TURNS="${MAX_TURNS:-2}"
else
    SUITE_ARGS=(--total 600 --seed 42)
    SUITE_PATH="$ROOT/physeval/tasks/benchmark_suite.jsonl"
    CLIENT="${CLIENT:-openai}"
    CONCURRENCY="${CONCURRENCY:-4}"
    MAX_TURNS="${MAX_TURNS:-4}"
fi

DATA_DIR="$OUT_ROOT/data"
REPORT_DIR="$OUT_ROOT/reports"
TRAJ="$DATA_DIR/trajectories.jsonl"
EVAL_RESULTS="$OUT_ROOT/eval_results.jsonl"

die() { echo "error: $*" >&2; exit 2; }
step() {
    STEP_NO="${STEP_NO:-0}"
    STEP_NO=$((STEP_NO + 1))
    echo ""
    echo "==> [Step $STEP_NO/5] $1"
}
trap 'echo "" ; echo "pipeline FAILED (exit $?) at line $LINENO" >&2' ERR

[[ -x "$(command -v "$PYTHON")" ]] || die "python interpreter not found: $PYTHON"

if [[ -n "${BASE_URL:-}" ]]; then
    BASE_URL_ARGS=(--base-url "$BASE_URL")
else
    BASE_URL_ARGS=()
fi

"$PYTHON" -c "import physeval" 2>/dev/null || die "physeval not importable (PYTHONPATH=$PYTHONPATH)"

if [[ "$CLIENT" == "openai" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        die "--client openai requires OPENAI_API_KEY (or run with --client mock)"
    fi
fi

mkdir -p "$(dirname "$SUITE_PATH")" "$DATA_DIR" "$REPORT_DIR"
rm -f "$TRAJ" "$EVAL_RESULTS" \
      "$DATA_DIR/prm_steps.jsonl" "$DATA_DIR/dpo_pairs.jsonl" "$DATA_DIR/stats.json" \
      "$REPORT_DIR/report.md" "$REPORT_DIR/pass_rates.png" "$REPORT_DIR/drift_reduction.png"

echo "PhysEval pipeline | mode=$MODE client=$CLIENT model=$MODEL concurrency=$CONCURRENCY max_turns=$MAX_TURNS"
echo "artifacts -> $OUT_ROOT"

# ---------------------------------------------------------------- #
step "Task synthesis -> $(basename "$SUITE_PATH")"
"$PYTHON" -m physeval.tasks.synthesizer "${SUITE_ARGS[@]}" --seed "${SEED:-42}" --out "$SUITE_PATH"
[[ -s "$SUITE_PATH" ]] || die "synthesis produced no suite file"

# ---------------------------------------------------------------- #
step "Batch rollouts ($CLIENT client)"
"$PYTHON" -m physeval.run_rollouts \
    --suite "$SUITE_PATH" \
    --client "$CLIENT" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --max-turns "$MAX_TURNS" \
    ${BASE_URL_ARGS[@]+"${BASE_URL_ARGS[@]}"} \
    --output "$TRAJ"
[[ -s "$TRAJ" ]] || die "rollouts produced no trajectories"

# ---------------------------------------------------------------- #
step "Dataset export (PRM + DPO)"
"$PYTHON" -m physeval.export_dataset --trajectories "$TRAJ" --out-dir "$DATA_DIR"
[[ -s "$DATA_DIR/prm_steps.jsonl" ]] || die "PRM export missing"
[[ -f "$DATA_DIR/dpo_pairs.jsonl" ]] || die "DPO export missing"
[[ -s "$DATA_DIR/stats.json" ]] || die "dataset stats missing"

# ---------------------------------------------------------------- #
step "Benchmark evaluation (Pass@1 vs Pass@${MAX_TURNS})"
"$PYTHON" -m physeval.eval_benchmark \
    --suite "$SUITE_PATH" \
    --client "$CLIENT" \
    --model "$MODEL" \
    --max-turns "$MAX_TURNS" \
    --concurrency "$CONCURRENCY" \
    --timeout-s "${TIMEOUT_S:-20}" \
    ${BASE_URL_ARGS[@]+"${BASE_URL_ARGS[@]}"} \
    --output "$EVAL_RESULTS"
[[ -s "$EVAL_RESULTS" ]] || die "evaluation produced no records"

# ---------------------------------------------------------------- #
step "Visual report generation"
"$PYTHON" -m physeval.generate_report \
    --results "$EVAL_RESULTS" \
    --out-dir "$REPORT_DIR" \
    --title "PhysEval pipeline report (${MODE})" \
    --k "$MAX_TURNS"
[[ -s "$REPORT_DIR/report.md" ]] || die "markdown report missing"

echo ""
echo "==================== PIPELINE COMPLETE ===================="
for f in "$SUITE_PATH" "$TRAJ" "$DATA_DIR/prm_steps.jsonl" \
         "$DATA_DIR/dpo_pairs.jsonl" "$DATA_DIR/stats.json" \
         "$EVAL_RESULTS" "$REPORT_DIR/report.md" \
         "$REPORT_DIR/pass_rates.png" "$REPORT_DIR/drift_reduction.png"; do
    if [[ -e "$f" ]]; then
        printf "  %-28s %8s  %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)" "$f"
    else
        printf "  %-28s %8s  %s\n" "$(basename "$f")" "-" "(not generated)"
    fi
done
echo "==========================================================="
