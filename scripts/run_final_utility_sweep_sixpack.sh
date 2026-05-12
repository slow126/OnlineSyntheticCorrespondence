#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "This script must be run with bash." >&2
  exit 2
fi

OUT_DIR="analysis_comprehensive_runs/final_utility_sweep"
K_VALUES="1,2,3,4,5,6,7,8"
MIN_RUNS=3
RUN_ROOTS=""
SELECTION_POLICY="ladder"
UNBOUNDED_TOP_N=10
SELECTION_OBJECTIVE="auto"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_final_utility_sweep_sixpack.sh [options]

Options:
  --run-roots CSV      Comma-separated run roots (optional; defaults to six-pack)
  --output-dir PATH   Output directory
  --k-values CSV      Capacity ladder (default: 1,2,3,4,5,6,7,8)
  --min-runs N        Minimum run coverage for selection (default: 3)
  --selection-policy MODE
                      ladder|unbounded|both (default: ladder)
  --unbounded-top-n N Top-N per lane to export in unbounded mode (default: 10)
  --selection-objective OBJ
                      auto|absolute|ranking (default: auto)
  -h, --help          Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUT_DIR="$2"; shift 2 ;;
    --run-roots)
      RUN_ROOTS="$2"; shift 2 ;;
    --k-values)
      K_VALUES="$2"; shift 2 ;;
    --min-runs)
      MIN_RUNS="$2"; shift 2 ;;
    --selection-policy)
      SELECTION_POLICY="$2"; shift 2 ;;
    --unbounded-top-n)
      UNBOUNDED_TOP_N="$2"; shift 2 ;;
    --selection-objective)
      SELECTION_OBJECTIVE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

CMD=(
  python scripts/build_final_utility_sweep.py
  --output-dir "$OUT_DIR"
  --k-values "$K_VALUES"
  --min-runs "$MIN_RUNS"
  --selection-policy "$SELECTION_POLICY"
  --unbounded-top-n "$UNBOUNDED_TOP_N"
  --selection-objective "$SELECTION_OBJECTIVE"
)
if [[ -n "$RUN_ROOTS" ]]; then
  CMD+=(--run-roots "$RUN_ROOTS")
fi

"${CMD[@]}"

echo "Done: $OUT_DIR"
