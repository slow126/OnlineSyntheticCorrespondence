#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "This script must be run with bash." >&2
  exit 2
fi

OUTPUT_PREFIX="analysis_comprehensive_runs/hof_motion_v3_density_jointood_full"
RUN_TAG="v1"
RIDGE_ALPHA="10"
TARGET="auc_normalized_observed"
METRIC="jointood_mae"
HOF_ALLOW_METHODS="hof_motion_k5,hof_motion_k10,hof_motion_k20,hof_motion_k40,hof_kl_k5"
JOBS=3
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_paper_tables_sixpack.sh [options]

Options:
  --output-prefix PATH     Base output prefix (default: analysis_comprehensive_runs/hof_motion_v3_density_jointood_full)
  --run-tag TAG            Run tag suffix (default: v1)
  --ridge-alpha VALUE      Ridge alpha used in run naming (default: 10)
  --target NAME            Target filter for paper tables (default: auc_normalized_observed)
  --metric NAME            Primary metric (default: jointood_mae)
  --hof-allow-methods CSV  Method bases for HOF-style comparisons
  --jobs N                 Parallel jobs (default: 3)
  --dry-run                Print commands only
  -h, --help               Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-prefix)
      OUTPUT_PREFIX="$2"; shift 2 ;;
    --run-tag)
      RUN_TAG="$2"; shift 2 ;;
    --ridge-alpha)
      RIDGE_ALPHA="$2"; shift 2 ;;
    --target)
      TARGET="$2"; shift 2 ;;
    --metric)
      METRIC="$2"; shift 2 ;;
    --hof-allow-methods)
      HOF_ALLOW_METHODS="$2"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "--jobs must be a positive integer (got: $JOBS)" >&2
  exit 2
fi

alpha_tag="${RIDGE_ALPHA//./p}"
LOG_DIR="${OUTPUT_PREFIX}_paper_tables_logs_${RUN_TAG}"
mkdir -p "$LOG_DIR"

declare -a ROOTS=(
  "${OUTPUT_PREFIX}_ridge_a${alpha_tag}_base_${RUN_TAG}"
  "${OUTPUT_PREFIX}_ridge_a${alpha_tag}_no_family_${RUN_TAG}"
  "${OUTPUT_PREFIX}_ridge_a${alpha_tag}_no_family_no_density_${RUN_TAG}"
  "${OUTPUT_PREFIX}_ols_base_${RUN_TAG}"
  "${OUTPUT_PREFIX}_ols_no_family_${RUN_TAG}"
  "${OUTPUT_PREFIX}_ols_no_family_no_density_${RUN_TAG}"
)

run_one() {
  local root="$1"
  local model="$2"
  local manifest="${root}/method_summary_manifest.yaml"
  local manifest_aug="${root}/method_summary_manifest.with_hof.yaml"
  local manifest_for_compile="${manifest}"
  local summary="${root}/method_summary.csv"
  local summary_abs="${root}/method_summary_abs.csv"
  local summary_rank="${root}/method_summary_rank.csv"
  local out_dir="${root}/paper_tables_eccv_${model}_${METRIC}"

  if [[ ! -f "$manifest" ]]; then
    echo "SKIP missing manifest: $manifest"
    return 0
  fi

  # Ensure HOF k-sweep outputs are present in the manifest used for compilation.
  python - "$root" "$manifest" "$manifest_aug" <<'PY'
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
manifest_aug = Path(sys.argv[3])

data = yaml.safe_load(manifest.read_text()) or {}
methods = data.get("methods", [])
if not isinstance(methods, list):
    methods = []

existing = {str(m.get("name")) for m in methods if isinstance(m, dict) and m.get("name")}
dj = root / "density_joint"

def note_from_name(name: str) -> str:
    notes = []
    if name.endswith("_pairwise"):
        notes.append("model=pairwise_rank")
    if "_eval_only" in name:
        notes.append("dir=eval_only")
    elif "_train_only" in name:
        notes.append("dir=train_only")
    return ",".join(notes)

for patt in ("leakage_free_hof_motion_k*", "leakage_free_hof_kl_k*"):
    for p in sorted(dj.glob(patt)):
        if not p.is_dir():
            continue
        name = p.name.replace("leakage_free_", "")
        if name in existing:
            continue
        entry = {
            "name": name,
            "path": str(p),
            "family": "motion",
            "symmetry": "asym",
        }
        note = note_from_name(name)
        if note:
            entry["notes"] = note
        methods.append(entry)
        existing.add(name)

data["methods"] = methods
manifest_aug.write_text(yaml.safe_dump(data, sort_keys=False))
PY
  manifest_for_compile="${manifest_aug}"

  local compile_cmd=(
    python scripts/compile_method_summary.py
    --manifest "$manifest_for_compile"
    --output "$summary"
    --output-abs "$summary_abs"
    --output-rank "$summary_rank"
    --output-md "${root}/method_summary.md"
  )
  local table_cmd=(
    python scripts/build_paper_tables_eccv.py
    --summary "$summary"
    --output-dir "$out_dir"
    --target "$TARGET"
    --model "$model"
    --metric "$METRIC"
    --hof-allow-methods "$HOF_ALLOW_METHODS"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN [$root]"
    printf '  %q ' "${compile_cmd[@]}"; echo
    printf '  %q ' "${table_cmd[@]}"; echo
    return 0
  fi

  "${compile_cmd[@]}"
  "${table_cmd[@]}"
}

active=0
failed=0

for root in "${ROOTS[@]}"; do
  model="ols"
  if [[ "$root" == *"_ridge_a"* ]]; then
    model="ridge"
  fi

  log_path="${LOG_DIR}/$(basename "$root").log"
  (
    set -euo pipefail
    echo "[$(date +%F' '%T)] START root=$root model=$model"
    run_one "$root" "$model"
    echo "[$(date +%F' '%T)] DONE  root=$root model=$model"
  ) >"$log_path" 2>&1 &

  active=$((active + 1))
  if (( active >= JOBS )); then
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  fi
done

while (( active > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  active=$((active - 1))
done

echo ""
echo "Logs: $LOG_DIR"
if [[ "$failed" -ne 0 ]]; then
  echo "One or more runs failed. Check logs above."
  exit 1
fi
echo "Paper table generation complete for six-pack roots."
