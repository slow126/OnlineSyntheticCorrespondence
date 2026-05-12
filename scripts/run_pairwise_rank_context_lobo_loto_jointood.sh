#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "This script requires bash. Run it as: bash scripts/run_pairwise_rank_context_lobo_loto_jointood.sh ..." >&2
  exit 1
fi

INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RANKING_GROUP="${RANKING_GROUP:-train_dataset}"
PAIRWISE_GROUP_COLS="${PAIRWISE_GROUP_COLS:-benchmark,model_family_encoder}"
RANKING_CONTEXT_COLS="${RANKING_CONTEXT_COLS:-model_family_encoder}"
K_VALUES="${K_VALUES:-5,10,20,40}"
HOF_SOURCE_CSV="${HOF_SOURCE_CSV:-}"
MMD_FLOW_V2="${MMD_FLOW_V2:-analysis/mmd_v2_flow_joint.csv}"
MMD_DINO_V2="${MMD_DINO_V2:-analysis/mmd_v2_dino.csv}"
MMD_FEATURE_V2="${MMD_FEATURE_V2:-analysis/mmd_v2_feature.csv}"
RIDGE_ALPHA="${RIDGE_ALPHA:-}"
FORCE_RERUN=0
NO_FAMILY_EFFECTS=0
NO_DENSITY_CONTROLS=0
DISABLE_MIXEDLM=1
COLLAPSE_CV_CELLS=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_pairwise_rank_context_lobo_loto_jointood.sh [options]

Runs pairwise-rank leakage-free analyses into one output root with:
- LOBO + LOTO + Joint-OOD summaries
- context-aware pair construction (pairwise_group_cols)
- context-aware rank evaluation summaries (ranking_context_cols)

Options:
  --input-root <path>             Input analysis root (default: analysis)
  --output-root <path>            Output root (default: analysis_comprehensive_runs/pairwise_rank_ctx_<timestamp>)
  --ranking-group <name>          Ranking option group (default: train_dataset)
  --pairwise-group-cols <csv>     Pairwise pair-construction groups
                                  (default: benchmark,model_family_encoder)
  --ranking-context-cols <csv>    Extra rank-eval context groups
                                  (default: model_family_encoder)
  --k-values <csv>                HOF k values (default: 5,10,20,40)
  --hof-source-csv <path>         HOF source CSV (auto-detected by default)
  --mmd-flow-v2 <path>            Flow MMD v2 CSV (default: analysis/mmd_v2_flow_joint.csv)
  --mmd-dino-v2 <path>            DINO MMD v2 CSV (default: analysis/mmd_v2_dino.csv)
  --mmd-feature-v2 <path>         Feature MMD v2 CSV (optional)
  --ridge-alpha <value>           Optional pairwise L2 regularization strength
  --no-family-effects             Disable encoder/model-family effects
  --no-density-controls           Disable density controls
  --mixedlm                       Enable mixedlm prediction/regression passes
  --no-mixedlm                    Disable mixedlm (default)
  --collapse-cv-cells             Collapse to train_dataset x benchmark cells before CV
  --no-collapse-cv-cells          Keep raw CV rows (default)
  --force-rerun                   Re-run existing output dirs
  --dry-run                       Print commands without executing
  -h, --help                      Show this help
USAGE
}

run_cmd() {
  local -a cmd=("$@")
  printf 'RUN:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --ranking-group)
      RANKING_GROUP="$2"
      shift 2
      ;;
    --pairwise-group-cols)
      PAIRWISE_GROUP_COLS="$2"
      shift 2
      ;;
    --ranking-context-cols)
      RANKING_CONTEXT_COLS="$2"
      shift 2
      ;;
    --k-values)
      K_VALUES="$2"
      shift 2
      ;;
    --hof-source-csv)
      HOF_SOURCE_CSV="$2"
      shift 2
      ;;
    --mmd-flow-v2)
      MMD_FLOW_V2="$2"
      shift 2
      ;;
    --mmd-dino-v2)
      MMD_DINO_V2="$2"
      shift 2
      ;;
    --mmd-feature-v2)
      MMD_FEATURE_V2="$2"
      shift 2
      ;;
    --ridge-alpha)
      RIDGE_ALPHA="$2"
      shift 2
      ;;
    --force-rerun)
      FORCE_RERUN=1
      shift 1
      ;;
    --no-family-effects)
      NO_FAMILY_EFFECTS=1
      shift 1
      ;;
    --no-density-controls)
      NO_DENSITY_CONTROLS=1
      shift 1
      ;;
    --mixedlm)
      DISABLE_MIXEDLM=0
      shift 1
      ;;
    --no-mixedlm)
      DISABLE_MIXEDLM=1
      shift 1
      ;;
    --collapse-cv-cells)
      COLLAPSE_CV_CELLS=1
      shift 1
      ;;
    --no-collapse-cv-cells)
      COLLAPSE_CV_CELLS=0
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="analysis_comprehensive_runs/pairwise_rank_ctx_${TS}"
fi

if [[ -z "${HOF_SOURCE_CSV}" ]]; then
  if [[ -f "${INPUT_ROOT}/coverage_v2_hof_full_occ.csv" ]]; then
    HOF_SOURCE_CSV="${INPUT_ROOT}/coverage_v2_hof_full_occ.csv"
  elif [[ -f "${INPUT_ROOT}/coverage_v2_hof_full.csv" ]]; then
    HOF_SOURCE_CSV="${INPUT_ROOT}/coverage_v2_hof_full.csv"
  fi
fi

COMMON_ARGS=(
  --ranking-group "${RANKING_GROUP}"
  --pairwise-group-cols "${PAIRWISE_GROUP_COLS}"
  --ranking-context-cols "${RANKING_CONTEXT_COLS}"
  --linear-model pairwise_rank
  --prediction-model pairwise_rank
  --joint-ood-holdout
  --no-per-encoder
  --no-pairwise-all
)
if [[ -n "${RIDGE_ALPHA}" ]]; then
  COMMON_ARGS+=(--ridge-alpha "${RIDGE_ALPHA}")
fi
if [[ "${NO_FAMILY_EFFECTS}" -eq 1 ]]; then
  COMMON_ARGS+=(--no-family-effects)
fi
if [[ "${NO_DENSITY_CONTROLS}" -eq 1 ]]; then
  COMMON_ARGS+=(--no-density-controls)
fi
if [[ "${FORCE_RERUN}" -eq 1 ]]; then
  COMMON_ARGS+=(--force-rerun)
fi
if [[ "${DISABLE_MIXEDLM}" -eq 1 ]]; then
  COMMON_ARGS+=(--no-mixedlm)
fi
if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
  COMMON_ARGS+=(--collapse-cv-cells)
else
  COMMON_ARGS+=(--no-collapse-cv-cells)
fi
if [[ -n "${HOF_SOURCE_CSV}" ]]; then
  COMMON_ARGS+=(--hof-coverage "${HOF_SOURCE_CSV}")
fi

HOF_SWEEP_ARGS=()
if [[ -n "${HOF_SOURCE_CSV}" ]]; then
  HOF_SWEEP_ARGS+=(--hof-source-csv "${HOF_SOURCE_CSV}")
fi

EXTRA_VARIANT_ARGS=()
if [[ "${FORCE_RERUN}" -eq 1 ]]; then
  EXTRA_VARIANT_ARGS+=(--force-rerun)
fi
if [[ "${NO_FAMILY_EFFECTS}" -eq 1 ]]; then
  EXTRA_VARIANT_ARGS+=(--no-family-effects)
fi
if [[ "${NO_DENSITY_CONTROLS}" -eq 1 ]]; then
  EXTRA_VARIANT_ARGS+=(--no-density-controls)
fi
if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
  EXTRA_VARIANT_ARGS+=(--collapse-cv-cells)
else
  EXTRA_VARIANT_ARGS+=(--no-collapse-cv-cells)
fi

run_cmd \
  bash scripts/run_comprehensive_sweep_latest.sh \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --mmd-flow-v2 "${MMD_FLOW_V2}" \
  --mmd-dino-v2 "${MMD_DINO_V2}" \
  --mmd-feature-v2 "${MMD_FEATURE_V2}" \
  "${COMMON_ARGS[@]}"

run_cmd \
  bash scripts/run_single_flow_single_dino_combos.sh \
  --root "${OUTPUT_ROOT}" \
  --analysis-root "${INPUT_ROOT}" \
  --ranking-group "${RANKING_GROUP}" \
  --pairwise-group-cols "${PAIRWISE_GROUP_COLS}" \
  --ranking-context-cols "${RANKING_CONTEXT_COLS}" \
  --linear-model pairwise_rank \
  --prediction-model pairwise_rank \
  "${EXTRA_VARIANT_ARGS[@]}"

run_cmd \
  bash scripts/run_hof_motion_k_sweep.sh \
  --root "${OUTPUT_ROOT}" \
  --analysis-root "${INPUT_ROOT}" \
  --k-values "${K_VALUES}" \
  "${HOF_SWEEP_ARGS[@]}" \
  --ranking-group "${RANKING_GROUP}" \
  --pairwise-group-cols "${PAIRWISE_GROUP_COLS}" \
  --ranking-context-cols "${RANKING_CONTEXT_COLS}" \
  --linear-model pairwise_rank \
  --prediction-model pairwise_rank \
  "${EXTRA_VARIANT_ARGS[@]}"

echo ""
echo "Done."
echo "Output root: ${OUTPUT_ROOT}"
