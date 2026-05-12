#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "This script requires bash. Run it as: bash scripts/run_ridge_weighted_ablation_triplet.sh ..." >&2
  exit 1
fi

INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-analysis_comprehensive_runs/ridge_abs_weighted}"
RUN_TAG="${RUN_TAG:-v1}"
RIDGE_ALPHA="${RIDGE_ALPHA:-10}"

RANKING_GROUP="${RANKING_GROUP:-train_dataset}"
PAIRWISE_GROUP_COLS="${PAIRWISE_GROUP_COLS:-benchmark,model_family_encoder}"
RANKING_CONTEXT_COLS="${RANKING_CONTEXT_COLS:-model_family_encoder}"

FIT_SAMPLE_WEIGHTING="${FIT_SAMPLE_WEIGHTING:-inverse_task}"
FIT_BALANCE_REAL_SYNTH=1
OVERALL_AGGREGATION="${OVERALL_AGGREGATION:-macro_fold}"
CV_RESIDUALIZE_TARGET_BY_CONTEXT=0
CV_RESIDUAL_CONTEXT_COLS=""
CV_RESIDUAL_EVAL_SPACE="residual"
CV_RESIDUAL_TARGET_TRANSFORM="residual"
CV_RESIDUAL_TARGET_STD_EPS="1e-9"
CV_FEWSHOT_CONTEXT_CALIBRATION=0
CV_FEWSHOT_CONTEXT_CALIBRATION_COLS=""
CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS="1e-9"
CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE="2"
CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=1
CV_FEWSHOT_CONTEXT_CALIBRATION_K="0"
CV_FEWSHOT_CONTEXT_CALIBRATION_SEED="0"
CV_REPEAT_AGGREGATION="none"
LOTO_SINGLE_PREDICTOR_BASELINES=1
JOINTOOD_SINGLE_PREDICTOR_BASELINES=1

MAX_PARALLEL="${MAX_PARALLEL:-3}"
PER_JOB_THREADS="${PER_JOB_THREADS:-4}"

COLLAPSE_CV_CELLS=0
ALLOW_MIXED_CONTEXT_COLLAPSE=0
JOINT_OOD_HOLDOUT=1
PAIRWISE_ALL=0
NO_MIXEDLM=1
PER_ENCODER=0
WITH_BASE=0
MODEL_FAMILY_INTERACTIONS=""
SPAIR_INDICATOR_INTERACTIONS=""
FORCE_RERUN=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_ridge_weighted_ablation_triplet.sh [options]

Runs two ridge ablations in parallel by default (density comparison):
  1) no_family
  2) no_family_no_density

Optional:
  - add `--with-base` to include `base` as a third run.

Each job calls scripts/run_comprehensive_sweep_latest.sh with ridge +
absolute target + weighting controls.

Options:
  --input-root <path>             Input analysis root (default: analysis)
  --output-prefix <path>          Output root prefix
                                  (default: analysis_comprehensive_runs/ridge_abs_weighted)
  --run-tag <tag>                 Suffix for output dirs (default: v1)
  --ridge-alpha <value>           Ridge alpha (default: 10)
  --ranking-group <name>          Ranking group (default: train_dataset)
  --pairwise-group-cols <csv>     Pairwise grouping cols for pairwise internals/rank metrics
                                  (default: benchmark,model_family_encoder)
  --ranking-context-cols <csv>    Ranking context grouping cols (default: model_family_encoder)
  --fit-sample-weighting <mode>   none|inverse_benchmark|inverse_train_dataset|inverse_task
                                  (default: inverse_task)
  --fit-balance-real-synth        Balance real vs synthetic fit weight (default: on)
  --no-fit-balance-real-synth     Disable real vs synthetic balancing
  --overall-aggregation <mode>    micro|macro_fold (default: macro_fold)
  --cv-residualize-target-by-context
                                  Residualize target by fold-safe context means
  --no-cv-residualize-target-by-context
                                  Disable fold-safe context residualization (default)
  --cv-residual-context-cols <csv>
                                  Context cols for residualization (default: none)
  --cv-residual-eval-space <mode> absolute|residual (default: residual)
  --cv-residual-target-transform <mode>
                                  residual|zscore (default: residual)
  --cv-residual-target-std-eps <v>
                                  Context std epsilon floor for zscore (default: 1e-9)
  --cv-fewshot-context-calibration
                                  Enable leaky per-fold context calibration (mean+std)
  --no-cv-fewshot-context-calibration
                                  Disable leaky context calibration (default)
  --cv-fewshot-context-calibration-cols <csv>
                                  Calibration context cols (default: none -> builder default)
  --cv-fewshot-context-calibration-std-eps <v>
                                  Calibration std epsilon floor (default: 1e-9)
  --cv-fewshot-context-calibration-min-group-size <n>
                                  Minimum rows per context group before backoff (default: 2)
  --cv-fewshot-context-calibration-backoff
                                  Enable hierarchical context backoff (default: on)
  --no-cv-fewshot-context-calibration-backoff
                                  Disable hierarchical context backoff
  --cv-fewshot-context-calibration-k <n>
                                  True K-shot calibration rows per context group (default: 0)
  --cv-fewshot-context-calibration-seed <n>
                                  Random seed for K-shot row sampling (default: 0)
  --cv-repeat-aggregation <mode>  none|mean|median (default: none)
  --loto-single-predictor-baselines
                                  Enable LOTO single-predictor baseline block (default)
  --no-loto-single-predictor-baselines
                                  Disable LOTO single-predictor baseline block
  --jointood-single-predictor-baselines
                                  Enable Joint-OOD single-predictor baseline block (default)
  --no-jointood-single-predictor-baselines
                                  Disable Joint-OOD single-predictor baseline block
  --max-parallel <n>              Max concurrent jobs (default: 3)
  --per-job-threads <n>           OMP/MKL/OpenBLAS threads per job (default: 4)
  --collapse-cv-cells             Enable CV cell collapse (default: off)
  --no-collapse-cv-cells          Disable CV cell collapse
  --allow-mixed-context-collapse  Allow collapse even when ranking-context cols are set
  --joint-ood-holdout             Enable joint-OOD (default: on)
  --no-joint-ood-holdout          Disable joint-OOD
  --pairwise-all                  Also run paired pairwise_rank companions
  --no-pairwise-all               Disable paired pairwise companions (default)
  --per-encoder                   Enable per-encoder outputs (default: disabled)
  --no-per-encoder                Disable per-encoder outputs (default)
  --model-family-interactions     Enable model-family interaction terms
  --no-model-family-interactions  Disable model-family interaction terms
  --spair-indicator-interactions  Enable spair indicator interactions
  --no-spair-indicator-interactions
                                  Disable spair indicator interactions
  --with-base                     Include base variant as a third run
  --density-compare-only          Run only no_family + no_family_no_density (default)
  --mixedlm                       Enable mixedlm
  --no-mixedlm                    Disable mixedlm (default)
  --force-rerun                   Re-run existing outputs
  --dry-run                       Print commands only
  -h, --help                      Show this help
USAGE
}

trim_ws() {
  local v="$1"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  printf '%s' "${v}"
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
    --input-root) INPUT_ROOT="$2"; shift 2 ;;
    --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --ridge-alpha) RIDGE_ALPHA="$2"; shift 2 ;;
    --ranking-group) RANKING_GROUP="$2"; shift 2 ;;
    --pairwise-group-cols) PAIRWISE_GROUP_COLS="$2"; shift 2 ;;
    --ranking-context-cols) RANKING_CONTEXT_COLS="$2"; shift 2 ;;
    --fit-sample-weighting) FIT_SAMPLE_WEIGHTING="$2"; shift 2 ;;
    --fit-balance-real-synth) FIT_BALANCE_REAL_SYNTH=1; shift 1 ;;
    --no-fit-balance-real-synth) FIT_BALANCE_REAL_SYNTH=0; shift 1 ;;
    --overall-aggregation) OVERALL_AGGREGATION="$2"; shift 2 ;;
    --cv-residualize-target-by-context) CV_RESIDUALIZE_TARGET_BY_CONTEXT=1; shift 1 ;;
    --no-cv-residualize-target-by-context) CV_RESIDUALIZE_TARGET_BY_CONTEXT=0; shift 1 ;;
    --cv-residual-context-cols) CV_RESIDUAL_CONTEXT_COLS="$2"; shift 2 ;;
    --cv-residual-eval-space) CV_RESIDUAL_EVAL_SPACE="$2"; shift 2 ;;
    --cv-residual-target-transform) CV_RESIDUAL_TARGET_TRANSFORM="$2"; shift 2 ;;
    --cv-residual-target-std-eps) CV_RESIDUAL_TARGET_STD_EPS="$2"; shift 2 ;;
    --cv-fewshot-context-calibration) CV_FEWSHOT_CONTEXT_CALIBRATION=1; shift 1 ;;
    --no-cv-fewshot-context-calibration) CV_FEWSHOT_CONTEXT_CALIBRATION=0; shift 1 ;;
    --cv-fewshot-context-calibration-cols) CV_FEWSHOT_CONTEXT_CALIBRATION_COLS="$2"; shift 2 ;;
    --cv-fewshot-context-calibration-std-eps) CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS="$2"; shift 2 ;;
    --cv-fewshot-context-calibration-min-group-size) CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE="$2"; shift 2 ;;
    --cv-fewshot-context-calibration-backoff) CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=1; shift 1 ;;
    --no-cv-fewshot-context-calibration-backoff) CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=0; shift 1 ;;
    --cv-fewshot-context-calibration-k) CV_FEWSHOT_CONTEXT_CALIBRATION_K="$2"; shift 2 ;;
    --cv-fewshot-context-calibration-seed) CV_FEWSHOT_CONTEXT_CALIBRATION_SEED="$2"; shift 2 ;;
    --cv-repeat-aggregation) CV_REPEAT_AGGREGATION="$2"; shift 2 ;;
    --loto-single-predictor-baselines) LOTO_SINGLE_PREDICTOR_BASELINES=1; shift 1 ;;
    --no-loto-single-predictor-baselines) LOTO_SINGLE_PREDICTOR_BASELINES=0; shift 1 ;;
    --jointood-single-predictor-baselines) JOINTOOD_SINGLE_PREDICTOR_BASELINES=1; shift 1 ;;
    --no-jointood-single-predictor-baselines) JOINTOOD_SINGLE_PREDICTOR_BASELINES=0; shift 1 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --per-job-threads) PER_JOB_THREADS="$2"; shift 2 ;;
    --collapse-cv-cells) COLLAPSE_CV_CELLS=1; shift 1 ;;
    --no-collapse-cv-cells) COLLAPSE_CV_CELLS=0; shift 1 ;;
    --allow-mixed-context-collapse) ALLOW_MIXED_CONTEXT_COLLAPSE=1; shift 1 ;;
    --joint-ood-holdout) JOINT_OOD_HOLDOUT=1; shift 1 ;;
    --no-joint-ood-holdout) JOINT_OOD_HOLDOUT=0; shift 1 ;;
    --pairwise-all) PAIRWISE_ALL=1; shift 1 ;;
    --no-pairwise-all) PAIRWISE_ALL=0; shift 1 ;;
    --per-encoder) PER_ENCODER=1; shift 1 ;;
    --no-per-encoder) PER_ENCODER=0; shift 1 ;;
    --model-family-interactions) MODEL_FAMILY_INTERACTIONS="on"; shift 1 ;;
    --no-model-family-interactions) MODEL_FAMILY_INTERACTIONS="off"; shift 1 ;;
    --spair-indicator-interactions) SPAIR_INDICATOR_INTERACTIONS="on"; shift 1 ;;
    --no-spair-indicator-interactions) SPAIR_INDICATOR_INTERACTIONS="off"; shift 1 ;;
    --with-base) WITH_BASE=1; shift 1 ;;
    --density-compare-only) WITH_BASE=0; shift 1 ;;
    --mixedlm) NO_MIXEDLM=0; shift 1 ;;
    --no-mixedlm) NO_MIXEDLM=1; shift 1 ;;
    --force-rerun) FORCE_RERUN=1; shift 1 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  echo "--max-parallel must be >= 1 (got: ${MAX_PARALLEL})" >&2
  exit 2
fi
if ! [[ "${PER_JOB_THREADS}" =~ ^[0-9]+$ ]] || [[ "${PER_JOB_THREADS}" -lt 1 ]]; then
  echo "--per-job-threads must be >= 1 (got: ${PER_JOB_THREADS})" >&2
  exit 2
fi

case "${FIT_SAMPLE_WEIGHTING}" in
  none|inverse_benchmark|inverse_train_dataset|inverse_task)
    ;;
  *)
    echo "Invalid --fit-sample-weighting: ${FIT_SAMPLE_WEIGHTING}" >&2
    exit 2
    ;;
esac

if [[ "${COLLAPSE_CV_CELLS}" -eq 1 && "${ALLOW_MIXED_CONTEXT_COLLAPSE}" -ne 1 ]]; then
  if [[ -n "$(trim_ws "${RANKING_CONTEXT_COLS}")" ]]; then
    echo "Refusing --collapse-cv-cells with non-empty --ranking-context-cols (${RANKING_CONTEXT_COLS})." >&2
    echo "This can mix model contexts and contaminate LOBO/LOTO/Joint ranking metrics." >&2
    echo "Use --no-collapse-cv-cells (recommended), or add --allow-mixed-context-collapse to override." >&2
    exit 2
  fi
fi

case "${OVERALL_AGGREGATION}" in
  micro|macro_fold)
    ;;
  *)
    echo "Invalid --overall-aggregation: ${OVERALL_AGGREGATION}" >&2
    exit 2
    ;;
esac
case "${CV_RESIDUAL_EVAL_SPACE}" in
  absolute|residual)
    ;;
  *)
    echo "Invalid --cv-residual-eval-space: ${CV_RESIDUAL_EVAL_SPACE}" >&2
    exit 2
    ;;
esac
case "${CV_RESIDUAL_TARGET_TRANSFORM}" in
  residual|zscore)
    ;;
  *)
    echo "Invalid --cv-residual-target-transform: ${CV_RESIDUAL_TARGET_TRANSFORM}" >&2
    exit 2
    ;;
esac
case "${CV_REPEAT_AGGREGATION}" in
  none|mean|median)
    ;;
  *)
    echo "Invalid --cv-repeat-aggregation: ${CV_REPEAT_AGGREGATION}" >&2
    exit 2
    ;;
esac

# Cap thread-heavy libraries per job so parallel jobs don't oversubscribe.
export OMP_NUM_THREADS="${PER_JOB_THREADS}"
export MKL_NUM_THREADS="${PER_JOB_THREADS}"
export OPENBLAS_NUM_THREADS="${PER_JOB_THREADS}"
export NUMEXPR_NUM_THREADS="${PER_JOB_THREADS}"

alpha_tag="${RIDGE_ALPHA//./p}"
LOG_DIR="${OUTPUT_PREFIX}_logs_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

declare -a VARIANTS=()
if [[ "${WITH_BASE}" -eq 1 ]]; then
  VARIANTS=("base" "no_family" "no_family_no_density")
else
  VARIANTS=("no_family" "no_family_no_density")
fi
declare -A PID_TO_LABEL=()
declare -A PID_TO_LOG=()
PIDS=()
LAUNCHED=0
SUCCEEDED=0
FAILED=0

build_args_for_variant() {
  local variant="$1"
  local -n out_ref="$2"
  out_ref=(
    bash scripts/run_comprehensive_sweep_latest.sh
    --input-root "${INPUT_ROOT}"
    --output-root "${OUTPUT_PREFIX}_ridge_a${alpha_tag}_${variant}_${RUN_TAG}"
    --linear-model ridge
    --prediction-model ridge
    --ridge-alpha "${RIDGE_ALPHA}"
    --ranking-group "${RANKING_GROUP}"
    --fit-sample-weighting "${FIT_SAMPLE_WEIGHTING}"
    --overall-aggregation "${OVERALL_AGGREGATION}"
  )

  if [[ -n "$(trim_ws "${PAIRWISE_GROUP_COLS}")" ]]; then
    out_ref+=(--pairwise-group-cols "${PAIRWISE_GROUP_COLS}")
  fi
  if [[ -n "$(trim_ws "${RANKING_CONTEXT_COLS}")" ]]; then
    out_ref+=(--ranking-context-cols "${RANKING_CONTEXT_COLS}")
  fi

  if [[ "${FIT_BALANCE_REAL_SYNTH}" -eq 1 ]]; then
    out_ref+=(--fit-balance-real-synth)
  else
    out_ref+=(--no-fit-balance-real-synth)
  fi
  if [[ "${CV_RESIDUALIZE_TARGET_BY_CONTEXT}" -eq 1 ]]; then
    out_ref+=(--cv-residualize-target-by-context)
  fi
  if [[ -n "$(trim_ws "${CV_RESIDUAL_CONTEXT_COLS}")" ]]; then
    out_ref+=(--cv-residual-context-cols "${CV_RESIDUAL_CONTEXT_COLS}")
  fi
  out_ref+=(--cv-residual-eval-space "${CV_RESIDUAL_EVAL_SPACE}")
  out_ref+=(--cv-residual-target-transform "${CV_RESIDUAL_TARGET_TRANSFORM}")
  out_ref+=(--cv-residual-target-std-eps "${CV_RESIDUAL_TARGET_STD_EPS}")
  if [[ "${CV_FEWSHOT_CONTEXT_CALIBRATION}" -eq 1 ]]; then
    out_ref+=(--cv-fewshot-context-calibration)
  fi
  if [[ -n "$(trim_ws "${CV_FEWSHOT_CONTEXT_CALIBRATION_COLS}")" ]]; then
    out_ref+=(--cv-fewshot-context-calibration-cols "${CV_FEWSHOT_CONTEXT_CALIBRATION_COLS}")
  fi
  out_ref+=(--cv-fewshot-context-calibration-std-eps "${CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS}")
  out_ref+=(--cv-fewshot-context-calibration-min-group-size "${CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE}")
  if [[ "${CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF}" -eq 1 ]]; then
    out_ref+=(--cv-fewshot-context-calibration-backoff)
  else
    out_ref+=(--no-cv-fewshot-context-calibration-backoff)
  fi
  out_ref+=(--cv-fewshot-context-calibration-k "${CV_FEWSHOT_CONTEXT_CALIBRATION_K}")
  out_ref+=(--cv-fewshot-context-calibration-seed "${CV_FEWSHOT_CONTEXT_CALIBRATION_SEED}")
  out_ref+=(--cv-repeat-aggregation "${CV_REPEAT_AGGREGATION}")
  if [[ "${LOTO_SINGLE_PREDICTOR_BASELINES}" -eq 1 ]]; then
    out_ref+=(--loto-single-predictor-baselines)
  else
    out_ref+=(--no-loto-single-predictor-baselines)
  fi
  if [[ "${JOINTOOD_SINGLE_PREDICTOR_BASELINES}" -eq 1 ]]; then
    out_ref+=(--jointood-single-predictor-baselines)
  else
    out_ref+=(--no-jointood-single-predictor-baselines)
  fi

  if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
    out_ref+=(--collapse-cv-cells)
  else
    out_ref+=(--no-collapse-cv-cells)
  fi

  if [[ "${JOINT_OOD_HOLDOUT}" -eq 1 ]]; then
    out_ref+=(--joint-ood-holdout)
  else
    out_ref+=(--no-joint-ood-holdout)
  fi

  if [[ "${PAIRWISE_ALL}" -eq 1 ]]; then
    out_ref+=(--pairwise-all)
  else
    out_ref+=(--no-pairwise-all)
  fi
  if [[ "${PER_ENCODER}" -eq 1 ]]; then
    out_ref+=(--per-encoder)
  else
    out_ref+=(--no-per-encoder)
  fi

  if [[ "${NO_MIXEDLM}" -eq 1 ]]; then
    out_ref+=(--no-mixedlm)
  fi
  if [[ "${MODEL_FAMILY_INTERACTIONS}" == "on" ]]; then
    out_ref+=(--model-family-interactions)
  elif [[ "${MODEL_FAMILY_INTERACTIONS}" == "off" ]]; then
    out_ref+=(--no-model-family-interactions)
  fi
  if [[ "${SPAIR_INDICATOR_INTERACTIONS}" == "on" ]]; then
    out_ref+=(--spair-indicator-interactions)
  elif [[ "${SPAIR_INDICATOR_INTERACTIONS}" == "off" ]]; then
    out_ref+=(--no-spair-indicator-interactions)
  fi
  if [[ "${FORCE_RERUN}" -eq 1 ]]; then
    out_ref+=(--force-rerun)
  fi

  case "${variant}" in
    base)
      ;;
    no_family)
      out_ref+=(--no-family-effects)
      ;;
    no_family_no_density)
      out_ref+=(--no-family-effects --no-density-controls)
      ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 2
      ;;
  esac
}

collect_finished() {
  local -a still_running=()
  local pid
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      still_running+=("${pid}")
      continue
    fi
    local label="${PID_TO_LABEL["${pid}"]}"
    local log_path="${PID_TO_LOG["${pid}"]}"
    if wait "${pid}"; then
      SUCCEEDED=$((SUCCEEDED + 1))
      echo "DONE ${label} (log: ${log_path})"
    else
      FAILED=$((FAILED + 1))
      echo "FAILED ${label} (log: ${log_path})" >&2
    fi
    unset PID_TO_LABEL["${pid}"]
    unset PID_TO_LOG["${pid}"]
  done
  PIDS=("${still_running[@]}")
}

launch_variant() {
  local variant="$1"
  local -a cmd=()
  build_args_for_variant "${variant}" cmd
  local label="ridge/${variant}"
  local log_path="${LOG_DIR}/ridge_${variant}.log"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY-RUN [%s]:' "${label}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    LAUNCHED=$((LAUNCHED + 1))
    return 0
  fi

  (
    set -euo pipefail
    echo "[$(date +%F' '%T)] START ${label}"
    run_cmd "${cmd[@]}"
    echo "[$(date +%F' '%T)] DONE  ${label}"
  ) >"${log_path}" 2>&1 &
  local pid=$!
  PIDS+=("${pid}")
  PID_TO_LABEL["${pid}"]="${label}"
  PID_TO_LOG["${pid}"]="${log_path}"
  LAUNCHED=$((LAUNCHED + 1))
  echo "LAUNCHED ${label} (pid=${pid}, log: ${log_path})"
}

for variant in "${VARIANTS[@]}"; do
  launch_variant "${variant}"
  while [[ "${DRY_RUN}" -eq 0 && "${#PIDS[@]}" -ge "${MAX_PARALLEL}" ]]; do
    collect_finished
    if [[ "${#PIDS[@]}" -ge "${MAX_PARALLEL}" ]]; then
      sleep 2
    fi
  done
done

while [[ "${DRY_RUN}" -eq 0 && "${#PIDS[@]}" -gt 0 ]]; do
  collect_finished
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    sleep 2
  fi
done

echo
echo "Done. launched=${LAUNCHED}, succeeded=${SUCCEEDED}, failed=${FAILED}, dry_run=${DRY_RUN}"
echo "Logs: ${LOG_DIR}"
if [[ "${DRY_RUN}" -eq 0 && "${FAILED}" -gt 0 ]]; then
  exit 1
fi
