#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "This script requires bash. Run it as: bash scripts/run_hypothesis_ablation_sixpack.sh ..." >&2
  exit 1
fi

# Orchestrate six hypothesis-ablation sweeps:
# - Models: ridge(alpha), ols
# - Variants: base, no_family, no_family_no_density
# For each (model, variant), run:
# 1) comprehensive sweep
# 2) 1-flow + 1-dino combo sweep
# 3) HOF motion-k sweep
#
# Example:
#   bash scripts/run_hypothesis_ablation_sixpack.sh \
#     --input-root analysis \
#     --output-prefix analysis_comprehensive_runs/hof_motion_v3_density_jointood_full \
#     --run-tag v4 \
#     --ridge-alpha 10

INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-analysis_comprehensive_runs/hof_motion_v3_density_jointood_full}"
RUN_TAG="${RUN_TAG:-}"
RIDGE_ALPHA="${RIDGE_ALPHA:-10}"
K_VALUES="${K_VALUES:-5,10,20,40}"
RANKING_GROUP="${RANKING_GROUP:-train_dataset}"
PAIRWISE_GROUP_COLS="${PAIRWISE_GROUP_COLS:-}"
RANKING_CONTEXT_COLS="${RANKING_CONTEXT_COLS:-}"
MODELS_CSV="${MODELS_CSV:-ridge,ols}"
VARIANTS_CSV="${VARIANTS_CSV:-base,no_family,no_family_no_density}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
LOG_DIR="${LOG_DIR:-}"

JOINT_OOD_HOLDOUT=1
PAIRWISE_ALL=1
FORCE_RERUN=0
ALLOW_BENCHMARK_RANKING_GROUP=0
DRY_RUN=0

# 16c/32t default profile:
# - run up to 6 variant jobs concurrently (the full sixpack),
# - keep each job to 2 BLAS/OpenMP threads by default.
# This yields ~12 active compute threads plus Python/runtime overhead.
# Users can override these via environment or CLI.
PER_JOB_THREADS="${PER_JOB_THREADS:-2}"
: "${OMP_NUM_THREADS:=${PER_JOB_THREADS}}"
: "${MKL_NUM_THREADS:=${PER_JOB_THREADS}}"
: "${OPENBLAS_NUM_THREADS:=${PER_JOB_THREADS}}"
: "${NUMEXPR_NUM_THREADS:=${PER_JOB_THREADS}}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_hypothesis_ablation_sixpack.sh [options]

Options:
  --input-root <path>             Input analysis root (default: analysis)
  --output-prefix <path>          Prefix for output roots
  --run-tag <tag>                 Optional suffix appended to each output root
  --models <csv>                  Subset of models: ridge,ols
  --variants <csv>                Subset of variants:
                                  base,no_family,no_density,no_family_no_density
  --ridge-alpha <value>           Ridge alpha (default: 10)
  --k-values <csv>                HOF k values (default: 5,10,20,40)
  --ranking-group <name>          Option ranking group (default: train_dataset)
  --pairwise-group-cols <csv>     Pairwise pair-construction groups for pairwise_rank
  --ranking-context-cols <csv>    Extra context groups for rank evaluation summaries
  --max-parallel <n>              Max model/variant jobs in parallel (default: 6)
  --per-job-threads <n>           BLAS/OpenMP thread cap per job (default: 2)
  --log-dir <path>                Directory for per-variant logs
  --allow-benchmark-ranking-group Allow benchmark grouping intentionally
  --joint-ood-holdout             Enable joint OOD holdout (default)
  --no-joint-ood-holdout          Disable joint OOD holdout
  --pairwise-all                  Run pairwise companions (default)
  --no-pairwise-all               Disable pairwise companions
  --force-rerun                   Re-run methods even if output dirs already exist
  --dry-run                       Print commands without executing
  -h, --help                      Show this help
USAGE
}

trim_ws() {
  local v="$1"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  printf '%s' "${v}"
}

parse_csv_into_array() {
  local csv="$1"
  local -n out_ref="$2"
  out_ref=()
  IFS=',' read -r -a _tmp <<< "${csv}"
  local item
  for item in "${_tmp[@]}"; do
    item="$(trim_ws "${item}")"
    [[ -n "${item}" ]] || continue
    out_ref+=("${item}")
  done
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

run_variant() {
  local model="$1"
  local variant="$2"
  local root="$3"

  local -a model_args=(--linear-model "${model}" --prediction-model "${model}")
  if [[ "${model}" == "ridge" ]]; then
    model_args+=(--ridge-alpha "${RIDGE_ALPHA}")
  fi

  local -a variant_args=()
  case "${variant}" in
    base)
      ;;
    no_family)
      variant_args+=(--no-family-effects)
      ;;
    no_density)
      variant_args+=(--no-density-controls)
      ;;
    no_family_no_density)
      variant_args+=(--no-family-effects --no-density-controls)
      ;;
  esac

  local -a joint_args=()
  if [[ "${JOINT_OOD_HOLDOUT}" -eq 1 ]]; then
    joint_args+=(--joint-ood-holdout)
  else
    joint_args+=(--no-joint-ood-holdout)
  fi

  local -a pairwise_args=()
  if [[ "${PAIRWISE_ALL}" -eq 1 ]]; then
    pairwise_args+=(--pairwise-all)
  else
    pairwise_args+=(--no-pairwise-all)
  fi
  local -a force_args=()
  if [[ "${FORCE_RERUN}" -eq 1 ]]; then
    force_args+=(--force-rerun)
  fi
  local -a context_args=()
  if [[ -n "${PAIRWISE_GROUP_COLS}" ]]; then
    context_args+=(--pairwise-group-cols "${PAIRWISE_GROUP_COLS}")
  fi
  if [[ -n "${RANKING_CONTEXT_COLS}" ]]; then
    context_args+=(--ranking-context-cols "${RANKING_CONTEXT_COLS}")
  fi

  run_cmd \
    bash scripts/run_comprehensive_sweep_latest.sh \
    --input-root "${INPUT_ROOT}" \
    --output-root "${root}" \
    --ranking-group "${RANKING_GROUP}" \
    --no-per-encoder \
    "${context_args[@]}" \
    "${force_args[@]}" \
    "${joint_args[@]}" \
    "${model_args[@]}" \
    "${pairwise_args[@]}" \
    "${variant_args[@]}"

  run_cmd \
    bash scripts/run_single_flow_single_dino_combos.sh \
    --root "${root}" \
    --analysis-root "${INPUT_ROOT}" \
    --ranking-group "${RANKING_GROUP}" \
    "${context_args[@]}" \
    "${force_args[@]}" \
    "${model_args[@]}" \
    "${pairwise_args[@]}" \
    "${variant_args[@]}"

  run_cmd \
    bash scripts/run_hof_motion_k_sweep.sh \
    --root "${root}" \
    --analysis-root "${INPUT_ROOT}" \
    --k-values "${K_VALUES}" \
    --ranking-group "${RANKING_GROUP}" \
    "${context_args[@]}" \
    "${force_args[@]}" \
    "${model_args[@]}" \
    "${pairwise_args[@]}" \
    "${variant_args[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-prefix)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --models)
      MODELS_CSV="$2"
      shift 2
      ;;
    --variants)
      VARIANTS_CSV="$2"
      shift 2
      ;;
    --ridge-alpha)
      RIDGE_ALPHA="$2"
      shift 2
      ;;
    --k-values)
      K_VALUES="$2"
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
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --per-job-threads)
      PER_JOB_THREADS="$2"
      OMP_NUM_THREADS="$2"
      MKL_NUM_THREADS="$2"
      OPENBLAS_NUM_THREADS="$2"
      NUMEXPR_NUM_THREADS="$2"
      export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --allow-benchmark-ranking-group)
      ALLOW_BENCHMARK_RANKING_GROUP=1
      shift 1
      ;;
    --joint-ood-holdout)
      JOINT_OOD_HOLDOUT=1
      shift 1
      ;;
    --no-joint-ood-holdout)
      JOINT_OOD_HOLDOUT=0
      shift 1
      ;;
    --pairwise-all)
      PAIRWISE_ALL=1
      shift 1
      ;;
    --no-pairwise-all)
      PAIRWISE_ALL=0
      shift 1
      ;;
    --force-rerun)
      FORCE_RERUN=1
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

if [[ "${RANKING_GROUP}" == "benchmark" && "${ALLOW_BENCHMARK_RANKING_GROUP}" -ne 1 ]]; then
  echo "Refusing --ranking-group benchmark. Use --ranking-group train_dataset (recommended)." >&2
  echo "Override only intentionally with --allow-benchmark-ranking-group." >&2
  exit 1
fi

if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  echo "Invalid --max-parallel value: ${MAX_PARALLEL}. Must be an integer >= 1." >&2
  exit 1
fi
if ! [[ "${PER_JOB_THREADS}" =~ ^[0-9]+$ ]] || [[ "${PER_JOB_THREADS}" -lt 1 ]]; then
  echo "Invalid --per-job-threads value: ${PER_JOB_THREADS}. Must be an integer >= 1." >&2
  exit 1
fi

parse_csv_into_array "${MODELS_CSV}" MODELS
parse_csv_into_array "${VARIANTS_CSV}" VARIANTS
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models selected." >&2
  exit 1
fi
if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
  echo "No variants selected." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  case "${model}" in
    ridge|ols) ;;
    *)
      echo "Invalid model: ${model}. Allowed: ridge, ols" >&2
      exit 1
      ;;
  esac
done

for variant in "${VARIANTS[@]}"; do
  case "${variant}" in
    base|no_family|no_density|no_family_no_density) ;;
    *)
      echo "Invalid variant: ${variant}. Allowed: base, no_family, no_density, no_family_no_density" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${OUTPUT_PREFIX}_logs"
  if [[ -n "${RUN_TAG}" ]]; then
    LOG_DIR="${LOG_DIR}_${RUN_TAG}"
  fi
fi
mkdir -p "${LOG_DIR}"

LAUNCHED=0
SUCCEEDED=0
FAILED=0
PIDS=()
declare -A PID_TO_LABEL=()
declare -A PID_TO_LOG=()

launch_one() {
  local model="$1"
  local variant="$2"
  local model_tag="$3"
  local root="$4"
  local label="${model}/${variant}"
  local log_path="${LOG_DIR}/${model_tag}_${variant}.log"

  echo
  echo "=== Variant: model=${model}, variant=${variant} ==="
  echo "root=${root}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    run_variant "${model}" "${variant}" "${root}"
    LAUNCHED=$((LAUNCHED + 1))
    SUCCEEDED=$((SUCCEEDED + 1))
    return 0
  fi

  (
    run_variant "${model}" "${variant}" "${root}"
  ) >"${log_path}" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  PID_TO_LABEL["${pid}"]="${label}"
  PID_TO_LOG["${pid}"]="${log_path}"
  LAUNCHED=$((LAUNCHED + 1))
  echo "LAUNCHED ${label} pid=${pid} log=${log_path}"
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

for model in "${MODELS[@]}"; do
  model_tag="${model}"
  if [[ "${model}" == "ridge" ]]; then
    model_tag="ridge_a${RIDGE_ALPHA}"
  fi

  for variant in "${VARIANTS[@]}"; do
    root="${OUTPUT_PREFIX}_${model_tag}_${variant}"
    if [[ -n "${RUN_TAG}" ]]; then
      root="${root}_${RUN_TAG}"
    fi

    launch_one "${model}" "${variant}" "${model_tag}" "${root}"

    while [[ "${DRY_RUN}" -eq 0 && "${#PIDS[@]}" -ge "${MAX_PARALLEL}" ]]; do
      collect_finished
      if [[ "${#PIDS[@]}" -ge "${MAX_PARALLEL}" ]]; then
        sleep 2
      fi
    done
  done
done

while [[ "${DRY_RUN}" -eq 0 && "${#PIDS[@]}" -gt 0 ]]; do
  collect_finished
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    sleep 2
  fi
done

echo
echo "Done. model_variant_runs=${LAUNCHED}, succeeded=${SUCCEEDED}, failed=${FAILED}, dry_run=${DRY_RUN}"
if [[ "${DRY_RUN}" -eq 0 && "${FAILED}" -gt 0 ]]; then
  exit 1
fi
