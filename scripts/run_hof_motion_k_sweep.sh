#!/usr/bin/env bash
set -euo pipefail

# Build HOF motion-k coverage tables (k in list) and run leakage-free evals:
# - leakage_free_hof_motion_k${k}
# - leakage_free_hof_motion_k${k}_train_only
# - leakage_free_hof_motion_k${k}_eval_only
#
# Supports ridge/ols and optional pairwise companion runs.
#
# Usage:
#   bash scripts/run_hof_motion_k_sweep.sh --root <analysis_root>
#   bash scripts/run_hof_motion_k_sweep.sh --root <analysis_root> --linear-model ols --prediction-model ols
#   bash scripts/run_hof_motion_k_sweep.sh --root <analysis_root> --no-density-controls --pairwise-all
#   bash scripts/run_hof_motion_k_sweep.sh --root <analysis_root> --force-rerun

ROOT="${ROOT:-analysis_comprehensive_runs/hof_motion_v3_density_jointood_full_ridge_a100_v3}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis}"
HOF_SOURCE_CSV="${HOF_SOURCE_CSV:-${ANALYSIS_ROOT}/coverage_v2_hof_full_occ.csv}"
FLOW_COVERAGE_CSV="${FLOW_COVERAGE_CSV:-${ANALYSIS_ROOT}/coverage_v2_flow_only_raw_joint_full.csv}"
FLOW_STATS_DIR="${FLOW_STATS_DIR:-/mnt/nvme_1tb_b/coverage_vectors/stats}"
K_VALUES="${K_VALUES:-5,10,20,40}"
LINEAR_MODEL="${LINEAR_MODEL:-ridge}"
PREDICTION_MODEL="${PREDICTION_MODEL:-ridge}"
RIDGE_ALPHA="${RIDGE_ALPHA:-100}"
RANKING_GROUP="${RANKING_GROUP:-train_dataset}"
PAIRWISE_GROUP_COLS="${PAIRWISE_GROUP_COLS:-}"
RANKING_CONTEXT_COLS="${RANKING_CONTEXT_COLS:-}"
FLOW_MMD_CSV="${FLOW_MMD_CSV:-}"
FEATURE_MMD_CSV="${FEATURE_MMD_CSV:-}"
DINO_MMD_CSV="${DINO_MMD_CSV:-}"
COLLAPSE_CV_CELLS=1
NO_DENSITY_CONTROLS=0
NO_FAMILY_EFFECTS=0
PAIRWISE_ALL=0
FORCE_RERUN=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --analysis-root) ANALYSIS_ROOT="$2"; shift 2 ;;
    --hof-source-csv) HOF_SOURCE_CSV="$2"; shift 2 ;;
    --flow-coverage-csv) FLOW_COVERAGE_CSV="$2"; shift 2 ;;
    --flow-stats-dir) FLOW_STATS_DIR="$2"; shift 2 ;;
    --k-values) K_VALUES="$2"; shift 2 ;;
    --linear-model) LINEAR_MODEL="$2"; shift 2 ;;
    --prediction-model) PREDICTION_MODEL="$2"; shift 2 ;;
    --ridge-alpha) RIDGE_ALPHA="$2"; shift 2 ;;
    --ranking-group) RANKING_GROUP="$2"; shift 2 ;;
    --pairwise-group-cols) PAIRWISE_GROUP_COLS="$2"; shift 2 ;;
    --ranking-context-cols) RANKING_CONTEXT_COLS="$2"; shift 2 ;;
    --flow-mmd-csv) FLOW_MMD_CSV="$2"; shift 2 ;;
    --feature-mmd-csv) FEATURE_MMD_CSV="$2"; shift 2 ;;
    --dino-mmd-csv) DINO_MMD_CSV="$2"; shift 2 ;;
    --collapse-cv-cells) COLLAPSE_CV_CELLS=1; shift 1 ;;
    --no-collapse-cv-cells) COLLAPSE_CV_CELLS=0; shift 1 ;;
    --no-density-controls) NO_DENSITY_CONTROLS=1; shift 1 ;;
    --no-family-effects) NO_FAMILY_EFFECTS=1; shift 1 ;;
    --pairwise-all) PAIRWISE_ALL=1; shift 1 ;;
    --no-pairwise-all) PAIRWISE_ALL=0; shift 1 ;;
    --force-rerun) FORCE_RERUN=1; shift 1 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

ROOT="$(realpath "${ROOT}")"
DENSITY_OUT="${ROOT}/density_joint"
mkdir -p "${DENSITY_OUT}"
if [[ -z "${FLOW_MMD_CSV}" ]]; then
  FLOW_MMD_CSV="${ROOT}/mmd/mmd_v2_flow_joint_v1.csv"
fi
if [[ -z "${DINO_MMD_CSV}" ]]; then
  DINO_MMD_CSV="${ROOT}/mmd/mmd_v2_dino_v1.csv"
fi
if [[ -z "${FEATURE_MMD_CSV}" ]]; then
  FEATURE_MMD_CSV="${ROOT}/mmd/mmd_v2_feature_v1.csv"
fi
if [[ ! -f "${FEATURE_MMD_CSV}" && -f "${DINO_MMD_CSV}" ]]; then
  FEATURE_MMD_CSV="${DINO_MMD_CSV}"
fi

if [[ ! -f "${HOF_SOURCE_CSV}" ]]; then
  echo "Missing HOF source CSV: ${HOF_SOURCE_CSV}" >&2
  exit 1
fi
if [[ ! -f "${FLOW_COVERAGE_CSV}" ]]; then
  echo "Missing flow coverage CSV: ${FLOW_COVERAGE_CSV}" >&2
  exit 1
fi

SNAPSHOTS=(
  /mnt/nvme_1tb_b/snapshots_ptody_fix
  /mnt/nvme_1tb_b/snapshots_synth_2d
  /mnt/nvme_1tb_b/snapshots_synthetic_long
  ./snapshots_2d_warps
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_mixed
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_raft
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_raft_2d_mix
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_spair_only
)

COMMON_ARGS=(
  --snapshots-dir "${SNAPSHOTS[@]}"
  --coverage-csv "${FLOW_COVERAGE_CSV}"
  --flow-stats-dir "${FLOW_STATS_DIR}"
  --target auc_normalized_observed
  --joint-ood-holdout
  --linear-model "${LINEAR_MODEL}"
  --prediction-model "${PREDICTION_MODEL}"
  --ranking-group "${RANKING_GROUP}"
  --cv-standardize-mode local
  --strict-dataset-match
  --no-allow-unsplit-coverage
  --no-allow-unsplit-mmd
  --no-allow-unsplit-flow-stats
  --no-per-encoder
  --no-logit-coverage
  --no-flow-eps-predictors
  --no-flow-eps-weighted-predictors
  --no-flow-density-interactions
)
if [[ -n "${PAIRWISE_GROUP_COLS}" ]]; then
  COMMON_ARGS+=(--pairwise-group-cols "${PAIRWISE_GROUP_COLS}")
fi
if [[ -n "${RANKING_CONTEXT_COLS}" ]]; then
  COMMON_ARGS+=(--ranking-context-cols "${RANKING_CONTEXT_COLS}")
fi
if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
  COMMON_ARGS+=(--collapse-cv-cells)
else
  COMMON_ARGS+=(--no-collapse-cv-cells)
fi
if [[ -f "${FLOW_MMD_CSV}" ]]; then
  COMMON_ARGS+=(--flow-mmd-csv "${FLOW_MMD_CSV}")
fi
if [[ -f "${FEATURE_MMD_CSV}" ]]; then
  COMMON_ARGS+=(--feature-mmd-csv "${FEATURE_MMD_CSV}")
fi
if [[ -f "${DINO_MMD_CSV}" ]]; then
  COMMON_ARGS+=(--dino-mmd-csv "${DINO_MMD_CSV}")
fi
if [[ "${NO_FAMILY_EFFECTS}" -eq 1 ]]; then
  COMMON_ARGS+=(
    --no-encoder-main-effects
    --no-encoder-interactions
    --no-model-family-main-effects
    --no-model-family-interactions
  )
else
  COMMON_ARGS+=(--model-family-main-effects)
fi
if [[ "${NO_DENSITY_CONTROLS}" -eq 1 ]]; then
  COMMON_ARGS+=(--no-use-flow-density-predictors)
else
  COMMON_ARGS+=(--use-flow-density-predictors)
fi
if [[ "${LINEAR_MODEL}" == "ridge" || "${PREDICTION_MODEL}" == "ridge" ]]; then
  COMMON_ARGS+=(--ridge-alpha "${RIDGE_ALPHA}")
fi

run_count=0
skip_count=0

run_one() {
  local out_dir="$1"
  local hof_csv="$2"
  local preds="$3"

  if [[ -d "${out_dir}" && "${FORCE_RERUN}" -eq 0 ]]; then
    echo "SKIP existing: ${out_dir}"
    skip_count=$((skip_count + 1))
  else
    if [[ -d "${out_dir}" && "${FORCE_RERUN}" -eq 1 ]]; then
      echo "RERUN existing: ${out_dir}"
    fi
    local cmd=(
      python scripts/build_leakage_free_eval.py
      "${COMMON_ARGS[@]}"
      --coverage-hof-csv "${hof_csv}"
      --output-dir "${out_dir}"
      --predictors "${preds}"
    )
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf 'DRY-RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      run_count=$((run_count + 1))
    else
      printf 'RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      "${cmd[@]}"
      run_count=$((run_count + 1))
    fi
  fi

  if [[ "${PAIRWISE_ALL}" -eq 1 ]]; then
    local pair_out="${out_dir}_pairwise"
    if [[ -d "${pair_out}" && "${FORCE_RERUN}" -eq 0 ]]; then
      echo "SKIP existing: ${pair_out}"
      skip_count=$((skip_count + 1))
      return 0
    fi
    if [[ -d "${pair_out}" && "${FORCE_RERUN}" -eq 1 ]]; then
      echo "RERUN existing: ${pair_out}"
    fi
    local pair_cmd=(
      python scripts/build_leakage_free_eval.py
      --snapshots-dir "${SNAPSHOTS[@]}"
      --coverage-csv "${FLOW_COVERAGE_CSV}"
      --flow-stats-dir "${FLOW_STATS_DIR}"
      --target auc_normalized_observed
      --joint-ood-holdout
      --linear-model pairwise_rank
      --prediction-model pairwise_rank
      --ranking-group "${RANKING_GROUP}"
      --cv-standardize-mode local
      --strict-dataset-match
      --no-allow-unsplit-coverage
      --no-allow-unsplit-mmd
      --no-allow-unsplit-flow-stats
      --no-per-encoder
      --no-logit-coverage
      --no-flow-eps-predictors
      --no-flow-eps-weighted-predictors
      --no-flow-density-interactions
      --coverage-hof-csv "${hof_csv}"
      --output-dir "${pair_out}"
      --predictors "${preds}"
    )
    if [[ -n "${PAIRWISE_GROUP_COLS}" ]]; then
      pair_cmd+=(--pairwise-group-cols "${PAIRWISE_GROUP_COLS}")
    fi
    if [[ -n "${RANKING_CONTEXT_COLS}" ]]; then
      pair_cmd+=(--ranking-context-cols "${RANKING_CONTEXT_COLS}")
    fi
    if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
      pair_cmd+=(--collapse-cv-cells)
    else
      pair_cmd+=(--no-collapse-cv-cells)
    fi
    if [[ -f "${FLOW_MMD_CSV}" ]]; then
      pair_cmd+=(--flow-mmd-csv "${FLOW_MMD_CSV}")
    fi
    if [[ -f "${FEATURE_MMD_CSV}" ]]; then
      pair_cmd+=(--feature-mmd-csv "${FEATURE_MMD_CSV}")
    fi
    if [[ -f "${DINO_MMD_CSV}" ]]; then
      pair_cmd+=(--dino-mmd-csv "${DINO_MMD_CSV}")
    fi
    if [[ "${NO_FAMILY_EFFECTS}" -eq 1 ]]; then
      pair_cmd+=(
        --no-encoder-main-effects
        --no-encoder-interactions
        --no-model-family-main-effects
        --no-model-family-interactions
      )
    else
      pair_cmd+=(--model-family-main-effects)
    fi
    if [[ "${NO_DENSITY_CONTROLS}" -eq 1 ]]; then
      pair_cmd+=(--no-use-flow-density-predictors)
    else
      pair_cmd+=(--use-flow-density-predictors)
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf 'DRY-RUN:'
      printf ' %q' "${pair_cmd[@]}"
      printf '\n'
      run_count=$((run_count + 1))
    else
      printf 'RUN:'
      printf ' %q' "${pair_cmd[@]}"
      printf '\n'
      "${pair_cmd[@]}"
      run_count=$((run_count + 1))
    fi
  fi
}

IFS=',' read -r -a K_ARR <<< "${K_VALUES}"
for k in "${K_ARR[@]}"; do
  k="$(echo "${k}" | xargs)"
  [[ -n "${k}" ]] || continue

  HOF_K_CSV="${ROOT}/hof_coverage_rnorm_motion_k${k}.csv"
  if [[ ! -f "${HOF_K_CSV}" ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "DRY-RUN: build ${HOF_K_CSV} from ${HOF_SOURCE_CSV} (k=${k})"
    else
      python - "$HOF_SOURCE_CSV" "$HOF_K_CSV" "$k" <<'PY'
import pandas as pd
import sys

src, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(src)
need = [
    f"train_covered_by_eval_rnorm_k{k}",
    f"eval_covered_by_train_rnorm_k{k}",
    f"train_outside_eval_rnorm_k{k}",
    f"mean_nn_eval_to_train_k{k}",
    f"median_nn_eval_to_train_k{k}",
    f"p90_nn_eval_to_train_k{k}",
    f"mean_nn_train_to_eval_k{k}",
    f"median_nn_train_to_eval_k{k}",
    f"p90_nn_train_to_eval_k{k}",
]
missing = [c for c in need if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns for k={k}: {missing}")

out_df = pd.DataFrame({
    "train_dataset": df["train_dataset"],
    "train_split": df["train_split"],
    "eval_dataset": df["eval_dataset"],
    "eval_split": df["eval_split"],
    "train_n_vectors": df.get("n_train"),
    "eval_n_vectors": df.get("n_eval"),
    "dim": df.get("dim"),
    "train_radius": None,
    "eval_radius": None,
    "train_to_eval_coverage": df[f"train_covered_by_eval_rnorm_k{k}"],
    "eval_to_train_coverage": df[f"eval_covered_by_train_rnorm_k{k}"],
    "outside": df[f"train_outside_eval_rnorm_k{k}"],
    "k": k,
    "radius_quantile": None,
    "mean_nn_eval_to_train": df[f"mean_nn_eval_to_train_k{k}"],
    "median_nn_eval_to_train": df[f"median_nn_eval_to_train_k{k}"],
    "p90_nn_eval_to_train": df[f"p90_nn_eval_to_train_k{k}"],
    "mean_nn_train_to_eval": df[f"mean_nn_train_to_eval_k{k}"],
    "median_nn_train_to_eval": df[f"median_nn_train_to_eval_k{k}"],
    "p90_nn_train_to_eval": df[f"p90_nn_train_to_eval_k{k}"],
    "hof_density_l2": df.get("hof_density_l2"),
    "hof_density_l1": df.get("hof_density_l1"),
    "hof_density_cosine": df.get("hof_density_cosine"),
})
out_df.to_csv(out, index=False)
print(f"Wrote {out} ({len(out_df)} rows)")
PY
    fi
  fi

  run_one "${DENSITY_OUT}/leakage_free_hof_motion_k${k}" "${HOF_K_CSV}" \
    "hof_eval_to_train_mean_dist,hof_train_to_eval_mean_dist"
  run_one "${DENSITY_OUT}/leakage_free_hof_motion_k${k}_train_only" "${HOF_K_CSV}" \
    "hof_train_to_eval_mean_dist"
  run_one "${DENSITY_OUT}/leakage_free_hof_motion_k${k}_eval_only" "${HOF_K_CSV}" \
    "hof_eval_to_train_mean_dist"
done

echo "Done. launched=${run_count}, skipped_existing=${skip_count}, dry_run=${DRY_RUN}"
