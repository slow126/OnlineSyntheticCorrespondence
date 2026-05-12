#!/usr/bin/env bash
set -euo pipefail

# Comprehensive sweep runner:
# - Runs joint density analysis into an isolated output root
# - Converts MMD v2 outputs to v1 format for leakage_free_eval
# - Runs MMD-only and asym+MMD analyses
# - Builds a method summary table

INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
DENSITY_ROOT_OVERRIDE=""
MMD_FLOW_V2="${MMD_FLOW_V2:-analysis/mmd_v2_flow_joint.csv}"
MMD_DINO_V2="${MMD_DINO_V2:-analysis/mmd_v2_dino.csv}"
MMD_FEATURE_V2="${MMD_FEATURE_V2:-analysis/mmd_v2_feature.csv}"
DINO_COVERAGE_CSV="${DINO_COVERAGE_CSV:-${INPUT_ROOT}/coverage_v2_dino_full_fast.csv}"
DINO_COVERAGE_RNORM=""
HOF_COVERAGE_CSV="${HOF_COVERAGE_CSV:-${INPUT_ROOT}/coverage_v2_hof_full.csv}"
HOF_COVERAGE_RNORM=""
KL_FLOW_V2="${KL_FLOW_V2:-analysis/kl_v2_flow_joint.csv}"
KL_DINO_V2="${KL_DINO_V2:-analysis/kl_v2_dino_features.csv}"
KL_HOF_V2="${KL_HOF_V2:-analysis/kl_v2_hof_full.csv}"
KL_K="${KL_K:-5}"
KL_FLOW_CSV=""
KL_DINO_CSV=""
KL_HOF_CSV=""

SKIP_DENSITY=0
SKIP_MMD=0
SKIP_SUMMARY=0
SKIP_COMBOS=0
PAIRWISE_ALL=1
FORCE_RERUN=0
DISABLE_MIXEDLM=0
NO_FAMILY_EFFECTS=0
NO_DENSITY_CONTROLS=0
MODEL_FAMILY_INTERACTIONS=""
SPAIR_INDICATOR_INTERACTIONS=""
RANKING_GROUP="train_dataset"
PAIRWISE_GROUP_COLS=""
RANKING_CONTEXT_COLS=""
JOINT_OOD_HOLDOUT=0
LINEAR_MODEL=""
PREDICTION_MODEL=""
RIDGE_ALPHA=""
PER_ENCODER=1
PREDICTION_CLIP_SET=0
PREDICTION_CLIP=0
COLLAPSE_CV_CELLS=0
ALLOW_MIXED_CONTEXT_COLLAPSE=0
FIT_SAMPLE_WEIGHTING="none"
FIT_BALANCE_REAL_SYNTH=0
OVERALL_AGGREGATION="micro"
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
    --density-root)
      DENSITY_ROOT_OVERRIDE="$2"
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
    --kl-flow-v2)
      KL_FLOW_V2="$2"
      shift 2
      ;;
    --kl-dino-v2)
      KL_DINO_V2="$2"
      shift 2
      ;;
    --kl-hof-v2)
      KL_HOF_V2="$2"
      shift 2
      ;;
    --kl-k)
      KL_K="$2"
      shift 2
      ;;
    --dino-coverage)
      DINO_COVERAGE_CSV="$2"
      shift 2
      ;;
    --hof-coverage)
      HOF_COVERAGE_CSV="$2"
      shift 2
      ;;
    --skip-density)
      SKIP_DENSITY=1
      shift 1
      ;;
    --skip-mmd)
      SKIP_MMD=1
      shift 1
      ;;
    --skip-summary)
      SKIP_SUMMARY=1
      shift 1
      ;;
    --skip-combos)
      SKIP_COMBOS=1
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
    --no-mixedlm)
      DISABLE_MIXEDLM=1
      shift 1
      ;;
    --no-family-effects)
      NO_FAMILY_EFFECTS=1
      shift 1
      ;;
    --model-family-interactions)
      MODEL_FAMILY_INTERACTIONS="on"
      shift 1
      ;;
    --no-model-family-interactions)
      MODEL_FAMILY_INTERACTIONS="off"
      shift 1
      ;;
    --spair-indicator-interactions)
      SPAIR_INDICATOR_INTERACTIONS="on"
      shift 1
      ;;
    --no-spair-indicator-interactions)
      SPAIR_INDICATOR_INTERACTIONS="off"
      shift 1
      ;;
    --no-density-controls)
      NO_DENSITY_CONTROLS=1
      shift 1
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
    --joint-ood-holdout)
      JOINT_OOD_HOLDOUT=1
      shift 1
      ;;
    --no-joint-ood-holdout)
      JOINT_OOD_HOLDOUT=0
      shift 1
      ;;
    --linear-model)
      LINEAR_MODEL="$2"
      shift 2
      ;;
    --prediction-model)
      PREDICTION_MODEL="$2"
      shift 2
      ;;
    --ridge-alpha)
      RIDGE_ALPHA="$2"
      shift 2
      ;;
    --per-encoder)
      PER_ENCODER=1
      shift 1
      ;;
    --no-per-encoder)
      PER_ENCODER=0
      shift 1
      ;;
    --prediction-clip)
      PREDICTION_CLIP_SET=1
      PREDICTION_CLIP=1
      shift 1
      ;;
    --no-prediction-clip)
      PREDICTION_CLIP_SET=1
      PREDICTION_CLIP=0
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
    --allow-mixed-context-collapse)
      ALLOW_MIXED_CONTEXT_COLLAPSE=1
      shift 1
      ;;
    --fit-sample-weighting)
      FIT_SAMPLE_WEIGHTING="$2"
      shift 2
      ;;
    --fit-balance-real-synth)
      FIT_BALANCE_REAL_SYNTH=1
      shift 1
      ;;
    --no-fit-balance-real-synth)
      FIT_BALANCE_REAL_SYNTH=0
      shift 1
      ;;
    --overall-aggregation)
      OVERALL_AGGREGATION="$2"
      shift 2
      ;;
    --cv-residualize-target-by-context)
      CV_RESIDUALIZE_TARGET_BY_CONTEXT=1
      shift 1
      ;;
    --no-cv-residualize-target-by-context)
      CV_RESIDUALIZE_TARGET_BY_CONTEXT=0
      shift 1
      ;;
    --cv-residual-context-cols)
      CV_RESIDUAL_CONTEXT_COLS="$2"
      shift 2
      ;;
    --cv-residual-eval-space)
      CV_RESIDUAL_EVAL_SPACE="$2"
      shift 2
      ;;
    --cv-residual-target-transform)
      CV_RESIDUAL_TARGET_TRANSFORM="$2"
      shift 2
      ;;
    --cv-residual-target-std-eps)
      CV_RESIDUAL_TARGET_STD_EPS="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration)
      CV_FEWSHOT_CONTEXT_CALIBRATION=1
      shift 1
      ;;
    --no-cv-fewshot-context-calibration)
      CV_FEWSHOT_CONTEXT_CALIBRATION=0
      shift 1
      ;;
    --cv-fewshot-context-calibration-cols)
      CV_FEWSHOT_CONTEXT_CALIBRATION_COLS="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration-std-eps)
      CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration-min-group-size)
      CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration-backoff)
      CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=1
      shift 1
      ;;
    --no-cv-fewshot-context-calibration-backoff)
      CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=0
      shift 1
      ;;
    --cv-fewshot-context-calibration-k)
      CV_FEWSHOT_CONTEXT_CALIBRATION_K="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration-seed)
      CV_FEWSHOT_CONTEXT_CALIBRATION_SEED="$2"
      shift 2
      ;;
    --cv-repeat-aggregation)
      CV_REPEAT_AGGREGATION="$2"
      shift 2
      ;;
    --loto-single-predictor-baselines)
      LOTO_SINGLE_PREDICTOR_BASELINES=1
      shift 1
      ;;
    --no-loto-single-predictor-baselines)
      LOTO_SINGLE_PREDICTOR_BASELINES=0
      shift 1
      ;;
    --jointood-single-predictor-baselines)
      JOINTOOD_SINGLE_PREDICTOR_BASELINES=1
      shift 1
      ;;
    --no-jointood-single-predictor-baselines)
      JOINTOOD_SINGLE_PREDICTOR_BASELINES=0
      shift 1
      ;;
    --mixedlm)
      DISABLE_MIXEDLM=0
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

trim_ws() {
  local v="$1"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  printf '%s' "${v}"
}

if [[ "${COLLAPSE_CV_CELLS}" -eq 1 && "${ALLOW_MIXED_CONTEXT_COLLAPSE}" -ne 1 ]]; then
  if [[ -n "$(trim_ws "${RANKING_CONTEXT_COLS}")" ]]; then
    echo "Refusing --collapse-cv-cells with non-empty --ranking-context-cols (${RANKING_CONTEXT_COLS})." >&2
    echo "This can mix model contexts and contaminate LOBO/LOTO/Joint ranking metrics." >&2
    echo "Use --no-collapse-cv-cells (recommended), or add --allow-mixed-context-collapse to override." >&2
    exit 1
  fi
fi
case "${CV_RESIDUAL_EVAL_SPACE}" in
  absolute|residual)
    ;;
  *)
    echo "Invalid --cv-residual-eval-space: ${CV_RESIDUAL_EVAL_SPACE}" >&2
    exit 1
    ;;
esac
case "${CV_RESIDUAL_TARGET_TRANSFORM}" in
  residual|zscore)
    ;;
  *)
    echo "Invalid --cv-residual-target-transform: ${CV_RESIDUAL_TARGET_TRANSFORM}" >&2
    exit 1
    ;;
esac
case "${CV_REPEAT_AGGREGATION}" in
  none|mean|median)
    ;;
  *)
    echo "Invalid --cv-repeat-aggregation: ${CV_REPEAT_AGGREGATION}" >&2
    exit 1
    ;;
esac

if [[ -z "${OUTPUT_ROOT}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="analysis_comprehensive_runs/${TS}"
fi

# Backward-compatible HOF coverage fallback:
# prefer explicit --hof-coverage if provided; otherwise allow *_occ.csv.
if [[ ! -f "${HOF_COVERAGE_CSV}" ]]; then
  HOF_FALLBACK="${INPUT_ROOT}/coverage_v2_hof_full_occ.csv"
  if [[ -f "${HOF_FALLBACK}" ]]; then
    echo "Using HOF fallback coverage CSV: ${HOF_FALLBACK}"
    HOF_COVERAGE_CSV="${HOF_FALLBACK}"
  fi
fi

if [[ ! -f "${KL_HOF_V2}" ]]; then
  KL_HOF_FALLBACK="analysis/kl_v2_hof_full_occ.csv"
  if [[ -f "${KL_HOF_FALLBACK}" ]]; then
    echo "Using HOF KL fallback CSV: ${KL_HOF_FALLBACK}"
    KL_HOF_V2="${KL_HOF_FALLBACK}"
  fi
fi

SNAPSHOT_DIRS=(
  /mnt/nvme_1tb_b/snapshots_ptody_fix
  /mnt/nvme_1tb_b/snapshots_synth_2d
  /mnt/nvme_1tb_b/snapshots_synthetic_long
  ./snapshots_2d_warps
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_mixed
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_raft
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_raft_2d_mix
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_spair_only
)

run_eval() {
  local args=("$@")
  local out_dir=""
  local i

  for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--output-dir" ]]; then
      out_dir="${args[i+1]:-}"
    fi
  done

  if [[ "${DISABLE_MIXEDLM}" -eq 1 ]]; then
    args+=(--no-prediction-mixedlm --no-regression-mixedlm)
  fi

  python scripts/build_leakage_free_eval.py "${args[@]}"

  if [[ "${PAIRWISE_ALL}" -ne 1 ]]; then
    return 0
  fi
  if [[ -z "${out_dir}" ]]; then
    echo "Warning: --output-dir not found; skipping pairwise run." >&2
    return 0
  fi

  local pair_args=("${args[@]}")
  for ((i=0; i<${#pair_args[@]}; i++)); do
    if [[ "${pair_args[i]}" == "--output-dir" ]]; then
      pair_args[i+1]="${out_dir}_pairwise"
      break
    fi
  done
  local replaced=0
  for ((i=0; i<${#pair_args[@]}; i++)); do
    if [[ "${pair_args[i]}" == "--linear-model" ]]; then
      pair_args[i+1]="pairwise_rank"
      replaced=1
      break
    fi
  done
  if [[ "${replaced}" -eq 0 ]]; then
    pair_args+=(--linear-model pairwise_rank)
  fi
  replaced=0
  for ((i=0; i<${#pair_args[@]}; i++)); do
    if [[ "${pair_args[i]}" == "--prediction-model" ]]; then
      pair_args[i+1]="pairwise_rank"
      replaced=1
      break
    fi
  done
  if [[ "${replaced}" -eq 0 ]]; then
    pair_args+=(--prediction-model pairwise_rank)
  fi
  python scripts/build_leakage_free_eval.py "${pair_args[@]}"
}

DENSITY_ARGS_BASE=(
  --flow-stats-dir /mnt/nvme_1tb_b/coverage_vectors/stats
  --model-family-main-effects
  --cv-standardize-mode local
  --strict-dataset-match
  --no-allow-unsplit-coverage
  --no-allow-unsplit-mmd
  --no-allow-unsplit-flow-stats
)
if [[ "${NO_DENSITY_CONTROLS}" -eq 0 ]]; then
  DENSITY_ARGS_BASE+=(
    --use-flow-density-predictors
    --flow-density-interactions
  )
fi
if [[ "${MODEL_FAMILY_INTERACTIONS}" == "on" ]]; then
  DENSITY_ARGS_BASE+=(--model-family-interactions)
elif [[ "${MODEL_FAMILY_INTERACTIONS}" == "off" ]]; then
  DENSITY_ARGS_BASE+=(--no-model-family-interactions)
fi
if [[ "${SPAIR_INDICATOR_INTERACTIONS}" == "on" ]]; then
  DENSITY_ARGS_BASE+=(--spair-indicator-interactions)
elif [[ "${SPAIR_INDICATOR_INTERACTIONS}" == "off" ]]; then
  DENSITY_ARGS_BASE+=(--no-spair-indicator-interactions)
fi
DENSITY_ARGS_BASE+=(--ranking-group "${RANKING_GROUP}")
if [[ -n "${PAIRWISE_GROUP_COLS}" ]]; then
  DENSITY_ARGS_BASE+=(--pairwise-group-cols "${PAIRWISE_GROUP_COLS}")
fi
if [[ -n "${RANKING_CONTEXT_COLS}" ]]; then
  DENSITY_ARGS_BASE+=(--ranking-context-cols "${RANKING_CONTEXT_COLS}")
fi
if [[ "${JOINT_OOD_HOLDOUT}" -eq 1 ]]; then
  DENSITY_ARGS_BASE+=(--joint-ood-holdout)
fi
if [[ -n "${LINEAR_MODEL}" ]]; then
  DENSITY_ARGS_BASE+=(--linear-model "${LINEAR_MODEL}")
fi
if [[ -n "${PREDICTION_MODEL}" ]]; then
  DENSITY_ARGS_BASE+=(--prediction-model "${PREDICTION_MODEL}")
fi
if [[ -n "${RIDGE_ALPHA}" ]]; then
  DENSITY_ARGS_BASE+=(--ridge-alpha "${RIDGE_ALPHA}")
fi
if [[ "${PER_ENCODER}" -eq 0 ]]; then
  DENSITY_ARGS_BASE+=(--no-per-encoder)
fi
if [[ "${PREDICTION_CLIP_SET}" -eq 1 ]]; then
  if [[ "${PREDICTION_CLIP}" -eq 1 ]]; then
    DENSITY_ARGS_BASE+=(--prediction-clip)
  else
    DENSITY_ARGS_BASE+=(--no-prediction-clip)
  fi
fi
if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
  DENSITY_ARGS_BASE+=(--collapse-cv-cells)
else
  DENSITY_ARGS_BASE+=(--no-collapse-cv-cells)
fi
DENSITY_ARGS_BASE+=(--fit-sample-weighting "${FIT_SAMPLE_WEIGHTING}")
if [[ "${FIT_BALANCE_REAL_SYNTH}" -eq 1 ]]; then
  DENSITY_ARGS_BASE+=(--fit-balance-real-synth)
else
  DENSITY_ARGS_BASE+=(--no-fit-balance-real-synth)
fi
DENSITY_ARGS_BASE+=(--overall-aggregation "${OVERALL_AGGREGATION}")
if [[ "${CV_RESIDUALIZE_TARGET_BY_CONTEXT}" -eq 1 ]]; then
  DENSITY_ARGS_BASE+=(--cv-residualize-target-by-context)
fi
if [[ -n "${CV_RESIDUAL_CONTEXT_COLS}" ]]; then
  DENSITY_ARGS_BASE+=(--cv-residual-context-cols "${CV_RESIDUAL_CONTEXT_COLS}")
fi
DENSITY_ARGS_BASE+=(--cv-residual-eval-space "${CV_RESIDUAL_EVAL_SPACE}")
DENSITY_ARGS_BASE+=(--cv-residual-target-transform "${CV_RESIDUAL_TARGET_TRANSFORM}")
DENSITY_ARGS_BASE+=(--cv-residual-target-std-eps "${CV_RESIDUAL_TARGET_STD_EPS}")
if [[ "${CV_FEWSHOT_CONTEXT_CALIBRATION}" -eq 1 ]]; then
  DENSITY_ARGS_BASE+=(--cv-fewshot-context-calibration)
fi
if [[ -n "${CV_FEWSHOT_CONTEXT_CALIBRATION_COLS}" ]]; then
  DENSITY_ARGS_BASE+=(--cv-fewshot-context-calibration-cols "${CV_FEWSHOT_CONTEXT_CALIBRATION_COLS}")
fi
DENSITY_ARGS_BASE+=(--cv-fewshot-context-calibration-std-eps "${CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS}")
DENSITY_ARGS_BASE+=(--cv-fewshot-context-calibration-min-group-size "${CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE}")
if [[ "${CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF}" -eq 1 ]]; then
  DENSITY_ARGS_BASE+=(--cv-fewshot-context-calibration-backoff)
else
  DENSITY_ARGS_BASE+=(--no-cv-fewshot-context-calibration-backoff)
fi
DENSITY_ARGS_BASE+=(--cv-fewshot-context-calibration-k "${CV_FEWSHOT_CONTEXT_CALIBRATION_K}")
DENSITY_ARGS_BASE+=(--cv-fewshot-context-calibration-seed "${CV_FEWSHOT_CONTEXT_CALIBRATION_SEED}")
DENSITY_ARGS_BASE+=(--cv-repeat-aggregation "${CV_REPEAT_AGGREGATION}")
if [[ "${LOTO_SINGLE_PREDICTOR_BASELINES}" -eq 1 ]]; then
  DENSITY_ARGS_BASE+=(--loto-single-predictor-baselines)
else
  DENSITY_ARGS_BASE+=(--no-loto-single-predictor-baselines)
fi
if [[ "${JOINTOOD_SINGLE_PREDICTOR_BASELINES}" -eq 1 ]]; then
  DENSITY_ARGS_BASE+=(--jointood-single-predictor-baselines)
else
  DENSITY_ARGS_BASE+=(--no-jointood-single-predictor-baselines)
fi
if [[ "${NO_FAMILY_EFFECTS}" -eq 1 ]]; then
  DENSITY_ARGS_BASE+=(
    --no-encoder-main-effects
    --no-encoder-interactions
    --no-model-family-main-effects
    --no-model-family-interactions
  )
fi
DENSITY_ARGS=(
  "${DENSITY_ARGS_BASE[@]}"
  --target auc_normalized_observed
)

FLOW_EPS_VALUES="0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64"

if [[ -n "${DENSITY_ROOT_OVERRIDE}" ]]; then
  DENSITY_OUT="${DENSITY_ROOT_OVERRIDE}"
  SKIP_DENSITY=1
else
  DENSITY_OUT="${OUTPUT_ROOT}/density_joint"
fi
MMD_OUT="${OUTPUT_ROOT}/mmd"
MMD_FLOW_V1="${MMD_OUT}/mmd_v2_flow_joint_v1.csv"
MMD_DINO_V1="${MMD_OUT}/mmd_v2_dino_v1.csv"
MMD_FEATURE_V1="${MMD_OUT}/mmd_v2_feature_v1.csv"

build_eps_pred_list() {
  local prefix="$1"
  local suffix="$2"
  local out=""
  IFS=',' read -ra vals <<< "${FLOW_EPS_VALUES}"
  for v in "${vals[@]}"; do
    local v_clean="${v}"
    v_clean="${v_clean%px}"
    local key
    key="$(echo "${v_clean}" | sed 's/\./p/g')"
    out+="${prefix}${key}${suffix},"
  done
  echo "${out%,}"
}

ensure_kl_csv() {
  local in_path="$1"
  local out_path="$2"
  local k="$3"
  local expected_space="$4"
  if [[ -z "${in_path}" || ! -f "${in_path}" ]]; then
    return 1
  fi
  if [[ -f "${out_path}" ]]; then
    if [[ "${FORCE_RERUN}" -eq 0 ]]; then
      return 0
    fi
  fi
  KL_INPUT="${in_path}" \
  KL_OUTPUT="${out_path}" \
  KL_K="${k}" \
  KL_SPACE="${expected_space}" \
  python - <<'PY'
import os
import pandas as pd

in_path = os.environ.get("KL_INPUT")
out_path = os.environ.get("KL_OUTPUT")
k = str(os.environ.get("KL_K", "5")).strip()
space = (os.environ.get("KL_SPACE") or "").strip()
if not in_path or not out_path:
    raise SystemExit("Missing KL_INPUT or KL_OUTPUT env var.")

df = pd.read_csv(in_path)
if "space" in df.columns and space:
    subset = df[df["space"].astype(str) == space]
    if not subset.empty:
        df = subset

col_train = f"kl_train_to_eval_k{k}"
col_eval = f"kl_eval_to_train_k{k}"
if col_train in df.columns and col_eval in df.columns:
    df = df.copy()
    df["kl_train_to_eval"] = df[col_train]
    df["kl_eval_to_train"] = df[col_eval]
elif "kl_train_to_eval" in df.columns and "kl_eval_to_train" in df.columns:
    pass
else:
    raise SystemExit(
        f"Missing KL columns for k={k}. "
        f"Expected {col_train}/{col_eval} or kl_train_to_eval/kl_eval_to_train."
    )

cols = [
    "space",
    "train_dataset",
    "train_split",
    "eval_dataset",
    "eval_split",
    "train_n_vectors",
    "eval_n_vectors",
    "dim",
    "kl_train_to_eval",
    "kl_eval_to_train",
]
keep = [c for c in cols if c in df.columns]
df = df[keep].copy()
if "k" not in df.columns:
    df["k"] = int(k)
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} ({len(df)} rows)")
PY
}

ensure_mmd_v1_csv() {
  local in_path="$1"
  local out_path="$2"
  if [[ -z "${in_path}" || ! -f "${in_path}" ]]; then
    return 1
  fi
  if [[ -f "${out_path}" && "${FORCE_RERUN}" -eq 0 ]]; then
    return 0
  fi
  python scripts/convert_mmd_v2_to_v1.py \
    --input "${in_path}" \
    --output "${out_path}"
}

merge_kl_into_coverage() {
  local coverage_path="$1"
  local kl_path="$2"
  local out_path="$3"
  if [[ -z "${coverage_path}" || ! -f "${coverage_path}" ]]; then
    return 1
  fi
  if [[ -z "${kl_path}" || ! -f "${kl_path}" ]]; then
    return 1
  fi
  if [[ -f "${out_path}" ]]; then
    if [[ "${FORCE_RERUN}" -eq 0 ]]; then
      return 0
    fi
  fi
  COVERAGE_INPUT="${coverage_path}" \
  KL_INPUT="${kl_path}" \
  KL_OUTPUT="${out_path}" \
  python - <<'PY'
import os
import pandas as pd

coverage_path = os.environ.get("COVERAGE_INPUT")
kl_path = os.environ.get("KL_INPUT")
out_path = os.environ.get("KL_OUTPUT")
if not coverage_path or not kl_path or not out_path:
    raise SystemExit("Missing COVERAGE_INPUT/KL_INPUT/KL_OUTPUT env vars.")

coverage = pd.read_csv(coverage_path)
kl = pd.read_csv(kl_path)

join_cols = [c for c in ["train_dataset", "train_split", "eval_dataset", "eval_split"] if c in coverage.columns and c in kl.columns]
if not join_cols:
    raise SystemExit("No common join columns for KL merge.")

kl_cols = [c for c in ["kl_train_to_eval", "kl_eval_to_train"] if c in kl.columns]
if not kl_cols:
    raise SystemExit("KL CSV missing kl_train_to_eval/kl_eval_to_train columns.")

merged = coverage.merge(kl[join_cols + kl_cols], on=join_cols, how="left")
merged.to_csv(out_path, index=False)
print(f"Wrote {out_path} ({len(merged)} rows)")
PY
}

ensure_dino_rnorm_csv() {
  if [[ -z "${DINO_COVERAGE_CSV}" || ! -f "${DINO_COVERAGE_CSV}" ]]; then
    return 0
  fi
  DINO_COVERAGE_RNORM="${OUTPUT_ROOT}/dino_coverage_rnorm_k5.csv"
  if [[ -f "${DINO_COVERAGE_RNORM}" ]]; then
    if [[ "${FORCE_RERUN}" -eq 0 ]]; then
      return 0
    fi
  fi
  DINO_COVERAGE_CSV="${DINO_COVERAGE_CSV}" \
  DINO_COVERAGE_RNORM="${DINO_COVERAGE_RNORM}" \
  python - <<'PY'
import pandas as pd
import numpy as np
import os

in_path = os.environ.get("DINO_COVERAGE_CSV")
out_path = os.environ.get("DINO_COVERAGE_RNORM")
if not in_path or not out_path:
    raise SystemExit("Missing DINO_COVERAGE_CSV or DINO_COVERAGE_RNORM env var.")

df = pd.read_csv(in_path)
required = [
    "train_dataset","train_split","eval_dataset","eval_split",
    "train_radius","eval_radius",
    "eval_covered_by_train_rnorm_k5","train_covered_by_eval_rnorm_k5","train_outside_eval_rnorm_k5",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in DINO coverage CSV: {missing}")

out = pd.DataFrame({
    "space": df.get("space", "features"),
    "train_dataset": df["train_dataset"],
    "train_split": df["train_split"],
    "eval_dataset": df["eval_dataset"],
    "eval_split": df["eval_split"],
    "train_n_vectors": df.get("train_n_vectors", np.nan),
    "eval_n_vectors": df.get("eval_n_vectors", np.nan),
    "train_radius": df["train_radius"],
    "eval_radius": df["eval_radius"],
    "train_to_eval_coverage": df["train_covered_by_eval_rnorm_k5"],
    "eval_to_train_coverage": df["eval_covered_by_train_rnorm_k5"],
    "outside": df["train_outside_eval_rnorm_k5"],
    "k": 5,
    "radius_quantile": np.nan,
    "mean_nn_eval_to_train": df.get("mean_nn_eval_to_train_k1", np.nan),
    "median_nn_eval_to_train": df.get("median_nn_eval_to_train_k1", np.nan),
    "p90_nn_eval_to_train": df.get("p90_nn_eval_to_train_k1", np.nan),
    "mean_nn_train_to_eval": df.get("mean_nn_train_to_eval_k1", np.nan),
    "median_nn_train_to_eval": df.get("median_nn_train_to_eval_k1", np.nan),
    "p90_nn_train_to_eval": df.get("p90_nn_train_to_eval_k1", np.nan),
})
out.to_csv(out_path, index=False)
print(f"Wrote {out_path} ({len(out)} rows)")
PY
}

ensure_hof_rnorm_csv() {
  if [[ -z "${HOF_COVERAGE_CSV}" || ! -f "${HOF_COVERAGE_CSV}" ]]; then
    return 0
  fi
  HOF_COVERAGE_RNORM="${OUTPUT_ROOT}/hof_coverage_rnorm_k5.csv"
  if [[ -f "${HOF_COVERAGE_RNORM}" ]]; then
    if [[ "${FORCE_RERUN}" -eq 0 ]]; then
      return 0
    fi
  fi
  HOF_COVERAGE_CSV="${HOF_COVERAGE_CSV}" \
  HOF_COVERAGE_RNORM="${HOF_COVERAGE_RNORM}" \
  python - <<'PY'
import pandas as pd
import numpy as np
import os

in_path = os.environ.get("HOF_COVERAGE_CSV")
out_path = os.environ.get("HOF_COVERAGE_RNORM")
if not in_path or not out_path:
    raise SystemExit("Missing HOF_COVERAGE_CSV or HOF_COVERAGE_RNORM env var.")

df = pd.read_csv(in_path)
required = [
    "train_dataset","train_split","eval_dataset","eval_split",
    "eval_covered_by_train_rnorm_k5","train_covered_by_eval_rnorm_k5","train_outside_eval_rnorm_k5",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in HOF coverage CSV: {missing}")

out = pd.DataFrame({
    "train_dataset": df["train_dataset"],
    "train_split": df["train_split"],
    "eval_dataset": df["eval_dataset"],
    "eval_split": df["eval_split"],
    "train_n_vectors": df.get("n_train", np.nan),
    "eval_n_vectors": df.get("n_eval", np.nan),
    "dim": df.get("dim", np.nan),
    "train_radius": np.nan,
    "eval_radius": np.nan,
    "train_to_eval_coverage": df["train_covered_by_eval_rnorm_k5"],
    "eval_to_train_coverage": df["eval_covered_by_train_rnorm_k5"],
    "outside": df["train_outside_eval_rnorm_k5"],
    "k": 5,
    "radius_quantile": np.nan,
    "mean_nn_eval_to_train": df.get("mean_nn_eval_to_train_k1", np.nan),
    "median_nn_eval_to_train": df.get("median_nn_eval_to_train_k1", np.nan),
    "p90_nn_eval_to_train": df.get("p90_nn_eval_to_train_k1", np.nan),
    "mean_nn_train_to_eval": df.get("mean_nn_train_to_eval_k1", np.nan),
    "median_nn_train_to_eval": df.get("median_nn_train_to_eval_k1", np.nan),
    "p90_nn_train_to_eval": df.get("p90_nn_train_to_eval_k1", np.nan),
    "hof_density_l2": df.get("hof_density_l2", np.nan),
    "hof_density_l1": df.get("hof_density_l1", np.nan),
    "hof_density_cosine": df.get("hof_density_cosine", np.nan),
})
out.to_csv(out_path, index=False)
print(f"Wrote {out_path} ({len(out)} rows)")
PY
}

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${MMD_OUT}"

if [[ -f "${MMD_FLOW_V2}" ]]; then
  ensure_mmd_v1_csv "${MMD_FLOW_V2}" "${MMD_FLOW_V1}" || true
fi
if [[ -f "${MMD_DINO_V2}" ]]; then
  ensure_mmd_v1_csv "${MMD_DINO_V2}" "${MMD_DINO_V1}" || true
fi
if [[ -f "${MMD_FEATURE_V2}" ]]; then
  ensure_mmd_v1_csv "${MMD_FEATURE_V2}" "${MMD_FEATURE_V1}" || true
fi

if [[ -f "${MMD_FLOW_V1}" ]]; then
  DENSITY_ARGS_BASE+=(--flow-mmd-csv "${MMD_FLOW_V1}")
fi
if [[ -f "${MMD_DINO_V1}" ]]; then
  DENSITY_ARGS_BASE+=(--dino-mmd-csv "${MMD_DINO_V1}")
fi
if [[ -f "${MMD_FEATURE_V1}" ]]; then
  DENSITY_ARGS_BASE+=(--feature-mmd-csv "${MMD_FEATURE_V1}")
elif [[ -f "${MMD_DINO_V1}" ]]; then
  DENSITY_ARGS_BASE+=(--feature-mmd-csv "${MMD_DINO_V1}")
fi

DENSITY_ARGS=(
  "${DENSITY_ARGS_BASE[@]}"
  --target auc_normalized_observed
)

KL_OUT_DIR="${OUTPUT_ROOT}/kl"
mkdir -p "${KL_OUT_DIR}"
if [[ -f "${KL_FLOW_V2}" ]]; then
  KL_FLOW_CSV="${KL_OUT_DIR}/kl_flow_k${KL_K}.csv"
  ensure_kl_csv "${KL_FLOW_V2}" "${KL_FLOW_CSV}" "${KL_K}" "joint"
else
  echo "KL flow CSV missing (expected: ${KL_FLOW_V2})"
fi
if [[ -f "${KL_DINO_V2}" ]]; then
  KL_DINO_CSV="${KL_OUT_DIR}/kl_dino_k${KL_K}.csv"
  ensure_kl_csv "${KL_DINO_V2}" "${KL_DINO_CSV}" "${KL_K}" "features"
else
  echo "KL dino CSV missing (expected: ${KL_DINO_V2})"
fi
if [[ -f "${KL_HOF_V2}" ]]; then
  KL_HOF_CSV="${KL_OUT_DIR}/kl_hof_k${KL_K}.csv"
  ensure_kl_csv "${KL_HOF_V2}" "${KL_HOF_CSV}" "${KL_K}" "hof"
else
  echo "KL hof CSV missing (expected: ${KL_HOF_V2})"
fi

if [[ "${SKIP_DENSITY}" -eq 0 ]]; then
  DENSITY_SCRIPT_ARGS=(
    --input-root "${INPUT_ROOT}"
    --output-root "${DENSITY_OUT}"
  )
  if [[ "${NO_FAMILY_EFFECTS}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--no-family-effects)
  fi
  if [[ "${MODEL_FAMILY_INTERACTIONS}" == "on" ]]; then
    DENSITY_SCRIPT_ARGS+=(--model-family-interactions)
  elif [[ "${MODEL_FAMILY_INTERACTIONS}" == "off" ]]; then
    DENSITY_SCRIPT_ARGS+=(--no-model-family-interactions)
  fi
  if [[ "${SPAIR_INDICATOR_INTERACTIONS}" == "on" ]]; then
    DENSITY_SCRIPT_ARGS+=(--spair-indicator-interactions)
  elif [[ "${SPAIR_INDICATOR_INTERACTIONS}" == "off" ]]; then
    DENSITY_SCRIPT_ARGS+=(--no-spair-indicator-interactions)
  fi
  if [[ "${NO_DENSITY_CONTROLS}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--no-density-controls)
  else
    DENSITY_SCRIPT_ARGS+=(--density-controls)
  fi
  DENSITY_SCRIPT_ARGS+=(--ranking-group "${RANKING_GROUP}")
  if [[ -n "${PAIRWISE_GROUP_COLS}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--pairwise-group-cols "${PAIRWISE_GROUP_COLS}")
  fi
  if [[ -n "${RANKING_CONTEXT_COLS}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--ranking-context-cols "${RANKING_CONTEXT_COLS}")
  fi
  if [[ -f "${MMD_FLOW_V1}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--flow-mmd-csv "${MMD_FLOW_V1}")
  fi
  if [[ -f "${MMD_DINO_V1}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--dino-mmd-csv "${MMD_DINO_V1}")
  fi
  if [[ -f "${MMD_FEATURE_V1}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--feature-mmd-csv "${MMD_FEATURE_V1}")
  elif [[ -f "${MMD_DINO_V1}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--feature-mmd-csv "${MMD_DINO_V1}")
  fi
  if [[ "${JOINT_OOD_HOLDOUT}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--joint-ood-holdout)
  fi
  if [[ -n "${LINEAR_MODEL}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--linear-model "${LINEAR_MODEL}")
  fi
  if [[ -n "${PREDICTION_MODEL}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--prediction-model "${PREDICTION_MODEL}")
  fi
  if [[ -n "${RIDGE_ALPHA}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--ridge-alpha "${RIDGE_ALPHA}")
  fi
  if [[ "${PER_ENCODER}" -eq 0 ]]; then
    DENSITY_SCRIPT_ARGS+=(--no-per-encoder)
  fi
  if [[ "${PREDICTION_CLIP_SET}" -eq 1 ]]; then
    if [[ "${PREDICTION_CLIP}" -eq 1 ]]; then
      DENSITY_SCRIPT_ARGS+=(--prediction-clip)
    else
      DENSITY_SCRIPT_ARGS+=(--no-prediction-clip)
    fi
  fi
  if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--collapse-cv-cells)
  else
    DENSITY_SCRIPT_ARGS+=(--no-collapse-cv-cells)
  fi
  DENSITY_SCRIPT_ARGS+=(--fit-sample-weighting "${FIT_SAMPLE_WEIGHTING}")
  if [[ "${FIT_BALANCE_REAL_SYNTH}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--fit-balance-real-synth)
  else
    DENSITY_SCRIPT_ARGS+=(--no-fit-balance-real-synth)
  fi
  DENSITY_SCRIPT_ARGS+=(--overall-aggregation "${OVERALL_AGGREGATION}")
  if [[ "${CV_RESIDUALIZE_TARGET_BY_CONTEXT}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--cv-residualize-target-by-context)
  fi
  if [[ -n "${CV_RESIDUAL_CONTEXT_COLS}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--cv-residual-context-cols "${CV_RESIDUAL_CONTEXT_COLS}")
  fi
  DENSITY_SCRIPT_ARGS+=(--cv-residual-eval-space "${CV_RESIDUAL_EVAL_SPACE}")
  DENSITY_SCRIPT_ARGS+=(--cv-residual-target-transform "${CV_RESIDUAL_TARGET_TRANSFORM}")
  DENSITY_SCRIPT_ARGS+=(--cv-residual-target-std-eps "${CV_RESIDUAL_TARGET_STD_EPS}")
  if [[ "${CV_FEWSHOT_CONTEXT_CALIBRATION}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--cv-fewshot-context-calibration)
  fi
  if [[ -n "${CV_FEWSHOT_CONTEXT_CALIBRATION_COLS}" ]]; then
    DENSITY_SCRIPT_ARGS+=(--cv-fewshot-context-calibration-cols "${CV_FEWSHOT_CONTEXT_CALIBRATION_COLS}")
  fi
  DENSITY_SCRIPT_ARGS+=(--cv-fewshot-context-calibration-std-eps "${CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS}")
  DENSITY_SCRIPT_ARGS+=(--cv-fewshot-context-calibration-min-group-size "${CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE}")
  if [[ "${CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--cv-fewshot-context-calibration-backoff)
  else
    DENSITY_SCRIPT_ARGS+=(--no-cv-fewshot-context-calibration-backoff)
  fi
  DENSITY_SCRIPT_ARGS+=(--cv-fewshot-context-calibration-k "${CV_FEWSHOT_CONTEXT_CALIBRATION_K}")
  DENSITY_SCRIPT_ARGS+=(--cv-fewshot-context-calibration-seed "${CV_FEWSHOT_CONTEXT_CALIBRATION_SEED}")
  DENSITY_SCRIPT_ARGS+=(--cv-repeat-aggregation "${CV_REPEAT_AGGREGATION}")
  if [[ "${LOTO_SINGLE_PREDICTOR_BASELINES}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--loto-single-predictor-baselines)
  else
    DENSITY_SCRIPT_ARGS+=(--no-loto-single-predictor-baselines)
  fi
  if [[ "${JOINTOOD_SINGLE_PREDICTOR_BASELINES}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--jointood-single-predictor-baselines)
  else
    DENSITY_SCRIPT_ARGS+=(--no-jointood-single-predictor-baselines)
  fi
  if [[ "${PAIRWISE_ALL}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--pairwise-all)
  else
    DENSITY_SCRIPT_ARGS+=(--no-pairwise-all)
  fi
  if [[ "${DISABLE_MIXEDLM}" -eq 1 ]]; then
    DENSITY_SCRIPT_ARGS+=(--no-mixedlm)
  fi

  bash scripts/run_density_analysis_joint_latest.sh \
    "${DENSITY_SCRIPT_ARGS[@]}"

  RAW_TRAIN_PREDS="$(build_eps_pred_list "flow_train_to_eval_eps" "px")"
  RAW_EVAL_PREDS="$(build_eps_pred_list "flow_eval_to_train_eps" "px")"
  WEIGHTED_TRAIN_PREDS="$(build_eps_pred_list "flow_train_to_eval_eps" "px_weighted")"
  WEIGHTED_EVAL_PREDS="$(build_eps_pred_list "flow_eval_to_train_eps" "px_weighted")"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_train_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
    --no-flow-eps-predictors \
    --predictors "${RAW_TRAIN_PREDS}" \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eval_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
    --no-flow-eps-predictors \
    --predictors "${RAW_EVAL_PREDS}" \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  if [[ -n "${KL_FLOW_CSV}" && -f "${KL_FLOW_CSV}" ]]; then
    run_eval \
      --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_flow_kl_k${KL_K}" \
      --coverage-csv "${KL_FLOW_CSV}" \
      --no-flow-eps-predictors \
      --no-flow-eps-weighted-predictors \
      --predictors flow_train_to_eval_kl_div,flow_eval_to_train_kl_div \
      --no-logit-coverage \
      "${DENSITY_ARGS[@]}"
  else
    echo "Skipping flow KL analysis (missing: ${KL_FLOW_V2})"
  fi

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_single" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_train_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv" \
    --no-flow-eps-predictors \
    --predictors "${WEIGHTED_TRAIN_PREDS}" \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_eval_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv" \
    --no-flow-eps-predictors \
    --predictors "${WEIGHTED_EVAL_PREDS}" \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50_train_only" \
    --coverage-csv "${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q50.csv" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_eps_at50 \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50_eval_only" \
    --coverage-csv "${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q50.csv" \
    --no-flow-eps-predictors \
    --predictors flow_eval_to_train_eps_at50 \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95_train_only" \
    --coverage-csv "${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_auc \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95_eval_only" \
    --coverage-csv "${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv" \
    --no-flow-eps-predictors \
    --predictors flow_eval_to_train_auc \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold_train_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_mean_dist_over_radius_eval \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold_eval_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv" \
    --no-flow-eps-predictors \
    --predictors flow_eval_to_train_mean_dist_over_radius_train \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  if [[ -f "${DINO_COVERAGE_CSV}" ]]; then
    ensure_dino_rnorm_csv
  fi
  DINO_KL_AVAILABLE=0
  if [[ -f "${DINO_COVERAGE_RNORM}" && -f "${KL_DINO_CSV}" ]]; then
    DINO_COVERAGE_RNORM_KL="${OUTPUT_ROOT}/dino_coverage_rnorm_k5_with_kl_k${KL_K}.csv"
    merge_kl_into_coverage "${DINO_COVERAGE_RNORM}" "${KL_DINO_CSV}" "${DINO_COVERAGE_RNORM_KL}"
    if [[ -f "${DINO_COVERAGE_RNORM_KL}" ]]; then
      DINO_COVERAGE_RNORM="${DINO_COVERAGE_RNORM_KL}"
      DINO_KL_AVAILABLE=1
    fi
  fi
  if [[ -f "${DINO_COVERAGE_RNORM}" ]]; then
    DINO_ONLY_ARGS=(
      --snapshots-dir "${SNAPSHOT_DIRS[@]}"
      --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv"
      --coverage-dino-csv "${DINO_COVERAGE_RNORM}"
      --no-logit-coverage
      --no-flow-eps-predictors
      --no-flow-eps-weighted-predictors
      "${DENSITY_ARGS[@]}"
    )

    run_eval \
      "${DINO_ONLY_ARGS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_dino_rnorm_k5" \
      --predictors dino_eval_to_train_mean_dist,dino_train_to_eval_mean_dist

    run_eval \
      "${DINO_ONLY_ARGS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_dino_rnorm_k5_eval_only" \
      --predictors dino_eval_to_train_mean_dist

    run_eval \
      "${DINO_ONLY_ARGS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_dino_rnorm_k5_train_only" \
      --predictors dino_train_to_eval_mean_dist

    if [[ "${DINO_KL_AVAILABLE}" -eq 1 ]]; then
      run_eval \
        "${DINO_ONLY_ARGS[@]}" \
        --output-dir "${DENSITY_OUT}/leakage_free_dino_kl_k${KL_K}" \
        --predictors dino_eval_to_train_kl_div,dino_train_to_eval_kl_div
    fi

    run_eval \
      --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
      --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
      --coverage-dino-csv "${DINO_COVERAGE_RNORM}" \
      --output-dir "${DENSITY_OUT}/leakage_free_flow_eval_only_plus_dino_eval_only" \
      --no-logit-coverage \
      --no-flow-eps-predictors \
      --predictors "${RAW_EVAL_PREDS},dino_eval_to_train_mean_dist" \
      "${DENSITY_ARGS[@]}"
  else
    echo "Skipping dino analyses (missing: ${DINO_COVERAGE_RNORM})"
  fi

  if [[ -f "${HOF_COVERAGE_CSV}" ]]; then
    ensure_hof_rnorm_csv
  fi
  HOF_KL_AVAILABLE=0
  if [[ -f "${HOF_COVERAGE_RNORM}" && -f "${KL_HOF_CSV}" ]]; then
    HOF_COVERAGE_RNORM_KL="${OUTPUT_ROOT}/hof_coverage_rnorm_k5_with_kl_k${KL_K}.csv"
    merge_kl_into_coverage "${HOF_COVERAGE_RNORM}" "${KL_HOF_CSV}" "${HOF_COVERAGE_RNORM_KL}"
    if [[ -f "${HOF_COVERAGE_RNORM_KL}" ]]; then
      HOF_COVERAGE_RNORM="${HOF_COVERAGE_RNORM_KL}"
      HOF_KL_AVAILABLE=1
    fi
  fi
  if [[ -f "${HOF_COVERAGE_RNORM}" ]]; then
    HOF_ONLY_ARGS=(
      --snapshots-dir "${SNAPSHOT_DIRS[@]}"
      --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv"
      --coverage-hof-csv "${HOF_COVERAGE_RNORM}"
      --no-logit-coverage
      --no-flow-eps-predictors
      --no-flow-eps-weighted-predictors
      "${DENSITY_ARGS[@]}"
      --no-flow-density-interactions
    )

    run_eval \
      "${HOF_ONLY_ARGS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_hof_motion_k1" \
      --predictors hof_eval_to_train_mean_dist,hof_train_to_eval_mean_dist

    run_eval \
      "${HOF_ONLY_ARGS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_hof_motion_k1_eval_only" \
      --predictors hof_eval_to_train_mean_dist

    run_eval \
      "${HOF_ONLY_ARGS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_hof_motion_k1_train_only" \
      --predictors hof_train_to_eval_mean_dist

    if [[ "${NO_DENSITY_CONTROLS}" -eq 0 ]]; then
      run_eval \
        "${HOF_ONLY_ARGS[@]}" \
        --output-dir "${DENSITY_OUT}/leakage_free_hof_density_l2" \
        --predictors hof_density_l2

      run_eval \
        "${HOF_ONLY_ARGS[@]}" \
        --output-dir "${DENSITY_OUT}/leakage_free_hof_motion_k1_plus_density_l2" \
        --predictors hof_eval_to_train_mean_dist,hof_train_to_eval_mean_dist,hof_density_l2
    fi

    if [[ "${HOF_KL_AVAILABLE}" -eq 1 ]]; then
      run_eval \
        "${HOF_ONLY_ARGS[@]}" \
        --output-dir "${DENSITY_OUT}/leakage_free_hof_kl_k${KL_K}" \
        --predictors hof_eval_to_train_kl_div,hof_train_to_eval_kl_div
    fi
  else
    echo "Skipping HOF analyses (missing: ${HOF_COVERAGE_RNORM})"
  fi

  run_eval \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_target_fixed" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv" \
    --use-flow-eps-weighted-predictors \
    --no-flow-eps-predictors \
    --no-logit-coverage \
    --target auc_normalized \
    --flow-eps-values "${FLOW_EPS_VALUES}" \
    "${DENSITY_ARGS_BASE[@]}"
fi

if [[ "${SKIP_MMD}" -eq 0 ]]; then
  mkdir -p "${MMD_OUT}"
  if [[ -f "${MMD_FLOW_V2}" ]]; then
    python scripts/convert_mmd_v2_to_v1.py \
      --input "${MMD_FLOW_V2}" \
      --output "${MMD_FLOW_V1}"
  else
    echo "Skipping flow MMD conversion (missing: ${MMD_FLOW_V2})"
  fi
  if [[ -f "${MMD_DINO_V2}" ]]; then
    python scripts/convert_mmd_v2_to_v1.py \
      --input "${MMD_DINO_V2}" \
      --output "${MMD_DINO_V1}"
  else
    echo "Skipping dino MMD conversion (missing: ${MMD_DINO_V2})"
  fi

  BASE_ARGS=(
    --snapshots-dir "${SNAPSHOT_DIRS[@]}"
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv"
    --no-logit-coverage
    --flow-eps-values "${FLOW_EPS_VALUES}"
    "${DENSITY_ARGS[@]}"
  )
  MMD_ONLY_ARGS=(
    --snapshots-dir "${SNAPSHOT_DIRS[@]}"
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv"
    --no-logit-coverage
    --flow-eps-values "${FLOW_EPS_VALUES}"
    "${DENSITY_ARGS[@]}"
    --no-flow-eps-predictors
    --no-flow-eps-weighted-predictors
  )
  if [[ -f "${MMD_FLOW_V1}" ]]; then
    BASE_ARGS+=(--flow-mmd-csv "${MMD_FLOW_V1}")
    MMD_ONLY_ARGS+=(--flow-mmd-csv "${MMD_FLOW_V1}")
  fi
  if [[ -f "${MMD_DINO_V1}" ]]; then
    BASE_ARGS+=(--dino-mmd-csv "${MMD_DINO_V1}")
    MMD_ONLY_ARGS+=(--dino-mmd-csv "${MMD_DINO_V1}")
  fi

  HAS_FLOW_MMD=0
  HAS_DINO_MMD=0
  if [[ -f "${MMD_FLOW_V1}" ]]; then
    HAS_FLOW_MMD=1
  fi
  if [[ -f "${MMD_DINO_V1}" ]]; then
    HAS_DINO_MMD=1
  fi

  if [[ "${HAS_FLOW_MMD}" -eq 1 || "${HAS_DINO_MMD}" -eq 1 ]]; then
    if [[ "${HAS_FLOW_MMD}" -eq 1 ]]; then
      run_eval \
        "${MMD_ONLY_ARGS[@]}" \
        --output-dir "${OUTPUT_ROOT}/mmd_flow_only" \
        --predictors flow_mmd
    fi

    if [[ "${HAS_DINO_MMD}" -eq 1 ]]; then
      run_eval \
        "${MMD_ONLY_ARGS[@]}" \
        --output-dir "${OUTPUT_ROOT}/mmd_dino_only" \
        --predictors dino_mmd
    fi

    if [[ "${HAS_FLOW_MMD}" -eq 1 && "${HAS_DINO_MMD}" -eq 1 ]]; then
      run_eval \
        "${MMD_ONLY_ARGS[@]}" \
        --output-dir "${OUTPUT_ROOT}/mmd_only" \
        --predictors flow_mmd,dino_mmd

      run_eval \
        "${BASE_ARGS[@]}" \
        --output-dir "${OUTPUT_ROOT}/asym_and_mmd" \
        --use-flow-eps-predictors \
        --predictors flow_mmd,dino_mmd
    else
      echo "Skipping combined mmd_only/asym_and_mmd (requires both flow and dino MMD CSVs)."
    fi
  else
    echo "Skipping MMD-based analyses (no converted MMD CSVs available)."
  fi
fi

COMBO_METHODS_FILE="${OUTPUT_ROOT}/combo_methods.txt"
: > "${COMBO_METHODS_FILE}"

if [[ "${SKIP_COMBOS}" -eq 0 ]]; then
  mkdir -p "${DENSITY_OUT}"
  if [[ -f "${DINO_COVERAGE_CSV}" ]]; then
    ensure_dino_rnorm_csv
  fi
  DINO_KL_AVAILABLE=0
  if [[ -f "${DINO_COVERAGE_RNORM}" && -f "${KL_DINO_CSV}" ]]; then
    DINO_COVERAGE_RNORM_KL="${OUTPUT_ROOT}/dino_coverage_rnorm_k5_with_kl_k${KL_K}.csv"
    merge_kl_into_coverage "${DINO_COVERAGE_RNORM}" "${KL_DINO_CSV}" "${DINO_COVERAGE_RNORM_KL}"
    if [[ -f "${DINO_COVERAGE_RNORM_KL}" ]]; then
      DINO_COVERAGE_RNORM="${DINO_COVERAGE_RNORM_KL}"
      DINO_KL_AVAILABLE=1
    fi
  fi

  DINO_AVAILABLE=0
  if [[ -f "${DINO_COVERAGE_RNORM}" ]]; then
    DINO_AVAILABLE=1
  fi
  HOF_AVAILABLE=0
  if [[ -f "${HOF_COVERAGE_RNORM}" ]]; then
    HOF_AVAILABLE=1
  fi

  MMD_AVAILABLE=0
  if [[ -f "${MMD_FLOW_V1}" || -f "${MMD_DINO_V1}" ]]; then
    MMD_AVAILABLE=1
  fi

  COMBO_BASE_ARGS=(
    --snapshots-dir "${SNAPSHOT_DIRS[@]}"
    --no-logit-coverage
    --flow-eps-values "${FLOW_EPS_VALUES}"
    "${DENSITY_ARGS[@]}"
  )
  if [[ -f "${MMD_FLOW_V1}" ]]; then
    COMBO_BASE_ARGS+=(--flow-mmd-csv "${MMD_FLOW_V1}")
  fi
  if [[ -f "${MMD_DINO_V1}" ]]; then
    COMBO_BASE_ARGS+=(--dino-mmd-csv "${MMD_DINO_V1}")
  fi
  if [[ "${DINO_AVAILABLE}" -eq 1 ]]; then
    COMBO_BASE_ARGS+=(--coverage-dino-csv "${DINO_COVERAGE_RNORM}")
  fi
  if [[ "${HOF_AVAILABLE}" -eq 1 ]]; then
    COMBO_BASE_ARGS+=(--coverage-hof-csv "${HOF_COVERAGE_RNORM}")
  fi

  join_preds() {
    local out=""
    for part in "$@"; do
      if [[ -n "${part}" ]]; then
        if [[ -n "${out}" ]]; then
          out="${out},${part}"
        else
          out="${part}"
        fi
      fi
    done
    echo "${out}"
  }

  run_combo() {
    local name="$1"
    local coverage_csv="$2"
    local flow_args="$3"
    local predictors="$4"
    local family="$5"
    local symmetry="$6"
    local notes="$7"
    local out_dir="${DENSITY_OUT}/leakage_free_${name}"

    if [[ ! -f "${coverage_csv}" ]]; then
      echo "Skipping combo ${name} (missing coverage csv: ${coverage_csv})"
      return 0
    fi

    if [[ -d "${out_dir}" && "${FORCE_RERUN}" -eq 0 ]]; then
      echo "${name}|${out_dir}|${family}|${symmetry}|${notes}" >> "${COMBO_METHODS_FILE}"
      if [[ "${PAIRWISE_ALL}" -eq 1 && -d "${out_dir}_pairwise" ]]; then
        echo "${name}_pairwise|${out_dir}_pairwise|${family}|${symmetry}|${notes},model=pairwise_rank" >> "${COMBO_METHODS_FILE}"
      fi
      return 0
    fi
    if [[ -d "${out_dir}" && "${FORCE_RERUN}" -eq 1 ]]; then
      echo "RERUN existing combo: ${out_dir}"
    fi

    local flow_arg_list=()
    if [[ -n "${flow_args}" ]]; then
      read -r -a flow_arg_list <<< "${flow_args}"
    fi

    local args=(
      "${COMBO_BASE_ARGS[@]}"
      --coverage-csv "${coverage_csv}"
      "${flow_arg_list[@]}"
      --output-dir "${out_dir}"
    )
    if [[ -n "${predictors}" ]]; then
      args+=(--predictors "${predictors}")
    fi

    run_eval "${args[@]}"
    echo "${name}|${out_dir}|${family}|${symmetry}|${notes}" >> "${COMBO_METHODS_FILE}"
    if [[ "${PAIRWISE_ALL}" -eq 1 && -d "${out_dir}_pairwise" ]]; then
      echo "${name}_pairwise|${out_dir}_pairwise|${family}|${symmetry}|${notes},model=pairwise_rank" >> "${COMBO_METHODS_FILE}"
    fi
  }

  FLOW_VARIANTS=(
    "flow_eps_raw|${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv|--use-flow-eps-predictors|"
    "flow_eps_raw_single|${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv|--no-flow-eps-predictors|flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px"
    "flow_eps_raw_eps_at50|${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q50.csv|--no-flow-eps-predictors|flow_train_to_eval_eps_at50,flow_eval_to_train_eps_at50"
    "flow_eps_raw_auc_at95|${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv|--no-flow-eps-predictors|flow_train_to_eval_auc,flow_eval_to_train_auc"
    "flow_kmeans_weighted|${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv|--use-flow-eps-weighted-predictors --no-flow-eps-predictors|"
    "flow_kmeans_manifold|${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv|--no-flow-eps-predictors|flow_train_to_eval_mean_dist_over_radius_eval,flow_eval_to_train_mean_dist_over_radius_train"
  )
  if [[ -n "${KL_FLOW_CSV}" && -f "${KL_FLOW_CSV}" ]]; then
    FLOW_VARIANTS+=(
      "flow_kl_k${KL_K}|${KL_FLOW_CSV}|--no-flow-eps-predictors --no-flow-eps-weighted-predictors|flow_train_to_eval_kl_div,flow_eval_to_train_kl_div"
    )
  fi

  DINO_VARIANTS=(
    "none||"
    "dino_rnorm_k5|dino_eval_to_train_mean_dist,dino_train_to_eval_mean_dist|"
    "dino_rnorm_k5_eval_only|dino_eval_to_train_mean_dist|dir=eval_only"
    "dino_rnorm_k5_train_only|dino_train_to_eval_mean_dist|dir=train_only"
  )
  if [[ "${DINO_KL_AVAILABLE}" -eq 1 ]]; then
    DINO_VARIANTS+=(
      "dino_kl_k${KL_K}|dino_eval_to_train_kl_div,dino_train_to_eval_kl_div|"
    )
  fi
  if [[ "${DINO_AVAILABLE}" -eq 0 ]]; then
    DINO_VARIANTS=("none||")
  fi

  HOF_VARIANTS=(
    "none||"
    "hof_motion_k1|hof_eval_to_train_mean_dist,hof_train_to_eval_mean_dist|"
    "hof_motion_k1_eval_only|hof_eval_to_train_mean_dist|dir=eval_only"
    "hof_motion_k1_train_only|hof_train_to_eval_mean_dist|dir=train_only"
  )
  if [[ "${NO_DENSITY_CONTROLS}" -eq 0 ]]; then
    HOF_VARIANTS+=(
      "hof_density_l2|hof_density_l2|"
      "hof_motion_k1_plus_density_l2|hof_eval_to_train_mean_dist,hof_train_to_eval_mean_dist,hof_density_l2|"
    )
  fi
  if [[ "${HOF_KL_AVAILABLE}" -eq 1 ]]; then
    HOF_VARIANTS+=(
      "hof_kl_k${KL_K}|hof_eval_to_train_kl_div,hof_train_to_eval_kl_div|"
    )
  fi
  if [[ "${HOF_AVAILABLE}" -eq 0 ]]; then
    HOF_VARIANTS=("none||")
  fi

  MMD_VARIANTS=(
    "none||"
    "mmd|flow_mmd,dino_mmd|"
  )
  if [[ "${MMD_AVAILABLE}" -eq 0 ]]; then
    MMD_VARIANTS=("none||")
  fi

  for flow_item in "${FLOW_VARIANTS[@]}"; do
    IFS='|' read -r flow_name flow_csv flow_args flow_preds <<< "${flow_item}"
    for dino_item in "${DINO_VARIANTS[@]}"; do
      IFS='|' read -r dino_name dino_preds dino_note <<< "${dino_item}"
      for hof_item in "${HOF_VARIANTS[@]}"; do
        IFS='|' read -r hof_name hof_preds hof_note <<< "${hof_item}"
        for mmd_item in "${MMD_VARIANTS[@]}"; do
          IFS='|' read -r mmd_name mmd_preds _ <<< "${mmd_item}"

          if [[ "${dino_name}" == "none" && "${hof_name}" == "none" && "${mmd_name}" == "none" ]]; then
            continue
          fi

          combo_preds="$(join_preds "${flow_preds}" "${dino_preds}" "${hof_preds}" "${mmd_preds}")"
          if [[ -z "${combo_preds}" ]]; then
            continue
          fi

          local_name="combo_${flow_name}"
          local_notes="combo:flow=${flow_name}"
          local_family="flow"
          local_symmetry="asym"

          if [[ "${dino_name}" != "none" ]]; then
            local_name+="__${dino_name}"
            local_notes+=",dino=${dino_name}"
            local_family="mixed"
          fi
          if [[ "${hof_name}" != "none" ]]; then
            local_name+="__${hof_name}"
            local_notes+=",hof=${hof_name}"
            local_family="mixed"
          fi
          if [[ "${mmd_name}" != "none" ]]; then
            local_name+="__${mmd_name}"
            local_notes+=",mmd=on"
            local_family="mixed"
            local_symmetry="mixed"
          fi
          if [[ -n "${dino_note}" ]]; then
            local_notes+=",${dino_note}"
          fi
          if [[ -n "${hof_note}" ]]; then
            local_notes+=",${hof_note}"
          fi

          run_combo "${local_name}" "${flow_csv}" "${flow_args}" "${combo_preds}" "${local_family}" "${local_symmetry}" "${local_notes}"
        done
      done
    done
  done
fi

MANIFEST_PATH="${OUTPUT_ROOT}/method_summary_manifest.yaml"
{
  echo "methods:"
  add_method() {
    local name="$1"
    local path="$2"
    local family="$3"
    local symmetry="$4"
    local notes="${5:-}"
    if [[ -d "${path}" ]]; then
      echo "  - name: ${name}"
      echo "    path: ${path}"
      echo "    family: ${family}"
      echo "    symmetry: ${symmetry}"
      if [[ -n "${notes}" ]]; then
        echo "    notes: ${notes}"
      fi
      if [[ "${PAIRWISE_ALL}" -eq 1 && -d "${path}_pairwise" ]]; then
        echo "  - name: ${name}_pairwise"
        echo "    path: ${path}_pairwise"
        echo "    family: ${family}"
        echo "    symmetry: ${symmetry}"
        if [[ -n "${notes}" ]]; then
          echo "    notes: ${notes},model=pairwise_rank"
        else
          echo "    notes: model=pairwise_rank"
        fi
      fi
    else
      echo "Skipping manifest entry (missing dir): ${path}" >&2
    fi
  }

  add_method "flow_eps_raw_joint" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint" "flow" "asym"
  add_method "flow_eps_raw_single" "${DENSITY_OUT}/leakage_free_flow_eps_raw_single" "flow" "asym" "single_eps"
  add_method "flow_eps_raw_joint_train_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_eps_raw_joint_eval_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_eps_raw_joint_auc_at95" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95" "flow" "asym"
  add_method "flow_eps_raw_joint_auc_at95_train_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_eps_raw_joint_auc_at95_eval_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_eps_raw_joint_eps_at50" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50" "flow" "asym"
  add_method "flow_eps_raw_joint_eps_at50_train_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_eps_raw_joint_eps_at50_eval_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_kmeans_weighted_all" "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all" "flow" "asym"
  add_method "flow_kmeans_weighted_all_train_only" "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_kmeans_weighted_all_eval_only" "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_kmeans_weighted_all_target_fixed" "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_target_fixed" "flow" "asym" "target=auc_normalized"
  add_method "flow_kmeans_manifold" "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold" "flow" "asym"
  add_method "flow_kmeans_manifold_train_only" "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_kmeans_manifold_eval_only" "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_kl_k${KL_K}" "${DENSITY_OUT}/leakage_free_flow_kl_k${KL_K}" "flow" "asym"
  add_method "dino_rnorm_k5" "${DENSITY_OUT}/leakage_free_dino_rnorm_k5" "appearance" "asym"
  add_method "dino_rnorm_k5_train_only" "${DENSITY_OUT}/leakage_free_dino_rnorm_k5_train_only" "appearance" "asym" "dir=train_only"
  add_method "dino_rnorm_k5_eval_only" "${DENSITY_OUT}/leakage_free_dino_rnorm_k5_eval_only" "appearance" "asym" "dir=eval_only"
  add_method "dino_kl_k${KL_K}" "${DENSITY_OUT}/leakage_free_dino_kl_k${KL_K}" "appearance" "asym"
  add_method "hof_motion_k1" "${DENSITY_OUT}/leakage_free_hof_motion_k1" "motion" "asym"
  add_method "hof_motion_k1_train_only" "${DENSITY_OUT}/leakage_free_hof_motion_k1_train_only" "motion" "asym" "dir=train_only"
  add_method "hof_motion_k1_eval_only" "${DENSITY_OUT}/leakage_free_hof_motion_k1_eval_only" "motion" "asym" "dir=eval_only"
  if [[ "${NO_DENSITY_CONTROLS}" -eq 0 ]]; then
    add_method "hof_density_l2" "${DENSITY_OUT}/leakage_free_hof_density_l2" "motion" "asym" "density_only"
    add_method "hof_motion_k1_plus_density_l2" "${DENSITY_OUT}/leakage_free_hof_motion_k1_plus_density_l2" "motion" "asym" "motion_plus_density"
  fi
  add_method "hof_kl_k${KL_K}" "${DENSITY_OUT}/leakage_free_hof_kl_k${KL_K}" "motion" "asym"
  add_method "flow_eval_only_plus_dino_eval_only" "${DENSITY_OUT}/leakage_free_flow_eval_only_plus_dino_eval_only" "mixed" "asym" "dir=eval_only"
  add_method "mmd_flow_only" "${OUTPUT_ROOT}/mmd_flow_only" "mmd" "sym" "source=flow"
  add_method "mmd_dino_only" "${OUTPUT_ROOT}/mmd_dino_only" "mmd" "sym" "source=dino"
  add_method "mmd_only" "${OUTPUT_ROOT}/mmd_only" "mmd" "sym" "source=flow+dino"
  add_method "asym_and_mmd" "${OUTPUT_ROOT}/asym_and_mmd" "mixed" "mixed"
  if [[ -s "${COMBO_METHODS_FILE}" ]]; then
    while IFS='|' read -r name path family symmetry notes; do
      add_method "${name}" "${path}" "${family}" "${symmetry}" "${notes}"
    done < "${COMBO_METHODS_FILE}"
  fi
} > "${MANIFEST_PATH}"

if [[ "${SKIP_SUMMARY}" -eq 0 ]]; then
  python scripts/compile_method_summary.py \
    --manifest "${MANIFEST_PATH}" \
    --output "${OUTPUT_ROOT}/method_summary.csv" \
    --output-md "${OUTPUT_ROOT}/method_summary.md"

  python scripts/build_hypothesis_tables.py \
    --summary "${OUTPUT_ROOT}/method_summary.csv" \
    --output-dir "${OUTPUT_ROOT}/hypothesis_tables"

  python scripts/build_final_tables.py \
    --summary "${OUTPUT_ROOT}/method_summary.csv" \
    --output-dir "${OUTPUT_ROOT}/final_tables"
fi

echo ""
echo "Done."
echo "Output root: ${OUTPUT_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"
