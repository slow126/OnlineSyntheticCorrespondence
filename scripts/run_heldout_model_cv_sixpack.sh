#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "This script must be run with bash." >&2
  exit 2
fi

OUT_DIR="analysis_comprehensive_runs/heldout_model_cv_v1"
RUN_ROOTS=""
TOP_N_PER_LANE=2
MIN_SIGNAL_K=1
MAX_SIGNAL_K=8
MAX_HARD_FOLDS=150
HOLDOUT_TRAIN_K=3
LANES="motion_only,appearance_only,hybrid"
HEADS="ols,ridge,pairwise_rank"
PROTOCOLS="model_benchmark_trainset_disjoint"
ROW_SOURCE="raw"
OPTION_COL="train_dataset"
MODEL_GROUP_COL="model_family_encoder"
PAIRWISE_GROUP_COLS="benchmark,model_family_encoder"
RANK_GROUPING="fold_benchmark"
RANK_CONTEXT_COLS="model_family_encoder"
CV_RESIDUALIZE_TARGET_BY_CONTEXT=1
CV_RESIDUAL_CONTEXT_COLS="benchmark,model_family_encoder"
CV_RESIDUAL_TARGET_TRANSFORM="zscore"
CV_RESIDUAL_TARGET_STD_EPS="1e-9"
CV_RESIDUAL_EVAL_SPACE="residual"
CV_FEWSHOT_CONTEXT_CALIBRATION=0
CV_FEWSHOT_CONTEXT_CALIBRATION_COLS="benchmark,model_family_encoder"
CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS="1e-9"
CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE=2
CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=1
CV_FEWSHOT_CONTEXT_CALIBRATION_K=0
CV_FEWSHOT_CONTEXT_CALIBRATION_SEED=0
INCLUDE_PAIRWISE_CANDIDATES=0
SAVE_PRED_ROWS=0
SAVE_FOLD_ROWS=0
PERMUTATION_SAMPLES=0
PERMUTATION_MODE="benchmark"
CLAIM_MAE_TIE_EPSILON=0.1
CANDIDATE_DEDUP_PRIMARY_METRIC="jointood_rank_spearman"
CANDIDATE_DEDUP_TIEBREAK_METRIC="jointood_rank_pairwise_cindex"
CANDIDATE_SELECTION_PRIMARY_METRIC="jointood_rank_spearman"
CANDIDATE_SELECTION_TIEBREAK_METRIC="jointood_rank_pairwise_cindex"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_heldout_model_cv_sixpack.sh [options]

Options:
  --output-dir PATH       Output directory
  --run-roots CSV         Comma-separated run roots (default: build_heldout_model_cv.py defaults)
  --top-n-per-lane N      Candidate count per lane (default: 2)
  --min-signal-k N        Minimum signal predictor count (default: 1)
  --max-signal-k N        Maximum signal predictor count (default: 8)
  --max-hard-folds N      Max folds for hard protocols (including model_benchmark_trainset_disjoint; default: 150, <=0 means all)
  --holdout-train-k N     Held-out train-dataset count for model_benchmark_trainset_disjoint (default: 3)
  --lanes CSV             Lane filter (default: motion_only,appearance_only,hybrid)
  --heads CSV             Head list (default: ols,ridge,pairwise_rank)
  --protocols CSV         Protocol list (default: model_benchmark_trainset_disjoint; also supports model_only,model_train_benchmark,model_benchmark,model_train_benchmark_disjoint)
  --row-source MODE       raw|prediction (default: raw)
  --option-col NAME       Option grouping column for rank metrics (default: train_dataset)
  --model-group-col NAME  Model grouping column for held-out model folds (default: model_family_encoder)
  --pairwise-group-cols CSV Pairwise training groups (default: benchmark,model_family_encoder)
  --rank-grouping MODE    fold_benchmark|benchmark (default: fold_benchmark)
  --rank-context-cols CSV Extra rank/permutation grouping columns (default: model_family_encoder)
  --cv-residualize-target-by-context Enable fold-safe target residualization by context (default: enabled)
  --no-cv-residualize-target-by-context Disable fold-safe target residualization by context
  --cv-residual-context-cols CSV Context columns for residualization (default: benchmark,model_family_encoder)
  --cv-residual-target-transform MODE residual|zscore (default: zscore)
  --cv-residual-target-std-eps X Std epsilon floor for zscore residualization (default: 1e-9)
  --cv-residual-eval-space MODE residual|absolute (default: residual)
  --cv-fewshot-context-calibration Enable few-shot context calibration on heldout folds
  --no-cv-fewshot-context-calibration Disable few-shot context calibration (default)
  --cv-fewshot-context-calibration-cols CSV Context columns for few-shot calibration (default: benchmark,model_family_encoder)
  --cv-fewshot-context-calibration-std-eps X Std epsilon for few-shot calibration (default: 1e-9)
  --cv-fewshot-context-calibration-min-group-size N Minimum calibration group size (default: 2)
  --cv-fewshot-context-calibration-backoff Enable hierarchical context backoff (default: enabled)
  --no-cv-fewshot-context-calibration-backoff Disable hierarchical context backoff
  --cv-fewshot-context-calibration-k N Calibration shots per context group (default: 0)
  --cv-fewshot-context-calibration-seed N Random seed base for few-shot row sampling (default: 0)
  --include-pairwise-candidates Include *_pairwise methods in candidate pool
  --permutation-samples N Within-fold target-shuffle permutations per run (default: 0, disabled)
  --permutation-mode MODE benchmark|fold|global (default: benchmark)
  --candidate-dedup-primary-metric NAME      Metric for per-signature variant dedup (default: jointood_rank_spearman)
  --candidate-dedup-tiebreak-metric NAME     Tie-break metric for dedup (default: jointood_rank_pairwise_cindex)
  --candidate-selection-primary-metric NAME  Metric for lane-wise candidate selection (default: jointood_rank_spearman)
  --candidate-selection-tiebreak-metric NAME Tie-break metric for selection (default: jointood_rank_pairwise_cindex)
  --claim-mae-tie-epsilon X  Near-tie threshold for collapsed claim MAE deltas (default: 0.1)
  --save-pred-rows        Save OOF prediction rows CSV
  --save-fold-rows        Save per-fold metrics CSV
  -h, --help              Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUT_DIR="$2"; shift 2 ;;
    --run-roots)
      RUN_ROOTS="$2"; shift 2 ;;
    --top-n-per-lane)
      TOP_N_PER_LANE="$2"; shift 2 ;;
    --min-signal-k)
      MIN_SIGNAL_K="$2"; shift 2 ;;
    --max-signal-k)
      MAX_SIGNAL_K="$2"; shift 2 ;;
    --max-hard-folds)
      MAX_HARD_FOLDS="$2"; shift 2 ;;
    --holdout-train-k)
      HOLDOUT_TRAIN_K="$2"; shift 2 ;;
    --lanes)
      LANES="$2"; shift 2 ;;
    --heads)
      HEADS="$2"; shift 2 ;;
    --protocols)
      PROTOCOLS="$2"; shift 2 ;;
    --row-source)
      ROW_SOURCE="$2"; shift 2 ;;
    --option-col)
      OPTION_COL="$2"; shift 2 ;;
    --model-group-col)
      MODEL_GROUP_COL="$2"; shift 2 ;;
    --pairwise-group-cols)
      PAIRWISE_GROUP_COLS="$2"; shift 2 ;;
    --rank-grouping)
      RANK_GROUPING="$2"; shift 2 ;;
    --rank-context-cols)
      RANK_CONTEXT_COLS="$2"; shift 2 ;;
    --cv-residualize-target-by-context)
      CV_RESIDUALIZE_TARGET_BY_CONTEXT=1; shift 1 ;;
    --no-cv-residualize-target-by-context)
      CV_RESIDUALIZE_TARGET_BY_CONTEXT=0; shift 1 ;;
    --cv-residual-context-cols)
      CV_RESIDUAL_CONTEXT_COLS="$2"; shift 2 ;;
    --cv-residual-target-transform)
      CV_RESIDUAL_TARGET_TRANSFORM="$2"; shift 2 ;;
    --cv-residual-target-std-eps)
      CV_RESIDUAL_TARGET_STD_EPS="$2"; shift 2 ;;
    --cv-residual-eval-space)
      CV_RESIDUAL_EVAL_SPACE="$2"; shift 2 ;;
    --cv-fewshot-context-calibration)
      CV_FEWSHOT_CONTEXT_CALIBRATION=1; shift 1 ;;
    --no-cv-fewshot-context-calibration)
      CV_FEWSHOT_CONTEXT_CALIBRATION=0; shift 1 ;;
    --cv-fewshot-context-calibration-cols)
      CV_FEWSHOT_CONTEXT_CALIBRATION_COLS="$2"; shift 2 ;;
    --cv-fewshot-context-calibration-std-eps)
      CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS="$2"; shift 2 ;;
    --cv-fewshot-context-calibration-min-group-size)
      CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE="$2"; shift 2 ;;
    --cv-fewshot-context-calibration-backoff)
      CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=1; shift 1 ;;
    --no-cv-fewshot-context-calibration-backoff)
      CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=0; shift 1 ;;
    --cv-fewshot-context-calibration-k)
      CV_FEWSHOT_CONTEXT_CALIBRATION_K="$2"; shift 2 ;;
    --cv-fewshot-context-calibration-seed)
      CV_FEWSHOT_CONTEXT_CALIBRATION_SEED="$2"; shift 2 ;;
    --include-pairwise-candidates)
      INCLUDE_PAIRWISE_CANDIDATES=1; shift 1 ;;
    --permutation-samples)
      PERMUTATION_SAMPLES="$2"; shift 2 ;;
    --permutation-mode)
      PERMUTATION_MODE="$2"; shift 2 ;;
    --candidate-dedup-primary-metric)
      CANDIDATE_DEDUP_PRIMARY_METRIC="$2"; shift 2 ;;
    --candidate-dedup-tiebreak-metric)
      CANDIDATE_DEDUP_TIEBREAK_METRIC="$2"; shift 2 ;;
    --candidate-selection-primary-metric)
      CANDIDATE_SELECTION_PRIMARY_METRIC="$2"; shift 2 ;;
    --candidate-selection-tiebreak-metric)
      CANDIDATE_SELECTION_TIEBREAK_METRIC="$2"; shift 2 ;;
    --claim-mae-tie-epsilon)
      CLAIM_MAE_TIE_EPSILON="$2"; shift 2 ;;
    --save-pred-rows)
      SAVE_PRED_ROWS=1; shift 1 ;;
    --save-fold-rows)
      SAVE_FOLD_ROWS=1; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

args=(
  --output-dir "$OUT_DIR"
  --run-roots "$RUN_ROOTS"
  --top-n-per-lane "$TOP_N_PER_LANE"
  --min-signal-k "$MIN_SIGNAL_K"
  --max-signal-k "$MAX_SIGNAL_K"
  --max-hard-folds "$MAX_HARD_FOLDS"
  --holdout-train-k "$HOLDOUT_TRAIN_K"
  --lanes "$LANES"
  --heads "$HEADS"
  --protocols "$PROTOCOLS"
  --row-source "$ROW_SOURCE"
  --option-col "$OPTION_COL"
  --model-group-col "$MODEL_GROUP_COL"
  --pairwise-group-cols "$PAIRWISE_GROUP_COLS"
  --rank-grouping "$RANK_GROUPING"
  --rank-context-cols "$RANK_CONTEXT_COLS"
  --cv-residual-context-cols "$CV_RESIDUAL_CONTEXT_COLS"
  --cv-residual-target-transform "$CV_RESIDUAL_TARGET_TRANSFORM"
  --cv-residual-target-std-eps "$CV_RESIDUAL_TARGET_STD_EPS"
  --cv-residual-eval-space "$CV_RESIDUAL_EVAL_SPACE"
  --cv-fewshot-context-calibration-cols "$CV_FEWSHOT_CONTEXT_CALIBRATION_COLS"
  --cv-fewshot-context-calibration-std-eps "$CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS"
  --cv-fewshot-context-calibration-min-group-size "$CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE"
  --cv-fewshot-context-calibration-k "$CV_FEWSHOT_CONTEXT_CALIBRATION_K"
  --cv-fewshot-context-calibration-seed "$CV_FEWSHOT_CONTEXT_CALIBRATION_SEED"
  --permutation-samples "$PERMUTATION_SAMPLES"
  --permutation-mode "$PERMUTATION_MODE"
  --candidate-dedup-primary-metric "$CANDIDATE_DEDUP_PRIMARY_METRIC"
  --candidate-dedup-tiebreak-metric "$CANDIDATE_DEDUP_TIEBREAK_METRIC"
  --candidate-selection-primary-metric "$CANDIDATE_SELECTION_PRIMARY_METRIC"
  --candidate-selection-tiebreak-metric "$CANDIDATE_SELECTION_TIEBREAK_METRIC"
  --claim-mae-tie-epsilon "$CLAIM_MAE_TIE_EPSILON"
)

if [[ "$INCLUDE_PAIRWISE_CANDIDATES" -eq 1 ]]; then
  args+=(--include-pairwise-candidates)
fi
if [[ "$CV_RESIDUALIZE_TARGET_BY_CONTEXT" -eq 1 ]]; then
  args+=(--cv-residualize-target-by-context)
else
  args+=(--no-cv-residualize-target-by-context)
fi
if [[ "$CV_FEWSHOT_CONTEXT_CALIBRATION" -eq 1 ]]; then
  args+=(--cv-fewshot-context-calibration)
else
  args+=(--no-cv-fewshot-context-calibration)
fi
if [[ "$CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF" -eq 1 ]]; then
  args+=(--cv-fewshot-context-calibration-backoff)
else
  args+=(--no-cv-fewshot-context-calibration-backoff)
fi
if [[ "$SAVE_PRED_ROWS" -eq 1 ]]; then
  args+=(--save-pred-rows)
fi
if [[ "$SAVE_FOLD_ROWS" -eq 1 ]]; then
  args+=(--save-fold-rows)
fi

python scripts/build_heldout_model_cv.py "${args[@]}"

echo "Done: $OUT_DIR"
