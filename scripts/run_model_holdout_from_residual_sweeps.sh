#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "This script must be run with bash." >&2
  exit 2
fi

ZERO_SHOT_ROOT="analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3"
FEWSHOT_ROOT="analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_fewshotcal_k1_v3"
OUTPUT_ROOT="analysis_comprehensive_runs/heldout_model_cv_residual_no_family_no_density_v1"

RUN_ROOT=""
RUN_LABEL="interactions"
RUN_MODE="fewshot"

PROTOCOLS="model_only"
HEADS="ridge"
LANES="motion_only,appearance_only,hybrid"
TOP_N_PER_LANE=2
MIN_SIGNAL_K=1
MAX_SIGNAL_K=8
MAX_HARD_FOLDS=150
HOLDOUT_TRAIN_K=3
PERMUTATION_SAMPLES=0
PERMUTATION_MODE="benchmark"
FEWSHOT_K=1
FEWSHOT_MIN_GROUP_SIZE=2
FEWSHOT_BACKOFF=1
FEWSHOT_SEED=0

RUN_PLOTS=1
PLOT_BEST_CV_METRIC="loto_pair_win"
PLOT_BUCKET_FILTER=""
PLOT_TOP_K=8
PLOT_CONTEXT_TARGET_TRANSFORM="zscore"
PLOT_CONTEXT_TARGET_PLOT_SPACE="model_space"
PLOT_PREDICTION_TRANSFORM="none"
PLOT_HELDOUT_PROTOCOLS="lobo,loto,jointood,model"
PLOT_HELDOUT_PLOT_SPACES="model_space"
PLOT_HELDOUT_COLOR_BY="train_dataset"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_model_holdout_from_residual_sweeps.sh [options]

Default mode:
  Runs heldout model CV for both zero-shot and few-shot sweep roots.

Single-root mode:
  Use --run-root to evaluate one root (supports passing either <root> or <root>/density_joint).

Options:
  --run-root PATH         Single-root mode input (root or density_joint path)
  --run-label TEXT        Single-root mode label (default: interactions)
  --run-mode MODE         Single-root mode: zeroshot|fewshot (default: fewshot)
  --zero-shot-root PATH   Zero-shot sweep root
  --fewshot-root PATH     Few-shot sweep root
  --output-root PATH      Parent output directory for heldout results
  --protocols CSV         model_only,model_train_benchmark,model_benchmark,
                          model_train_benchmark_disjoint,model_benchmark_trainset_disjoint
  --heads CSV             Head list (default: ridge)
  --lanes CSV             Lane filter (default: motion_only,appearance_only,hybrid)
  --top-n-per-lane N      Candidate count per lane (default: 2)
  --min-signal-k N        Minimum signal predictor count (default: 1)
  --max-signal-k N        Maximum signal predictor count (default: 8)
  --max-hard-folds N      Max folds for hard protocols (default: 150)
  --holdout-train-k N     Held-out train-dataset count (default: 3)
  --permutation-samples N Target-shuffle permutations per run (default: 0)
  --permutation-mode MODE benchmark|fold|global (default: benchmark)
  --fewshot-k N           Calibration shots per context (default: 1)
  --fewshot-min-group-size N  Minimum calibration group size (default: 2)
  --fewshot-backoff       Enable hierarchical context backoff (default: enabled)
  --no-fewshot-backoff    Disable hierarchical context backoff
  --fewshot-seed N        Few-shot calibration seed base (default: 0)
  --no-plots              Skip heldout visualization plotting stage
  --plot-best-cv-metric KEY
                          Metric key for plot selection (default: loto_pair_win)
  --plot-bucket-filter CSV
                          Optional bucket filter pass-through to plotting wrapper
  --plot-top-k N          Top-k lines for plot script (default: 8)
  --plot-context-target-transform MODE
                          residual|zscore (default: zscore)
  --plot-context-target-plot-space MODE
                          model_space|residual|absolute (default: model_space)
  --plot-prediction-transform MODE
                          none|zscore (default: none)
  --plot-heldout-protocols CSV
                          Heldout protocols for fit plots (default: lobo,loto,jointood,model)
  --plot-heldout-spaces CSV
                          Heldout plot spaces (default: model_space)
  --plot-heldout-color-by COL
                          Heldout scatter color column (default: train_dataset)
  --dry-run               Print commands without executing heavy jobs
  -h, --help              Show help
EOF
}

sanitize_tag() {
  local raw="$1"
  raw="${raw// /_}"
  raw="${raw//,/_}"
  raw="${raw//\//_}"
  raw="${raw//:/_}"
  raw="${raw//|/_}"
  raw="${raw//__/_}"
  raw="${raw//__/_}"
  raw="$(echo "$raw" | tr -cd '[:alnum:]_.+-')"
  if [[ -z "$raw" ]]; then
    raw="na"
  fi
  echo "$raw"
}

resolve_summary_root() {
  local in_root="$1"
  if [[ "$(basename "$in_root")" == "density_joint" ]]; then
    echo "$(dirname "$in_root")"
    return
  fi
  echo "$in_root"
}

resolve_density_root() {
  local in_root="$1"
  local summary_root="$2"
  if [[ "$(basename "$in_root")" == "density_joint" ]]; then
    echo "$in_root"
    return
  fi
  if [[ -d "${summary_root}/density_joint" ]]; then
    echo "${summary_root}/density_joint"
    return
  fi
  echo "${summary_root}/density_joint"
}

ensure_method_summary() {
  local summary_root="$1"
  local density_root="$2"
  local summary_csv="${summary_root}/method_summary.csv"
  if [[ -f "$summary_csv" ]]; then
    return 0
  fi
  if [[ ! -d "$density_root" ]]; then
    echo "Missing density directory for summary bootstrap: ${density_root}" >&2
    return 1
  fi
  local manifest="${summary_root}/method_summary_manifest.autogen.yaml"
  echo "Bootstrapping method_summary.csv from ${density_root}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would bootstrap manifest: ${manifest}"
    echo "[dry-run] would compile method summary at: ${summary_csv}"
    return 0
  fi
  python - "$density_root" "$manifest" <<'PY'
import json
import sys
from pathlib import Path
import yaml

density_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])

methods = []
for run_dir in sorted(density_root.iterdir()):
    if not run_dir.is_dir():
        continue
    if not (run_dir / "auc_with_features.csv").exists():
        continue
    if not (run_dir / "prediction_jointood_summary.csv").exists():
        continue

    meta = {}
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    preds = meta.get("predictors") or []
    if isinstance(preds, str):
        preds = [p.strip() for p in preds.split(",") if p.strip()]
    preds = [str(p) for p in preds]

    has_flow = any(p.startswith("flow_") for p in preds)
    has_hof = any(p.startswith("hof_") for p in preds)
    has_dino = any(p.startswith("dino_") for p in preds)
    has_mmd = any(p.endswith("_mmd") for p in preds)

    if has_mmd and not (has_flow or has_hof or has_dino):
        family = "mmd"
        symmetry = "sym"
    elif (has_flow or has_hof) and has_dino:
        family = "mixed"
        symmetry = "asym"
    elif has_dino:
        family = "appearance"
        symmetry = "asym"
    elif has_hof:
        family = "motion"
        symmetry = "asym"
    elif has_flow:
        family = "flow"
        symmetry = "asym"
    else:
        family = "mixed"
        symmetry = "mixed"

    methods.append(
        {
            "name": run_dir.name.replace("leakage_free_", ""),
            "path": str(run_dir),
            "family": family,
            "symmetry": symmetry,
            "notes": "autogen_manifest=1",
        }
    )

if not methods:
    raise SystemExit(f"No method dirs found under {density_root}")

manifest_path.write_text(yaml.safe_dump({"methods": methods}, sort_keys=False))
print(f"Wrote manifest: {manifest_path} ({len(methods)} methods)")
PY

  python scripts/compile_method_summary.py \
    --manifest "$manifest" \
    --output "$summary_csv" \
    --output-md "${summary_root}/method_summary.md"
}

check_ready() {
  local summary_root="$1"
  local density_root="$2"
  ensure_method_summary "$summary_root" "$density_root"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  local summary="${summary_root}/method_summary.csv"
  if [[ ! -f "$summary" ]]; then
    echo "Missing required file: $summary" >&2
    exit 1
  fi
}

run_plots_for_heldout() {
  local label="$1"
  local density_root="$2"
  local heldout_dir="$3"
  if [[ "$RUN_PLOTS" -ne 1 ]]; then
    return 0
  fi
  if [[ ! -d "$density_root" ]]; then
    echo "Warning: skipping plots (missing density root): ${density_root}" >&2
    return 0
  fi
  local first_head="${HEADS%%,*}"
  local plot_out="${heldout_dir}/paper_plots_${label}_${PLOT_BEST_CV_METRIC}"
  local -a args=(
    --run-root "$density_root"
    --best-cv-metric "$PLOT_BEST_CV_METRIC"
    --max-signal-k "$MAX_SIGNAL_K"
    --output-dir "$plot_out"
    --top-k "$PLOT_TOP_K"
    --context-target-transform "$PLOT_CONTEXT_TARGET_TRANSFORM"
    --context-target-plot-space "$PLOT_CONTEXT_TARGET_PLOT_SPACE"
    --prediction-transform "$PLOT_PREDICTION_TRANSFORM"
    --heldout-protocols "$PLOT_HELDOUT_PROTOCOLS"
    --heldout-plot-spaces "$PLOT_HELDOUT_PLOT_SPACES"
    --heldout-model-cv-dir "$heldout_dir"
    --heldout-model-cv-head "$first_head"
    --heldout-color-by "$PLOT_HELDOUT_COLOR_BY"
    --heldout-save-points
  )
  if [[ -n "$PLOT_BUCKET_FILTER" ]]; then
    args+=(--bucket-filter "$PLOT_BUCKET_FILTER")
  fi
  echo "Running heldout visualization plots"
  echo "  density root: ${density_root}"
  echo "  output:       ${plot_out}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] cmd: python scripts/run_plot_residual_rank_param_matched.py'
    printf ' %q' "${args[@]}"
    echo
  else
    python scripts/run_plot_residual_rank_param_matched.py "${args[@]}"
  fi
}

run_one() {
  local label="$1"
  local raw_root="$2"
  local mode="$3"
  local summary_root
  local density_root
  local root_tag
  local protocols_tag
  local heads_tag

  summary_root="$(resolve_summary_root "$raw_root")"
  density_root="$(resolve_density_root "$raw_root" "$summary_root")"
  check_ready "$summary_root" "$density_root"

  root_tag="$(sanitize_tag "$(basename "$summary_root")")"
  protocols_tag="$(sanitize_tag "$PROTOCOLS")"
  heads_tag="$(sanitize_tag "$HEADS")"
  local out_dir="${OUTPUT_ROOT}/${label}__${root_tag}__p_${protocols_tag}__h_${heads_tag}"
  mkdir -p "$out_dir"

  echo "Running heldout model CV for ${label}"
  echo "  summary root: ${summary_root}"
  echo "  density root: ${density_root}"
  echo "  output:       ${out_dir}"

  local -a args=(
    --run-roots "$summary_root"
    --output-dir "$out_dir"
    --row-source raw
    --heads "$HEADS"
    --protocols "$PROTOCOLS"
    --lanes "$LANES"
    --top-n-per-lane "$TOP_N_PER_LANE"
    --min-signal-k "$MIN_SIGNAL_K"
    --max-signal-k "$MAX_SIGNAL_K"
    --candidate-dedup-primary-metric jointood_rank_spearman
    --candidate-dedup-tiebreak-metric jointood_rank_pairwise_cindex
    --candidate-selection-primary-metric jointood_rank_spearman
    --candidate-selection-tiebreak-metric jointood_rank_pairwise_cindex
    --target-col auc_normalized_observed
    --train-col train_dataset
    --benchmark-col benchmark
    --option-col train_dataset
    --model-group-col model_family_encoder
    --pairwise-group-cols benchmark,model_family_encoder
    --rank-grouping fold_benchmark
    --rank-context-cols model_family_encoder
    --cv-residualize-target-by-context
    --cv-residual-context-cols benchmark,model_family_encoder
    --cv-residual-target-transform zscore
    --cv-residual-target-std-eps 1e-9
    --cv-residual-eval-space residual
    --cv-fewshot-context-calibration-cols benchmark,model_family_encoder
    --cv-fewshot-context-calibration-k "$FEWSHOT_K"
    --cv-fewshot-context-calibration-min-group-size "$FEWSHOT_MIN_GROUP_SIZE"
    --cv-fewshot-context-calibration-seed "$FEWSHOT_SEED"
    --ridge-alpha 10
    --max-hard-folds "$MAX_HARD_FOLDS"
    --holdout-train-k "$HOLDOUT_TRAIN_K"
    --permutation-samples "$PERMUTATION_SAMPLES"
    --permutation-mode "$PERMUTATION_MODE"
    --save-pred-rows
    --save-fold-rows
  )
  if [[ "$mode" == "fewshot" ]]; then
    args+=(--cv-fewshot-context-calibration)
    if [[ "$FEWSHOT_BACKOFF" -eq 1 ]]; then
      args+=(--cv-fewshot-context-calibration-backoff)
    else
      args+=(--no-cv-fewshot-context-calibration-backoff)
    fi
  else
    args+=(--no-cv-fewshot-context-calibration)
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] cmd: python scripts/build_heldout_model_cv.py'
    printf ' %q' "${args[@]}"
    echo
  else
    python scripts/build_heldout_model_cv.py "${args[@]}"
  fi
  run_plots_for_heldout "$label" "$density_root" "$out_dir"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      RUN_ROOT="$2"; shift 2 ;;
    --run-label)
      RUN_LABEL="$2"; shift 2 ;;
    --run-mode)
      RUN_MODE="$2"; shift 2 ;;
    --zero-shot-root)
      ZERO_SHOT_ROOT="$2"; shift 2 ;;
    --fewshot-root)
      FEWSHOT_ROOT="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --protocols)
      PROTOCOLS="$2"; shift 2 ;;
    --heads)
      HEADS="$2"; shift 2 ;;
    --lanes)
      LANES="$2"; shift 2 ;;
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
    --permutation-samples)
      PERMUTATION_SAMPLES="$2"; shift 2 ;;
    --permutation-mode)
      PERMUTATION_MODE="$2"; shift 2 ;;
    --fewshot-k)
      FEWSHOT_K="$2"; shift 2 ;;
    --fewshot-min-group-size)
      FEWSHOT_MIN_GROUP_SIZE="$2"; shift 2 ;;
    --fewshot-backoff)
      FEWSHOT_BACKOFF=1; shift 1 ;;
    --no-fewshot-backoff)
      FEWSHOT_BACKOFF=0; shift 1 ;;
    --fewshot-seed)
      FEWSHOT_SEED="$2"; shift 2 ;;
    --no-plots)
      RUN_PLOTS=0; shift 1 ;;
    --plot-best-cv-metric)
      PLOT_BEST_CV_METRIC="$2"; shift 2 ;;
    --plot-bucket-filter)
      PLOT_BUCKET_FILTER="$2"; shift 2 ;;
    --plot-top-k)
      PLOT_TOP_K="$2"; shift 2 ;;
    --plot-context-target-transform)
      PLOT_CONTEXT_TARGET_TRANSFORM="$2"; shift 2 ;;
    --plot-context-target-plot-space)
      PLOT_CONTEXT_TARGET_PLOT_SPACE="$2"; shift 2 ;;
    --plot-prediction-transform)
      PLOT_PREDICTION_TRANSFORM="$2"; shift 2 ;;
    --plot-heldout-protocols)
      PLOT_HELDOUT_PROTOCOLS="$2"; shift 2 ;;
    --plot-heldout-spaces)
      PLOT_HELDOUT_PLOT_SPACES="$2"; shift 2 ;;
    --plot-heldout-color-by)
      PLOT_HELDOUT_COLOR_BY="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -n "$RUN_ROOT" ]]; then
  if [[ "$RUN_MODE" != "fewshot" && "$RUN_MODE" != "zeroshot" ]]; then
    echo "Invalid --run-mode: ${RUN_MODE} (expected fewshot|zeroshot)" >&2
    exit 2
  fi
  run_one "$RUN_LABEL" "$RUN_ROOT" "$RUN_MODE"
else
  run_one "zero_shot" "$ZERO_SHOT_ROOT" "zeroshot"
  run_one "fewshot_k${FEWSHOT_K}" "$FEWSHOT_ROOT" "fewshot"
fi

echo "Done. Heldout outputs under: ${OUTPUT_ROOT}"
