#!/usr/bin/env bash
# Transfer Analysis v3 — end-to-end pipeline
# Run from project root: bash scripts/transfer_analysis_v3/run_pipeline.sh
# Each step is skipped if its output already exists (safe to restart).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/scripts/transfer_analysis_v3/logs"
mkdir -p "$LOG_DIR" analysis_v3

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Run mode: set MINIMAL=1 to skip mixed training sets (10 train × 9 eval = 90
# pairs instead of 20 × 9 = 180). Useful for the first pass when mixed-dataset
# DINO vectors may not be needed / disk space is tight.
#
#   MINIMAL=1 bash scripts/transfer_analysis_v3/run_pipeline.sh   # minimal
#   bash scripts/transfer_analysis_v3/run_pipeline.sh              # full
# ---------------------------------------------------------------------------
MINIMAL="${MINIMAL:-0}"
LOCO="${LOCO:-1}"
FLOW_ONLY="${FLOW_ONLY:-0}"
PURE_ONLY="${PURE_ONLY:-1}"

# Path overrides — set VEC_DIR for a non-default vector cache location (e.g. RC cluster).
VEC_DIR="${VEC_DIR:-/mnt/nvme_1tb_b/coverage_vectors}"
CLEAN_RESULTS="${CLEAN_RESULTS:-0}"
TARGETS="${TARGETS:-auc_normalized peak_pck}"
FLOW_SPLITS="${FLOW_SPLITS:-loto lobo joint_cell}"
FLOW_SWEEP_MODE="${FLOW_SWEEP_MODE:-diagnostic}"
RUN_TP_KRR="${RUN_TP_KRR:-0}"
RUN_SYM_SELF="${RUN_SYM_SELF:-1}"
REFRESH_FLOW_FEATURES="${REFRESH_FLOW_FEATURES:-0}"
REFRESH_FLOW_MMD="${REFRESH_FLOW_MMD:-0}"
REQUIRE_FLOW_AUDIT_CLEAN="${REQUIRE_FLOW_AUDIT_CLEAN:-0}"
FLOW_REFRESH_MODE="${FLOW_REFRESH_MODE:-append}"
FLOW_REFRESH_PARALLEL="${FLOW_REFRESH_PARALLEL:-0}"
FLOW_REFRESH_ONLY="${FLOW_REFRESH_ONLY:-0}"
FLOW_KMEANS_CUDA_VISIBLE_DEVICES="${FLOW_KMEANS_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES="${FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
FLOW_MMD_CUDA_VISIBLE_DEVICES="${FLOW_MMD_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
FLOW_AUDIT_REQUIRE_FAMILIES="${FLOW_AUDIT_REQUIRE_FAMILIES:-flow_raw_coverage flow_kmeans_coverage flow_fid_sw2 pairwise_self_flow_train_eval}"
FLOW_DIAGNOSTIC_FEATURE_GROUPS="${FLOW_DIAGNOSTIC_FEATURE_GROUPS:-density_train density_eval density_idw random_idw sample_count vector_density_simple train_profile_simple profile_simple flow_fid_only flow_w2_only flow_kl motion_km flow_fid_profile flow_w2_profile flow_kl_profile motion_km_profile}"
FLOW_DIAGNOSTIC_MODELS="${FLOW_DIAGNOSTIC_MODELS:-ridge_abs two_way_mixed_ridge anchor_additive_ridge anchor_lowrank_bilinear_ridge anchor_bilinear_ridge anchor_bilinear_shrunk_ridge kernel_mixed_additive kernel_mixed_interaction ridge_pairwise idw_prior_two_way idw_prior_two_way_rank uniform_prior_two_way random_prior_two_way}"
EXCLUDE_FIT_TRAIN_DATASETS="${EXCLUDE_FIT_TRAIN_DATASETS:-}"
DROP_TRAIN_DATASETS="${DROP_TRAIN_DATASETS:-}"

PURE_TRAIN_DATASETS="flyingthings imagenet2dwarp movi_f pointodyssey sintel spair synthetic synthetic_2d_warp synthetic_large_zoom synthetic_random_flipping synthetic_small_zoom"

if [ "$FLOW_ONLY" = "1" ]; then
    log "Mode: FLOW_ONLY (rerunnable focused flow-feature experiments)"
    case "$FLOW_SWEEP_MODE" in
        diagnostic|full) ;;
        *)
            log "ERROR: FLOW_SWEEP_MODE must be 'diagnostic' or 'full' (got '$FLOW_SWEEP_MODE')."
            exit 1
            ;;
    esac
    if [ "$PURE_ONLY" = "1" ]; then
        BUILD_TABLE_ARGS="--train-datasets ${PURE_TRAIN_DATASETS} --min-context-size 8"
        OUT_PREFIX="flow_only_pure"
        log "Dataset scope: pure training datasets only"
    else
        BUILD_TABLE_ARGS="--min-context-size 8"
        OUT_PREFIX="flow_only"
        log "Dataset scope: all training datasets in AUC table"
    fi
    EXCLUDE_FIT_ARGS=""
    EXCLUDE_MI_ARGS=""
    if [ -n "$EXCLUDE_FIT_TRAIN_DATASETS" ]; then
        EXCLUDE_TAG="${EXCLUDE_FIT_TRAIN_DATASETS// /_}"
        OUT_PREFIX="${OUT_PREFIX}_nofit_${EXCLUDE_TAG}"
        EXCLUDE_FIT_ARGS="--exclude-fit-train-datasets ${EXCLUDE_FIT_TRAIN_DATASETS}"
        EXCLUDE_MI_ARGS="--exclude-train-datasets ${EXCLUDE_FIT_TRAIN_DATASETS}"
        log "Fit-pool exclusion: train_dataset in [$EXCLUDE_FIT_TRAIN_DATASETS]"
    fi
    DROP_TRAIN_ARGS=""
    if [ -n "$DROP_TRAIN_DATASETS" ]; then
        DROP_TAG="${DROP_TRAIN_DATASETS// /_}"
        OUT_PREFIX="${OUT_PREFIX}_drop_${DROP_TAG}"
        DROP_TRAIN_ARGS="--drop-train-datasets ${DROP_TRAIN_DATASETS}"
        EXCLUDE_MI_ARGS="--exclude-train-datasets ${DROP_TRAIN_DATASETS}"
        log "Analysis-row drop: train_dataset in [$DROP_TRAIN_DATASETS]"
    fi
    if [ "$FLOW_SWEEP_MODE" = "diagnostic" ]; then
        OUT_PREFIX="${OUT_PREFIX}_diagnostic"
    fi
    log "Targets: $TARGETS"
    log "Experiment splits: $FLOW_SPLITS"
    log "Sweep mode: $FLOW_SWEEP_MODE"
    log "Output prefix: scripts/transfer_analysis_v3/results/${OUT_PREFIX}_<target>"
    case "$FLOW_REFRESH_MODE" in
        append|scratch) ;;
        *)
            log "ERROR: FLOW_REFRESH_MODE must be 'append' or 'scratch' (got '$FLOW_REFRESH_MODE')."
            exit 1
            ;;
    esac
    case "$FLOW_REFRESH_PARALLEL" in
        0|1) ;;
        *)
            log "ERROR: FLOW_REFRESH_PARALLEL must be 0 or 1 (got '$FLOW_REFRESH_PARALLEL')."
            exit 1
            ;;
    esac
    case "$FLOW_REFRESH_ONLY" in
        0|1) ;;
        *)
            log "ERROR: FLOW_REFRESH_ONLY must be 0 or 1 (got '$FLOW_REFRESH_ONLY')."
            exit 1
            ;;
    esac
    if [[ " $FLOW_SPLITS " == *" loco "* ]]; then
        log "NOTE: FLOW_SPLITS includes loco; this is the slow leave-context-out split."
    else
        log "NOTE: loco is not included. Add FLOW_SPLITS=\"loto lobo joint_cell loco\" to run it."
    fi

    if [ ! -f analysis_v3/pairwise_self_distances.csv ]; then
        log "ERROR: analysis_v3/pairwise_self_distances.csv is required for ridge_pairwise and flow_kl."
        log "Run compute_pairwise_self_distances.py first, or set up Step 0d in the full pipeline."
        exit 1
    fi

    if [ "$REFRESH_FLOW_FEATURES" = "1" ]; then
        if [ "$FLOW_REFRESH_MODE" = "scratch" ]; then
            STAMP="$(date '+%Y%m%d_%H%M%S')"
            log "Flow-only feature refresh mode: scratch (archiving existing feature CSVs with suffix .bak_${STAMP})"
            for csv in \
                analysis/coverage_v2_flow_only_raw_joint_full.csv \
                analysis/coverage_v2_flow_only_raw_joint_curves_full.csv \
                analysis/coverage_v2_flow_only_raw_joint_kmeans_full.csv \
                analysis/coverage_v2_flow_only_raw_joint_kmeans_curves_full.csv \
                analysis_v3/kl_flow_features.csv \
                analysis_v3/symmetric_distances.csv
            do
                if [ -f "$csv" ]; then
                    mv "$csv" "${csv}.bak_${STAMP}"
                    log "  archived $csv"
                fi
            done
        else
            log "Flow-only feature refresh mode: append/resume (existing compatible rows are reused)"
        fi
        log "Flow-only Step 0f: Materializing directed raw flow coverage from pairwise_self_distances.csv..."
        python scripts/transfer_analysis_v3/materialize_flow_raw_coverage_from_pairwise.py \
            --pairwise-self analysis_v3/pairwise_self_distances.csv \
            --output analysis/coverage_v2_flow_only_raw_joint_full.csv \
            2>&1 | tee "$LOG_DIR/flow_only_refresh_flow_coverage.log"

        if [ "$FLOW_REFRESH_PARALLEL" = "1" ]; then
            log "Flow-only Step 0g/0h: Refreshing k-means and FID/SW2 in parallel."
            log "  k-means CUDA_VISIBLE_DEVICES=${FLOW_KMEANS_CUDA_VISIBLE_DEVICES:-<inherited>}"
            log "  FID/SW2  CUDA_VISIBLE_DEVICES=${FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES:-<inherited>}"
            (
                set -o pipefail
                if [ -n "$FLOW_KMEANS_CUDA_VISIBLE_DEVICES" ]; then
                    export CUDA_VISIBLE_DEVICES="$FLOW_KMEANS_CUDA_VISIBLE_DEVICES"
                fi
                python scripts/calculate_coverage_faiss_flow_kmeans.py \
                    --config src/configs/coverage_configs/coverage_faiss_flow_only_raw_joint_full.yaml
            ) 2>&1 | tee "$LOG_DIR/flow_only_refresh_flow_kmeans.log" &
            KMEANS_PID=$!
            (
                set -o pipefail
                if [ -n "$FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES" ]; then
                    export CUDA_VISIBLE_DEVICES="$FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES"
                fi
                python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
                    --flow-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
                    --vec-dir $VEC_DIR \
                    --output analysis_v3/symmetric_distances.csv \
                    --n-proj 200 \
                    --sw-samples 100000 \
                    --fid-samples 200000 \
                    --skip-dino
            ) 2>&1 | tee "$LOG_DIR/flow_only_refresh_symmetric_distances.log" &
            SYM_PID=$!
            set +e
            wait "$KMEANS_PID"
            KMEANS_STATUS=$?
            wait "$SYM_PID"
            SYM_STATUS=$?
            set -e
            if [ "$KMEANS_STATUS" -ne 0 ] || [ "$SYM_STATUS" -ne 0 ]; then
                log "ERROR: parallel flow refresh failed (k-means=$KMEANS_STATUS, FID/SW2=$SYM_STATUS)."
                exit 1
            fi
        else
            log "Flow-only Step 0g: Refreshing flow k-means coverage features..."
            if [ -n "$FLOW_KMEANS_CUDA_VISIBLE_DEVICES" ]; then
                CUDA_VISIBLE_DEVICES="$FLOW_KMEANS_CUDA_VISIBLE_DEVICES" \
                python scripts/calculate_coverage_faiss_flow_kmeans.py \
                    --config src/configs/coverage_configs/coverage_faiss_flow_only_raw_joint_full.yaml \
                    2>&1 | tee "$LOG_DIR/flow_only_refresh_flow_kmeans.log"
            else
                python scripts/calculate_coverage_faiss_flow_kmeans.py \
                    --config src/configs/coverage_configs/coverage_faiss_flow_only_raw_joint_full.yaml \
                    2>&1 | tee "$LOG_DIR/flow_only_refresh_flow_kmeans.log"
            fi

            log "Flow-only Step 0h: Refreshing flow FID/SW2 train-eval features..."
            if [ -n "$FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES" ]; then
                CUDA_VISIBLE_DEVICES="$FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES" \
                python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
                    --flow-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
                    --vec-dir $VEC_DIR \
                    --output analysis_v3/symmetric_distances.csv \
                    --n-proj 200 \
                    --sw-samples 100000 \
                    --fid-samples 200000 \
                    --skip-dino \
                    2>&1 | tee "$LOG_DIR/flow_only_refresh_symmetric_distances.log"
            else
                python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
                    --flow-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
                    --vec-dir $VEC_DIR \
                    --output analysis_v3/symmetric_distances.csv \
                    --n-proj 200 \
                    --sw-samples 100000 \
                    --fid-samples 200000 \
                    --skip-dino \
                    2>&1 | tee "$LOG_DIR/flow_only_refresh_symmetric_distances.log"
            fi
        fi
    else
        log "Flow-only feature refresh disabled. Set REFRESH_FLOW_FEATURES=1 to recompute stale raw/k-means/FID/SW2 pairs."
    fi

    if [ "$REFRESH_FLOW_MMD" = "1" ]; then
        if [ "$FLOW_REFRESH_MODE" = "scratch" ] && [ -f flow_mmd_results_fast.csv ]; then
            STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
            mv flow_mmd_results_fast.csv "flow_mmd_results_fast.csv.bak_${STAMP}"
            log "  archived flow_mmd_results_fast.csv"
            for mmd_cache in flow_mmd_results_fast.state.pt flow_mmd_results_fast.state.npz flow_mmd_results_fast.counts.json; do
                if [ -f "$mmd_cache" ]; then
                    mv "$mmd_cache" "${mmd_cache}.bak_${STAMP}"
                    log "  archived $mmd_cache"
                fi
            done
        fi
        log "Flow-only Step 0m: Refreshing flow MMD features..."
        if [ -n "$FLOW_MMD_CUDA_VISIBLE_DEVICES" ]; then
            CUDA_VISIBLE_DEVICES="$FLOW_MMD_CUDA_VISIBLE_DEVICES" \
            python scripts/calculate_flow_mmd.py \
                --config src/configs/mmd_configs/flow_mmd_config_full.yaml \
                2>&1 | tee "$LOG_DIR/flow_only_refresh_flow_mmd.log"
        else
            python scripts/calculate_flow_mmd.py \
                --config src/configs/mmd_configs/flow_mmd_config_full.yaml \
                2>&1 | tee "$LOG_DIR/flow_only_refresh_flow_mmd.log"
        fi
    fi

    if [ "$FLOW_REFRESH_ONLY" = "1" ]; then
        log "Flow-only refresh-only mode: auditing feature coverage and exiting before model sweeps."
        if [ "$REQUIRE_FLOW_AUDIT_CLEAN" = "1" ]; then
            # shellcheck disable=SC2086
            python scripts/transfer_analysis_v3/audit_flow_feature_coverage.py \
                --require-clean-families $FLOW_AUDIT_REQUIRE_FAMILIES \
                2>&1 | tee "$LOG_DIR/flow_only_feature_audit.log"
        else
            python scripts/transfer_analysis_v3/audit_flow_feature_coverage.py \
                2>&1 | tee "$LOG_DIR/flow_only_feature_audit.log"
        fi
        exit 0
    fi

    if [ "$RUN_SYM_SELF" = "1" ]; then
        SYM_SELF_READY=$(python -c "
import pandas as pd
from pathlib import Path
p=Path('analysis_v3/pairwise_self_distances.csv')
if not p.exists():
    print(0)
else:
    df=pd.read_csv(p)
    cols={'flow_fid_self','flow_sliced_w2_self','flow_mmd_self'}
    if not cols.issubset(df.columns):
        print(0)
    else:
        sub=df[(df.get('space')=='flow') & (df.get('pair_type').isin(['train_train','eval_eval']))]
        ok=len(sub)>0 and sub['flow_fid_self'].notna().all() and sub['flow_sliced_w2_self'].notna().all()
        print(1 if ok else 0)
")
        if [ "$SYM_SELF_READY" = "1" ]; then
            log "Flow-only Step 0s: Symmetric self-distances already merged — skipping"
        else
            log "Flow-only Step 0s: Building symmetric self-distances for IDW neighborhoods..."
            python scripts/transfer_analysis_v3/build_symmetric_self_distances.py \
                --self-dist analysis_v3/pairwise_self_distances.csv \
                --output analysis_v3/pairwise_symmetric_distances.csv \
                --vec-dir $VEC_DIR \
                --flow-mmd-csv flow_mmd_results_fast.csv \
                --pair-types train_train eval_eval \
                --fid-samples 200000 \
                --sw-samples 100000 \
                --n-proj 200 \
                2>&1 | tee "$LOG_DIR/flow_only_symmetric_self_distances.log"
        fi
    fi

    log "Flow-only Step 1: Assembling transfer table..."
    # shellcheck disable=SC2086
    python scripts/transfer_analysis_v3/build_table.py \
        $BUILD_TABLE_ARGS \
        2>&1 | tee "$LOG_DIR/flow_only_build_table.log"

    log "Flow-only Step 1a: Auditing active flow feature coverage..."
    if [ "$REQUIRE_FLOW_AUDIT_CLEAN" = "1" ]; then
        # shellcheck disable=SC2086
        python scripts/transfer_analysis_v3/audit_flow_feature_coverage.py \
            --require-clean-families $FLOW_AUDIT_REQUIRE_FAMILIES \
            2>&1 | tee "$LOG_DIR/flow_only_feature_audit.log"
    else
        python scripts/transfer_analysis_v3/audit_flow_feature_coverage.py \
            2>&1 | tee "$LOG_DIR/flow_only_feature_audit.log"
    fi

    for TARGET in $TARGETS; do
        OUT_DIR="scripts/transfer_analysis_v3/results/${OUT_PREFIX}_${TARGET}"
        MI_DIR="analysis_v3/feature_mi_${TARGET}"
        if [ -n "$EXCLUDE_FIT_TRAIN_DATASETS" ] || [ -n "$DROP_TRAIN_DATASETS" ]; then
            MI_DIR="analysis_v3/feature_mi_${OUT_PREFIX}_${TARGET}"
        fi
        if [ "$CLEAN_RESULTS" = "1" ]; then
            log "Flow-only (${TARGET}): Cleaning $OUT_DIR"
            rm -rf "$OUT_DIR"
        fi
        mkdir -p "$OUT_DIR"

        log "Flow-only (${TARGET}): Computing feature MI..."
        python scripts/transfer_analysis_v3/compute_feature_mi.py \
            --table scripts/transfer_analysis_v3/transfer_table.csv \
            --target "$TARGET" \
            --out-dir "$MI_DIR" \
            --n-boot 500 --k-neighbors 5 \
            $EXCLUDE_MI_ARGS \
            2>&1 | tee "$LOG_DIR/flow_only_feature_mi_${TARGET}.log"

        log "Flow-only (${TARGET}): Running baselines..."
        # shellcheck disable=SC2086
        PYTHONUNBUFFERED=1 python scripts/transfer_analysis_v3/run_experiments.py \
            --splits $FLOW_SPLITS \
            --models random global_prior \
            --feature-groups flow_nn \
            --target "$TARGET" \
            --output-dir "$OUT_DIR" \
            $DROP_TRAIN_ARGS \
            $EXCLUDE_FIT_ARGS \
            2>&1 | tee "$LOG_DIR/flow_only_baselines_${TARGET}.log"

        if [ "$FLOW_SWEEP_MODE" = "diagnostic" ]; then
            log "Flow-only (${TARGET}): Running narrow density-vs-geometry diagnostic grid..."
            # Matches HANDOFF.md's current diagnostic set. This keeps the run focused on
            # whether flow geometry adds value beyond sample/vector profile controls.
            # shellcheck disable=SC2086
            PYTHONUNBUFFERED=1 python scripts/transfer_analysis_v3/run_experiments.py \
                --splits $FLOW_SPLITS \
                --models $FLOW_DIAGNOSTIC_MODELS \
                --feature-groups $FLOW_DIAGNOSTIC_FEATURE_GROUPS \
                --pairwise-spaces flow \
                --self-dist-csv analysis_v3/pairwise_self_distances.csv \
                --target "$TARGET" \
                --output-dir "$OUT_DIR" \
                $DROP_TRAIN_ARGS \
                $EXCLUDE_FIT_ARGS \
                2>&1 | tee "$LOG_DIR/flow_only_diagnostic_${TARGET}.log"
        else
            log "Flow-only (${TARGET}): Running vanilla Ridge baselines and feature ablation..."
            # ridge = rank-score baseline; ridge_abs = absolute AUC/PCK Ridge baseline.
            # shellcheck disable=SC2086
            PYTHONUNBUFFERED=1 python scripts/transfer_analysis_v3/run_experiments.py \
                --splits $FLOW_SPLITS \
                --models ridge ridge_abs two_way_mixed_ridge \
                --feature-groups flow_nn flow_eps flow_km flow_kl motion motion_km sym_flow \
                                 flow_mmd_only flow_fid_only flow_w2_only \
                                 density_train density_eval \
                                 sample_count sample_count_train sample_count_eval \
                                 vector_density vector_density_train vector_density_eval \
                                 train_profile eval_profile profile_density \
                                 flow_mmd_profile flow_fid_profile flow_w2_profile \
                                 flow_kl_profile motion_km_profile \
                --target "$TARGET" \
                --output-dir "$OUT_DIR" \
                $DROP_TRAIN_ARGS \
                $EXCLUDE_FIT_ARGS \
                2>&1 | tee "$LOG_DIR/flow_only_ridge_${TARGET}.log"

            log "Flow-only (${TARGET}): Running variant-aware coupled Ridge+IDW..."
            # shellcheck disable=SC2086
            PYTHONUNBUFFERED=1 python scripts/transfer_analysis_v3/run_experiments.py \
                --splits $FLOW_SPLITS \
                --models anchor_additive_ridge anchor_lowrank_bilinear_ridge \
                     anchor_bilinear_ridge anchor_bilinear_shrunk_ridge \
                     kernel_mixed_additive kernel_mixed_interaction \
                     ridge_pairwise ridge_pairwise_cross_resid \
                     ridge_pairwise_cross_resid_spline \
                     idw_prior_residual idw_prior_context idw_prior_context_local \
                     idw_prior_two_way idw_prior_two_way_rank \
                     idw_prior_two_way_spline \
                     ridge_pairwise_uniform ridge_pairwise_random \
                     uniform_prior_residual random_prior_residual \
                     uniform_prior_two_way random_prior_two_way \
                     uniform_prior_two_way_spline random_prior_two_way_spline \
                --feature-groups flow_nn flow_eps flow_km flow_kl motion motion_km sym_flow \
                                 flow_mmd_only flow_fid_only flow_w2_only \
                                 density_train density_eval density_idw random_idw \
                                 sample_count sample_count_train sample_count_eval \
                                 vector_density vector_density_train vector_density_eval \
                                 train_profile eval_profile profile_density \
                                 flow_mmd_profile flow_fid_profile flow_w2_profile \
                                 flow_kl_profile motion_km_profile \
                --pairwise-spaces flow \
                --self-dist-csv analysis_v3/pairwise_self_distances.csv \
                --target "$TARGET" \
                --output-dir "$OUT_DIR" \
                $DROP_TRAIN_ARGS \
                $EXCLUDE_FIT_ARGS \
                2>&1 | tee "$LOG_DIR/flow_only_ridge_pairwise_${TARGET}.log"
        fi

        if [ "$RUN_TP_KRR" = "1" ]; then
            log "Flow-only (${TARGET}): Running optional TP-KRR diagnostic..."
            # TP-KRR is not useful for LOTO with unseen train datasets, so keep it
            # to LOBO/LOCO-cell unless FLOW_TP_SPLITS is explicitly overridden.
            FLOW_TP_SPLITS="${FLOW_TP_SPLITS:-lobo loco_cell}"
            # shellcheck disable=SC2086
            PYTHONUNBUFFERED=1 python scripts/transfer_analysis_v3/run_experiments.py \
                --splits $FLOW_TP_SPLITS \
                --models krr_tp_flow_nn \
                --feature-groups flow_nn \
                --self-dist-csv analysis_v3/pairwise_self_distances.csv \
                --target "$TARGET" \
                --output-dir "$OUT_DIR" \
                $DROP_TRAIN_ARGS \
                $EXCLUDE_FIT_ARGS \
                2>&1 | tee "$LOG_DIR/flow_only_tpkrr_${TARGET}.log"
        fi

        log "Flow-only (${TARGET}): Compiling report..."
        python scripts/transfer_analysis_v3/compile_results.py \
            --results-dir "$OUT_DIR" \
            --output "$OUT_DIR/results.md" \
            --mi-csv "$MI_DIR/feature_mi.csv" \
            2>&1 | tee "$LOG_DIR/flow_only_compile_${TARGET}.log"
        log "Flow-only (${TARGET}): Report at $OUT_DIR/results.md"
    done

    log "Flow-only pipeline complete."
    exit 0
fi

N_EVAL="${N_EVAL:-9}"   # override to 10 on RC (adds synthetic/val)

if [ "$MINIMAL" = "1" ]; then
    DINO_CONFIG="${DINO_CONFIG:-src/configs/coverage_configs/coverage_faiss_dino_v3_minimal.yaml}"
    DINO_EXPECTED=$((11 * N_EVAL))
    BUILD_TABLE_ARGS="--train-datasets ${PURE_TRAIN_DATASETS} --min-context-size 8"
    log "Mode: MINIMAL (11 pure training datasets incl. movi_f, ${DINO_EXPECTED} DINO pairs expected, min-context-size=8)"
else
    DINO_CONFIG="${DINO_CONFIG:-src/configs/coverage_configs/coverage_faiss_dino_v3.yaml}"
    DINO_EXPECTED=$((22 * N_EVAL))
    BUILD_TABLE_ARGS="--min-context-size 8"
    log "Mode: FULL (including mixed training sets, ${DINO_EXPECTED} DINO pairs expected, min-context-size=8)"
fi

# ---------------------------------------------------------------------------
# Step 0a: DINO pairwise coverage (GPU FAISS, nprobe=4)
# Resumable — already-computed pairs are skipped by the coverage script.
# ---------------------------------------------------------------------------
DINO_CSV="$ROOT/analysis_v3/coverage_dino_full.csv"
if python3 -c "
import pandas as pd, sys
df = pd.read_csv('$DINO_CSV')
ok = 'mean_nn_eval_to_train_k1' in df.columns and len(df) >= $DINO_EXPECTED
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    log "Step 0a: DINO coverage already complete — skipping"
else
    log "Step 0a: Running DINO pairwise coverage (GPU IVF4096, ~1-2 hrs)..."
    python scripts/calculate_coverage_faiss_v2.py \
        --config "$DINO_CONFIG" \
        2>&1 | tee "$LOG_DIR/step0a_dino_coverage.log"
    log "Step 0a: Done"
fi

# ---------------------------------------------------------------------------
# Step 0b: DINO null-calibrated cosine coverage
# Requires DINO vectors and coverage_dino_full.csv from Step 0a. ~5-15 min GPU.
# ---------------------------------------------------------------------------
DINO_NULL_CSV="$ROOT/analysis_v3/dino_null_coverage.csv"
DINO_NULL_EXPECTED=$DINO_EXPECTED
if python3 -c "
import pandas as pd, sys
df = pd.read_csv('$DINO_NULL_CSV')
ok = 'eval_covered_by_train_null90' in df.columns and len(df) >= $DINO_NULL_EXPECTED
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    log "Step 0b: DINO null coverage already complete — skipping"
else
    log "Step 0b: Computing DINO null-calibrated coverage..."
    python scripts/transfer_analysis_v3/compute_dino_null_coverage.py \
        --coverage-csv analysis_v3/coverage_dino_full.csv \
        --vec-dir $VEC_DIR \
        --output analysis_v3/dino_null_coverage.csv \
        --null-percentiles 80 90 95 99 \
        --gpu \
        2>&1 | tee "$LOG_DIR/step0b_dino_null_coverage.log"
    log "Step 0b: Done"
fi

# ---------------------------------------------------------------------------
# Step 0c: Symmetric distribution distances (FID + sliced Wasserstein)
# Resumable — already-computed pairs are skipped automatically.
# ---------------------------------------------------------------------------
SYM_CSV="$ROOT/analysis_v3/symmetric_distances.csv"
N_PAIRS=180
N_EXISTING=0
if [ -f "$SYM_CSV" ]; then
    N_EXISTING=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$SYM_CSV')))" 2>/dev/null || echo 0)
fi
if [ "$N_EXISTING" -ge "$N_PAIRS" ]; then
    log "Step 0c: Symmetric distances already complete ($N_EXISTING rows) — skipping"
else
    log "Step 0c: Computing FID + sliced Wasserstein (CPU, ~5-10 min)..."
    python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
        --flow-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
        --vec-dir $VEC_DIR \
        --output "$SYM_CSV" \
        --n-proj 200 \
        --sw-samples 100000 \
        --fid-samples 200000 \
        2>&1 | tee "$LOG_DIR/step0c_symmetric_distances.log"
    log "Step 0c: Done"
fi

# ---------------------------------------------------------------------------
# Step 0d: All pairwise distances + KL divergences (train-train, eval-eval,
#          train-eval cross pairs). Replaces the separate KL-only coverage run.
#          Used for TensorProductKRR kernels and KL features in the transfer table.
#          ~1-3 hrs GPU depending on space. Resumable — already-computed pairs skipped.
# ---------------------------------------------------------------------------
SELF_DIST_CSV="$ROOT/analysis_v3/pairwise_self_distances.csv"
N_TRAIN=11
# N_EVAL is already set above (9 default, 10 on RC with synthetic/val)
# Rows after symmetrization:
#   train-train: C(11,2)*2 + 11 = 121   eval-eval: C(9,2)*2 + 9 = 81
#   train-eval cross (no symmetrization): 11*9 = 99
#   Total per space: 301 × 2 spaces = 602
SELF_DIST_MIN_ROWS=$(python3 -c "
from math import comb
n_tr, n_ev = ${N_TRAIN}, ${N_EVAL}
tt    = comb(n_tr, 2) * 2 + n_tr
ee    = comb(n_ev, 2) * 2 + n_ev
cross = n_tr * n_ev
print((tt + ee + cross) * 2)
")
N_SELF_DIST=0
if [ -f "$SELF_DIST_CSV" ]; then
    N_SELF_DIST=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$SELF_DIST_CSV')))" 2>/dev/null || echo 0)
fi
if [ "$N_SELF_DIST" -ge "$SELF_DIST_MIN_ROWS" ]; then
    log "Step 0d: Pairwise distances already complete ($N_SELF_DIST rows) — skipping"
else
    log "Step 0d: Computing all pairwise distances + KL (~2-6 hrs GPU)..."
    python scripts/transfer_analysis_v3/compute_pairwise_self_distances.py \
        --vec-dir $VEC_DIR \
        --output "$SELF_DIST_CSV" \
        --max-flow 16000000 \
        --max-dino 8000000 \
        --batch-size 500000 \
        --gpu \
        2>&1 | tee "$LOG_DIR/step0d_pairwise_self_distances.log"
    log "Step 0d: Done"
fi

# ---------------------------------------------------------------------------
# Step 0e: Symmetric self-distances for IDW neighborhoods (flow FID/SW2/MMD).
# Adds flow_fid_self, flow_sliced_w2_self, flow_mmd_self to pairwise_self_distances.csv.
# Resumable and additive; creates pairwise_self_distances.before_symmetric.csv backup.
# ---------------------------------------------------------------------------
if [ "$RUN_SYM_SELF" = "1" ]; then
    SYM_SELF_READY=$(python -c "
import pandas as pd
from pathlib import Path
p=Path('analysis_v3/pairwise_self_distances.csv')
if not p.exists():
    print(0)
else:
    df=pd.read_csv(p)
    cols={'flow_fid_self','flow_sliced_w2_self','flow_mmd_self'}
    if not cols.issubset(df.columns):
        print(0)
    else:
        sub=df[(df.get('space')=='flow') & (df.get('pair_type').isin(['train_train','eval_eval']))]
        ok=len(sub)>0 and sub['flow_fid_self'].notna().all() and sub['flow_sliced_w2_self'].notna().all()
        print(1 if ok else 0)
")
    if [ "$SYM_SELF_READY" = "1" ]; then
        log "Step 0e: Symmetric self-distances already merged — skipping"
    else
        log "Step 0e: Building symmetric self-distances for IDW neighborhoods..."
        python scripts/transfer_analysis_v3/build_symmetric_self_distances.py \
            --self-dist analysis_v3/pairwise_self_distances.csv \
            --output analysis_v3/pairwise_symmetric_distances.csv \
            --vec-dir $VEC_DIR \
            --flow-mmd-csv flow_mmd_results_fast.csv \
            --pair-types train_train eval_eval \
            --fid-samples 200000 \
            --sw-samples 100000 \
            --n-proj 200 \
            2>&1 | tee "$LOG_DIR/step0e_symmetric_self_distances.log"
        log "Step 0e: Done"
    fi
fi

# ---------------------------------------------------------------------------
# Step 1: Assemble transfer table
# Always re-runs (fast, ~5s) so it picks up any new feature files.
# ---------------------------------------------------------------------------
log "Step 1: Assembling transfer table..."
# shellcheck disable=SC2086
python scripts/transfer_analysis_v3/build_table.py \
    $BUILD_TABLE_ARGS \
    2>&1 | tee "$LOG_DIR/step1_build_table.log"
log "Step 1: Done"

# ---------------------------------------------------------------------------
# Step 1b: Feature mutual information (predictive MI + redundancy matrix)
# Fast (~1 min with bootstrap). Always re-runs to pick up new features.
# ---------------------------------------------------------------------------
log "Step 1b: Computing feature MI..."
for TARGET in auc_normalized peak_pck; do
    python scripts/transfer_analysis_v3/compute_feature_mi.py \
        --table scripts/transfer_analysis_v3/transfer_table.csv \
        --target "$TARGET" \
        --out-dir "analysis_v3/feature_mi_${TARGET}" \
        --n-boot 500 --k-neighbors 5 \
        2>&1 | tee "$LOG_DIR/step1b_feature_mi_${TARGET}.log"
done
log "Step 1b: Done"

# ---------------------------------------------------------------------------
# Step 2: Run experiments — both targets
# Results are written incrementally; re-running is safe.
# We run with auc_normalized (average early performance) and peak_pck
# (best-ever performance) so rankings can be compared across target choices.
# ---------------------------------------------------------------------------
SPLITS="loto loto_grouped lobo lomo loco_cell"
if [ "$LOCO" = "1" ]; then SPLITS="$SPLITS loco"; fi

_run_experiments() {
    local TARGET="$1"
    local OUT_DIR="scripts/transfer_analysis_v3/results/${TARGET}"
    local LOG="$LOG_DIR/step2_experiments_${TARGET}.log"
    log "Step 2 (${TARGET}): Running experiments (splits: $SPLITS)..."
    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=1 python scripts/transfer_analysis_v3/run_experiments.py \
        --splits $SPLITS \
        --models ridge bradley_terry plackett_luce kernel_ridge random global_prior \
                 krr_tp_flow_nn krr_tp_flow_eps krr_tp_flow_eps16 \
                 krr_tp_dino_nn krr_tp_dino_eps \
                 ridge_pairwise_nn ridge_pairwise_eps1px ridge_pairwise_eps16px \
                 ridge_pairwise_kl \
        --self-dist-csv analysis_v3/pairwise_self_distances.csv \
        --feature-groups flow_nn flow_eps flow_km flow_kl \
                         dino_nn dino_cov dino_kl \
                         flow_mmd_only dino_mmd_only \
                         flow_fid_only dino_fid_only \
                         flow_w2_only  dino_w2_only  \
                         sym_flow sym_dino \
                         sym_mmd sym_fid sym_w2 \
                         density sample_count vector_density train_profile profile_density \
                         motion motion_km appearance \
                         motion_appearance \
                         flow_mmd_profile flow_fid_profile flow_w2_profile \
                         flow_kl_profile motion_km_profile \
                         all \
        --target "$TARGET" \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$LOG"
    log "Step 2 (${TARGET}): Done"
}

_run_experiments auc_normalized
_run_experiments peak_pck

# ---------------------------------------------------------------------------
# Step 2b: Subsampling stability analysis (~30-60 min GPU)
# ---------------------------------------------------------------------------
STAB_CSV="$ROOT/scripts/transfer_analysis_v3/results/subsampling_stability/stability_table.csv"
if [ -f "$STAB_CSV" ]; then
    log "Step 2b: Subsampling stability already complete — skipping"
else
    log "Step 2b: Running subsampling stability analysis (~30-60 min, GPU)..."
    python scripts/transfer_analysis_v3/run_subsampling_stability.py \
        --coverage-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
        --vec-dir $VEC_DIR \
        --output-dir scripts/transfer_analysis_v3/results/subsampling_stability \
        --gpu \
        --caps 50000 200000 500000 2000000 -1 \
        2>&1 | tee "$LOG_DIR/step2b_subsampling_stability.log"
    log "Step 2b: Done"
fi

# ---------------------------------------------------------------------------
# Step 2b-sym: Symmetric baseline subsampling stability (FID, SW2, ~5-15 min)
# ---------------------------------------------------------------------------
SYM_STAB_CSV="$ROOT/scripts/transfer_analysis_v3/results/subsampling_stability/stability_symmetric.csv"
if [ -f "$SYM_STAB_CSV" ]; then
    log "Step 2b-sym: Symmetric stability already complete — skipping"
else
    log "Step 2b-sym: Running symmetric stability (FID + SW2 vs subsample cap)..."
    python scripts/transfer_analysis_v3/run_symmetric_stability.py \
        --sym-csv analysis_v3/symmetric_distances.csv \
        --vec-dir $VEC_DIR \
        --output-dir scripts/transfer_analysis_v3/results/subsampling_stability \
        --caps 10000 25000 50000 100000 \
        --n-proj 200 \
        2>&1 | tee "$LOG_DIR/step2b_sym_stability.log"
    log "Step 2b-sym: Done"
fi

# ---------------------------------------------------------------------------
# Step 2c: Few-shot mixed-effects analysis (~1-2 min)
# ---------------------------------------------------------------------------
FEW_SHOT_CURVE="$ROOT/scripts/transfer_analysis_v3/results/few_shot/few_shot_learning_curve.csv"
if [ -f "$FEW_SHOT_CURVE" ]; then
    log "Step 2c: Few-shot analysis already complete — skipping"
else
    log "Step 2c: Running few-shot mixed-effects analysis..."
    python scripts/transfer_analysis_v3/run_few_shot_analysis.py \
        --table scripts/transfer_analysis_v3/transfer_table.csv \
        --output-dir scripts/transfer_analysis_v3/results/few_shot \
        --k-values 0 1 2 5 \
        --feature-groups motion motion_appearance all \
        2>&1 | tee "$LOG_DIR/step2c_few_shot.log"
    log "Step 2c: Done"
fi

# ---------------------------------------------------------------------------
# Step 3: Compile results → results.md (one report per target)
# ---------------------------------------------------------------------------
for TARGET in auc_normalized peak_pck; do
    log "Step 3 (${TARGET}): Compiling results report..."
    python scripts/transfer_analysis_v3/compile_results.py \
        --results-dir "scripts/transfer_analysis_v3/results/${TARGET}" \
        --output "scripts/transfer_analysis_v3/results/${TARGET}/results.md" \
        --mi-csv "analysis_v3/feature_mi_${TARGET}/feature_mi.csv" \
        2>&1 | tee "$LOG_DIR/step3_compile_${TARGET}.log"
    log "Step 3 (${TARGET}): Done"
done

log "Pipeline complete."
log "  auc_normalized results : scripts/transfer_analysis_v3/results/auc_normalized/results.md"
log "  peak_pck results       : scripts/transfer_analysis_v3/results/peak_pck/results.md"
log "  Feature MI             : analysis_v3/feature_mi.csv"
log "  Stability CSV          : scripts/transfer_analysis_v3/results/subsampling_stability/stability_table.csv"
log "  Few-shot CSV           : scripts/transfer_analysis_v3/results/few_shot/few_shot_learning_curve.csv"
