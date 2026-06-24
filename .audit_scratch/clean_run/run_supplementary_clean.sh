#!/usr/bin/env bash
# Supplementary clean producers — everything NOT in run_v5.sh that the audit
# flagged as Middlebury-contaminated or default-contaminated. Post table-swap
# (transfer_table.csv is now middlebury-free) these all read clean by default.
# Must run AFTER run_v5.sh (they consume results_rule_v5core/predictions/).
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
V5=scripts/transfer_analysis_v5
RES=$V5/results
log(){ echo "[$(date '+%H:%M:%S')] supp: $*"; }
run(){ log "RUN $*"; "$@" && log "OK" || log "FAIL ($*)"; }

# table+dist driven (contaminated-by-default; clean now)
run python $V5/controls_fresh.py --out $RES/controls_fresh.csv
run python $V5/ceiling_oracles.py --out $RES/ceiling_oracles.csv
# core-rows driven (must follow run_v5 stage 4)
run python $V5/per_regime_linear.py
run python $V5/conditional_combination.py
run python $V5/loao_weight_transfer.py
run python $V5/policy_vs_fit_regret.py
run python $V5/joint_anchor.py
# new audit analyses (already nomid; re-run for a coherent generation)
run python $V5/anticorr_by_benchmark.py
run python $V5/diagonal_sensitivity.py
run python $V5/regime_vs_level_deconfound.py
log "DONE"
