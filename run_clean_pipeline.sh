#!/bin/bash
# Clean end-to-end transfer-analysis pipeline on real, current features.
# Features (coverage vectors + pairwise distances) are reused (datasets unchanged);
# everything downstream is rebuilt fresh in one consistent pass. No synthetic values.
set -euo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
OSC=/home/spencer/Projects/OnlineSyntheticCorrespondence
IS=/home/spencer/Projects/interventional-study
OUT=scripts/transfer_analysis_v4/results_glunet_clean
echo "============ CLEAN PIPELINE START ============"

echo; echo "### Stage 0 — input provenance (real, current features) ###"
echo "[features] pairwise_self_distances.csv:"; ls -la analysis_v3/pairwise_self_distances.csv | awk '{print "   ",$5,"bytes,",$6,$7,$8}'
python3 -c "import pandas as pd;d=pd.read_csv('analysis_v3/pairwise_self_distances.csv');print('    rows',len(d),'spaces',d.space.unique().tolist())"
echo "[transfer PCK] auc_results.csv families:"; python3 -c "import csv,collections;r=list(csv.DictReader(open('analysis/leakage_free_flow_kmeans_manifold/auc_results.csv')));print('   ',dict(collections.Counter(x['model_family'] for x in r)))"

echo; echo "### Stage 1 — clean coverage CSV (truncate corrupt 37-field rows; verified lossless for the 11 PURE sources) ###"
python3 -c "
fin='analysis/coverage_v2_flow_only_raw_joint_full.csv'
fout='analysis/coverage_v2_flow_only_raw_joint_full.clean17.csv'
open(fout,'w').writelines(','.join(l.rstrip(chr(10)).split(',')[:17])+chr(10) for l in open(fin))
print('    wrote',fout)"

echo; echo "### Stage 2 — rebuild transfer_table.csv ###"
python scripts/transfer_analysis_v3/build_table.py --flow-raw-csv analysis/coverage_v2_flow_only_raw_joint_full.clean17.csv >/tmp/clean_buildtable.log 2>&1
python3 -c "import csv,collections;r=list(csv.DictReader(open('scripts/transfer_analysis_v3/transfer_table.csv')));print('    transfer_table:',len(r),'rows',dict(collections.Counter(x['model_family'] for x in r)))"

echo; echo "### Stage 3 — refit predictors (motion mean_nn) on the clean table ###"
cd "$IS"
for b in kitti2015 kitti2012 middlebury flyingthings; do
  python full_fit_predictor.py --targets peak_pck --families motion --feature-subset mean_nn --score-benchmarks "$b" >/dev/null 2>&1
  echo "    fit predictor: $b"
done
cd "$OSC"

echo; echo "### Stage 4 — v4 pipeline (experiments -> bootstrap N=200 -> figures -> compile) ###"
rm -rf "$OUT"
TARGETS=peak_pck N_BOOT=200 L_MODE=mixed PURE_ONLY=1 OUT_DIR="$OUT" bash scripts/transfer_analysis_v4/run_v4.sh >/tmp/clean_v4.log 2>&1
echo "    v4 done -> $OUT/summary.csv"

echo; echo "### Stage 5 — cross-architecture consensus (CATs++ x RAFT x GLU-Net) ###"
python scripts/transfer_analysis_v4/regenerate_consensus_csv.py \
  --rows-dir "$OUT/predictions/peak_pck" --min-src 4 --n-boot 500 --seed 0 \
  --out "$OUT/CROSS_ARCHITECTURE_CONSENSUS_ALL_SPLITS.csv" >/tmp/clean_consensus.log 2>&1 && echo "    consensus -> $OUT/CROSS_ARCHITECTURE_CONSENSUS_ALL_SPLITS.csv" || echo "    (consensus step failed; see /tmp/clean_consensus.log)"

echo; echo "============ CLEAN PIPELINE DONE ============"
