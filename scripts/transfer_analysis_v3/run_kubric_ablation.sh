#!/usr/bin/env bash
# Kubric appearance-ablation: extract flow+dino coverage vectors for the two
# materialized datasets (hq trial19 vs lowtex/matte), compute distances, print
# the decoupling table. Runs sequentially — safe to launch once and monitor.
#
#   Launch (survives terminal close, single master log):
#     cd /home/spencer/Projects/OnlineSyntheticCorrespondence
#     nohup bash scripts/transfer_analysis_v3/run_kubric_ablation.sh \
#       > /mnt/nvme_1tb_b/kubric_ablation_master_$(date +%m%d_%H%M).log 2>&1 &
#   Monitor:  tail -f /mnt/nvme_1tb_b/kubric_ablation_master_*.log

set -euo pipefail

# --- config ---------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=1                 # training is on GPU 0
REPO=/home/spencer/Projects/OnlineSyntheticCorrespondence
VEC=/mnt/nvme_1tb_b/coverage_vectors
TS=$(date +%m%d_%H%M)
FLOW_CFG=src/configs/coverage_configs/coverage_faiss_flow_only_raw_joint_full.yaml
DINO_CFG=src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml
PAIRS=scripts/transfer_analysis_v3/kubric_ablation_pairs.csv
SYM_OUT=analysis_v3/symmetric_distances_kubric_ablation.csv
DINO_LOG=/mnt/nvme_1tb_b/extract_kubric_dino_${TS}.log

cd "$REPO"
banner(){ echo; echo "============================================================"; echo ">>> $*"; echo "============================================================"; }

# --- 1. flow coverage vectors (only the 2 new kubric datasets extract) ----
banner "1/5  flow coverage vectors  (GPU ${CUDA_VISIBLE_DEVICES})"
python scripts/calculate_coverage_faiss_v2.py --config "$FLOW_CFG"

# --- 2. dino coverage vectors (heavy ViT-G; runs alone) -------------------
banner "2/5  dino coverage vectors  (GPU ${CUDA_VISIBLE_DEVICES})  -> $DINO_LOG"
python scripts/calculate_coverage_faiss_v2.py --config "$DINO_CFG" 2>&1 | tee "$DINO_LOG"

# sanity: PCA should be LOADED (reused), not refit — else dino dists not comparable
banner "PCA check (expect 'loaded', NOT 'fitting')"
grep -i -E "pca" "$DINO_LOG" | grep -i -E "load|fit|model" || echo "  (no PCA log line found — verify manually)"

# --- 3. confirm the 4 coverage files exist --------------------------------
banner "3/5  verify coverage vectors"
ok=1
for n in kitti2015_hq_trial19 kitti2015_lowtex_matte; do
  for r in flow dino_pca256_l2norm; do
    f="$VEC/${n}_train_${r}.npy"
    if [[ -f "$f" ]]; then echo "  OK  $(ls -lh "$f" | awk '{print $5, $9}')"
    else echo "  MISSING  $f"; ok=0; fi
  done
done
[[ $ok -eq 1 ]] || { echo "ERROR: coverage vectors missing — stopping before distances."; exit 1; }

# --- 4. distances ---------------------------------------------------------
banner "4/5  distances (FID/SW2/MMD + mean_nn/KL, flow & dino)"
cp -n analysis_v3/pairwise_self_distances.csv analysis_v3/pairwise_self_distances.csv.bak_pre_kubric_${TS} || true

python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
  --vec-dir "$VEC" --flow-csv "$PAIRS" --output "$SYM_OUT"

python scripts/transfer_analysis_v3/compute_pairwise_self_distances.py \
  --vec-dir "$VEC" --pair-types train_eval

# --- 5. decoupling table (hq vs lowtex -> kitti2015) ----------------------
banner "5/5  decoupling table"
python3 - "$SYM_OUT" <<'PY'
import sys, pandas as pd
sym = sys.argv[1]
for path,label in [(sym,"SYMMETRIC (FID / SW2 / MMD)"),
                   ("analysis_v3/pairwise_self_distances.csv","PAIRWISE (mean_nn / KL / coverage)")]:
    try: df = pd.read_csv(path)
    except Exception as e: print(f"\n{label}: cannot read {path} ({e})"); continue
    sub = df[(df.train_dataset.isin(["kitti2015_hq_trial19","kitti2015_lowtex_matte"])) &
             (df.eval_dataset=="kitti2015")]
    if sub.empty: print(f"\n{label}: no kitti2015 rows yet"); continue
    cols=[c for c in sub.columns if any(t in c.lower() for t in
          ("flow","dino","mean_nn","kl","fid","w2","mmd","sliced"))]
    print(f"\n=== {label} — train -> kitti2015 ===")
    print(sub.set_index("train_dataset")[cols].T.to_string())
print("\nRead: flow rows hq~=lowtex (motion held) ; dino rows hq!=lowtex (appearance moved).")
PY
banner "DONE"
