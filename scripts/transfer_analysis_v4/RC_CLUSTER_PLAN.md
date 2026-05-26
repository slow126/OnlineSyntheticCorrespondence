# RC Cluster Compute Plan — DINO Symmetric + Density Invariance

Handoff document for running the heavy feature-extraction compute on
`ssh.rc.byu.edu` (user `slow1`). Local already has fresh flow features
and the recent DINO RC pull; this plan adds **DINO FID + SW2** and
**density-invariance recomputes** so that tomorrow we can finalize the
predictor and start the interventional study.

The plan parallelizes everything that can parallelize. Two RC nodes are
reportedly down so available compute may be limited; the plan is
**partitioned into independent jobs** so we can run on whatever nodes are
available, including a single node serially.

---

## Step 1 — Verify cluster state (do this first)

Have codex SSH and run these audit commands. Expected outputs in parentheses.

```bash
# A. Confirm session
ssh slow1@ssh.rc.byu.edu hostname

# B. Project dir
ssh slow1@ssh.rc.byu.edu '
    cd /home/slow1/Projects/OnlineSyntheticCorrespondence && \
    pwd && \
    git status -uno 2>/dev/null | head -5 && \
    ls scripts/transfer_analysis_v3/ | head
'
# Expected: project exists; scripts/transfer_analysis_v3 has the compute scripts

# C. Are flow vectors there?
ssh slow1@ssh.rc.byu.edu '
    ls -d /home/slow1/Projects/OnlineSyntheticCorrespondence/coverage_vectors 2>/dev/null
    find /home/slow1 -name "*_flow.npy" 2>/dev/null | head -5
    find /mnt -name "*_flow.npy" 2>/dev/null | head -5
    find /scratch -name "*_flow.npy" 2>/dev/null | head -5
'
# If empty → need to transfer flow vectors (Step 2A)
# If found → record the directory path, use it as VEC_DIR

# D. Are DINO vectors there?
ssh slow1@ssh.rc.byu.edu '
    find /home/slow1 -name "*_dino_pca256_l2norm.npy" 2>/dev/null | head -5
    find /mnt -name "*_dino_pca256_l2norm.npy" 2>/dev/null | head -5
'
# User says DINO is likely already there from the original RC extraction.
# Confirm by checking ALL 11 train datasets + 10 benchmarks have files.

# E. Required code present?
ssh slow1@ssh.rc.byu.edu '
    cd /home/slow1/Projects/OnlineSyntheticCorrespondence
    test -f scripts/transfer_analysis_v3/compute_symmetric_distances.py && echo "compute_symmetric_distances.py: OK"
    test -f scripts/transfer_analysis_v3/compute_pairwise_self_distances.py && echo "compute_pairwise_self_distances.py: OK"
    test -f analysis/coverage_v2_flow_only_raw_joint_full.csv && echo "flow_csv: OK"  || echo "flow_csv: MISSING"
    test -f analysis_v3/pairwise_self_distances.csv && echo "baseline: OK"            || echo "baseline: MISSING"
'

# F. GPU availability
ssh slow1@ssh.rc.byu.edu 'nvidia-smi -L 2>/dev/null | head -8 || echo "no nvidia-smi"'
# Or via SLURM:
ssh slow1@ssh.rc.byu.edu 'sinfo -p gpu 2>/dev/null | head -10 || echo "no slurm gpu queue visible"'

# G. Available scratch / disk
ssh slow1@ssh.rc.byu.edu 'df -h /scratch /home/slow1 /mnt 2>/dev/null | head -5'
```

Codex should report back which paths exist, which files are present,
which GPUs are available, and how much scratch space is free.

---

## Step 2 — Transfer missing data

### 2A. If flow vectors are NOT on RC

Flow vectors live at `/mnt/nvme_1tb_b/coverage_vectors/*_flow.npy` locally.
There are 25 (training) + 10 (benchmark) = 35 datasets, each with 1-2 GB
of flow vectors. **Total ~30-50 GB** to transfer.

Decide where to put them on RC:
- If RC has `/scratch/slow1/` or similar — use that (fast, ephemeral)
- Otherwise `/home/slow1/coverage_vectors/`

Rsync from local:
```bash
# From local machine (push to RC)
rsync -avz --progress \
    /mnt/nvme_1tb_b/coverage_vectors/*_flow.npy \
    slow1@ssh.rc.byu.edu:/scratch/slow1/coverage_vectors/
```

ETA: 30 GB over typical campus uplink ≈ 1-3 hours.

**If you skip flow transfer**: density invariance for flow has to run
locally. DINO compute (everything below) still runs on RC.

### 2B. If DINO vectors are missing on RC

Similar rsync but for `*_dino_pca256_l2norm.npy`. Total ~5-10 GB; faster.

User believes DINO is already on RC from the original extraction. Audit
in Step 1D confirms.

### 2C. Sync code changes

If RC has an older git state, push current changes:
```bash
# From local
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
git push   # if RC is set up to pull from your remote
# OR scp the new scripts:
scp scripts/transfer_analysis_v4/density_invariance.py \
    scripts/transfer_analysis_v4/density_invariance_train_eval_only.py \
    slow1@ssh.rc.byu.edu:/home/slow1/Projects/OnlineSyntheticCorrespondence/scripts/transfer_analysis_v4/
```

---

## Step 3 — Parallel jobs on RC

Three independent jobs. Each writes to its own output CSV in
`analysis_v3/`. Run as separate SLURM jobs (or as backgrounded screens)
so they parallelize across whatever nodes are available.

### Job A — DINO FID + SW2 (NEW; never computed before)

```bash
# As SLURM job (preferred)
cat > ~/rc_jobs/dino_sym.sh << 'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=dino_sym
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.log

cd /home/slow1/Projects/OnlineSyntheticCorrespondence
python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
    --flow-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
    --vec-dir <SET TO ACTUAL DINO VEC DIR FROM STEP 1D> \
    --output analysis_v3/symmetric_distances_dino.csv \
    --skip-flow \
    --n-proj 200 --sw-samples 100000 --fid-samples 200000
EOF
sbatch ~/rc_jobs/dino_sym.sh
```

**Time: ~1-2 hours.** Output: `analysis_v3/symmetric_distances_dino.csv`
with columns `dino_fid` and `dino_sliced_w2`.

### Job B — Flow density invariance (3 levels)

Skip if flow vectors weren't transferred to RC (do this one locally).

```bash
cat > ~/rc_jobs/density_flow.sh << 'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=density_flow
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.log

cd /home/slow1/Projects/OnlineSyntheticCorrespondence
python scripts/transfer_analysis_v4/density_invariance_train_eval_only.py \
    --space flow \
    --levels 50000 200000 1000000 \
    --vec-dir <SET TO ACTUAL FLOW VEC DIR>
EOF
sbatch ~/rc_jobs/density_flow.sh
```

**Time: ~3-5 hours total** (50k=~10min, 200k=~30min, 1M=~3h).
Output: 3 CSVs under `analysis_v3/density_invariance/pairwise_self_flow_N*.csv`
plus a stability heatmap.

### Job C — DINO density invariance (3 levels)

```bash
cat > ~/rc_jobs/density_dino.sh << 'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=density_dino
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.log

cd /home/slow1/Projects/OnlineSyntheticCorrespondence
python scripts/transfer_analysis_v4/density_invariance_train_eval_only.py \
    --space dino \
    --levels 10000 50000 200000 \
    --vec-dir <SET TO ACTUAL DINO VEC DIR>
EOF
sbatch ~/rc_jobs/density_dino.sh
```

**Time: ~1-2 hours.**
Output: 3 CSVs under `analysis_v3/density_invariance/pairwise_self_dino_N*.csv`
plus heatmap.

### Single-node alternative (no SLURM, run serially)

If you only have one node and SLURM isn't available, run in screen:

```bash
ssh slow1@ssh.rc.byu.edu
screen -S rc_compute
cd /home/slow1/Projects/OnlineSyntheticCorrespondence

# Job A then B then C (~6-9 hours total)
CUDA_VISIBLE_DEVICES=0 python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
    --flow-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
    --vec-dir <DINO_VEC_DIR> \
    --output analysis_v3/symmetric_distances_dino.csv \
    --skip-flow --n-proj 200 --sw-samples 100000 --fid-samples 200000 \
    2>&1 | tee /tmp/dino_sym.log && \
CUDA_VISIBLE_DEVICES=0 python scripts/transfer_analysis_v4/density_invariance_train_eval_only.py \
    --space flow --levels 50000 200000 1000000 --vec-dir <FLOW_VEC_DIR> \
    2>&1 | tee /tmp/density_flow.log && \
CUDA_VISIBLE_DEVICES=0 python scripts/transfer_analysis_v4/density_invariance_train_eval_only.py \
    --space dino --levels 10000 50000 200000 --vec-dir <DINO_VEC_DIR> \
    2>&1 | tee /tmp/density_dino.log

# Detach: Ctrl+A then D. Reattach: screen -r rc_compute
```

---

## Step 4 — Local concurrent work (run while RC is busy)

These don't depend on RC compute and should run locally tonight:

### 4A. The 17-ablation sweep (uses current features; CPU; ~45 min)

```bash
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
ARCHIVE="scripts/transfer_analysis_v4/_archive_pre_rc_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE"
mv scripts/transfer_analysis_v4/results scripts/transfer_analysis_v4/results_* "$ARCHIVE/" 2>/dev/null

for mode in mixed symmetric_informed symmetric_uninformed targeted_informed eb_shrunk density_idw; do
    OUT_DIR=scripts/transfer_analysis_v4/results_${mode} \
    L_MODE=${mode} N_BOOT=500 USE_RANKNET=1 \
        bash scripts/transfer_analysis_v4/run_v4.sh > /tmp/v4_${mode}.log 2>&1 &
done
for sub in mean_nn mean_nn_sym mean_nn_asym coverage eps_1px eps_4px eps_16px \
           kl kl_k5 kl_k20 asym_only; do
    OUT_DIR=scripts/transfer_analysis_v4/results_fsub_${sub} \
    FEATURE_SUBSET=${sub} N_BOOT=500 USE_RANKNET=1 \
        bash scripts/transfer_analysis_v4/run_v4.sh > /tmp/v4_fsub_${sub}.log 2>&1 &
done
wait
python scripts/transfer_analysis_v4/compile_ablation_summary.py
```

The appearance_sym/fid/w2 families gracefully skip until Step 5 lands.

---

## Step 5 — Merge RC outputs back to local

When RC jobs finish (codex monitors via `squeue -u slow1`):

```bash
# 5A. Pull DINO sym from RC
rsync -avz slow1@ssh.rc.byu.edu:/home/slow1/Projects/OnlineSyntheticCorrespondence/analysis_v3/symmetric_distances_dino.csv \
    analysis_v3/symmetric_distances_dino.csv

# 5B. Pull density invariance results
mkdir -p analysis_v3/density_invariance
rsync -avz slow1@ssh.rc.byu.edu:/home/slow1/Projects/OnlineSyntheticCorrespondence/analysis_v3/density_invariance/ \
    analysis_v3/density_invariance/

# 5C. Merge DINO sym into the main symmetric_distances.csv
python -c "
import pandas as pd
a = pd.read_csv('analysis_v3/symmetric_distances.csv')
b = pd.read_csv('analysis_v3/symmetric_distances_dino.csv')
m = a.merge(b, on=['train_dataset','train_split','eval_dataset','eval_split'], how='outer')
m.to_csv('analysis_v3/symmetric_distances.csv', index=False)
print(f'merged: {len(m)} rows, cols: {list(m.columns)}')
"

# 5D. Rebuild transfer_table and re-apply spair patch
python scripts/transfer_analysis_v3/build_table.py
python scripts/transfer_analysis_v4/patch_spair_long.py

# 5E. Re-run the headline mode runs to pick up appearance_sym/fid/w2
for mode in mixed eb_shrunk density_idw; do
    OUT_DIR=scripts/transfer_analysis_v4/results_${mode} \
    L_MODE=${mode} N_BOOT=500 USE_RANKNET=1 \
        bash scripts/transfer_analysis_v4/run_v4.sh > /tmp/v4_post_${mode}.log 2>&1 &
done
wait

# 5F. Regenerate ablation summary
python scripts/transfer_analysis_v4/compile_ablation_summary.py

# 5G. Look at density-invariance recommendation
cat analysis_v3/density_invariance/stability_flow_train_eval.csv
cat analysis_v3/density_invariance/stability_dino_train_eval.csv
# Heatmap PNGs at analysis_v3/density_invariance/stability_heatmap_*.png
```

---

## Step 6 — Final outputs

After all of the above, you have:

| Artifact | Location | Use |
|---|---|---|
| Updated transfer_table.csv | `scripts/transfer_analysis_v3/` | predictor input |
| Full ablation matrix | `scripts/transfer_analysis_v4/results_*/` | paper tables |
| ABLATION.md summary | `scripts/transfer_analysis_v4/ABLATION.md` | paper-prep cross-mode |
| CLAIMS.md (needs manual update) | `scripts/transfer_analysis_v4/` | per-claim evidence |
| Density stability heatmaps | `analysis_v3/density_invariance/*.png` | "minimum N for stable features" — for paper + sets interventional-study sample size |
| DINO sym features | `transfer_table.csv` cols `dino_fid`, `dino_sliced_w2` | enables `appearance_sym` family |

Headline questions answered:
1. **What's the strongest predictor configuration?** → from ABLATION.md (current best is `motion_sym` with FID+SW2+MMD, but `appearance_sym` might compete after Step 5)
2. **How many vectors / frames does the interventional study need per candidate?** → density invariance recommendation
3. **Is `appearance_sym` competitive with `motion_sym`?** → after Step 5 this is finally answerable

---

## Things codex needs to decide / fill in

1. **Which dirs hold the flow and DINO vectors on RC?** Step 1C and 1D
   audits give the answer; the SLURM scripts above have `<SET TO ACTUAL ...>`
   placeholders that need editing.
2. **SLURM partition name** (default assumed `gpu` in Step 3 examples) —
   change to whatever your cluster uses.
3. **If flow vectors aren't on RC**: do you transfer them (Step 2A, several
   hours upload) or run density-invariance-flow locally instead? If local
   has the GPU free, local is faster than upload+compute.
4. **If only one node is available**: run everything serially per the
   "Single-node alternative" in Step 3. Total wall time ~6-9 hours.

---

## Quick estimate matrix

| Scenario | DINO sym | Flow density | DINO density | Total wall (parallel) |
|---|---|---|---|---|
| 3 nodes available | 1.5h | 4h | 1.5h | **~4h** (longest = flow density) |
| 1 node, serial | 1.5h | 4h | 1.5h | **~7h** |
| Flow stays local, DINO on RC | local 4h flow + RC 1.5h DINO sym + RC 1.5h DINO dens | parallel | parallel | **~4h** (longest = local flow) |

---

## What codex should report back to user

After running the RC pieces:

1. Confirmation that each job's output CSV exists and looks sane
   (row counts, no all-NaN columns, etc.)
2. Path to where the level-CSVs landed
3. The "RECOMMENDED minimum N" lines from the density invariance output
4. Total wall time consumed

Then the user runs Step 5 locally (merge + final ablation re-run) to
finalize.
