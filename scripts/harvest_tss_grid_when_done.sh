#!/usr/bin/env bash
# Wait for the TSS grid driver to finish all 6 cells, then harvest the comparison
# table to COMPARISON.txt. Runs unattended via nohup.
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
PY=/home/spencer/miniconda3/envs/cuda/bin/python
LOG=/mnt/nvme_1tb_a/snapshots/tssgrid_logs
DRIVER="$LOG/driver.log"
OUT="$LOG/COMPARISON.txt"
WLOG="$LOG/harvest_watcher.log"
mkdir -p "$LOG"
echo "[$(date +%F_%H:%M:%S)] harvest watcher started (pid $$); waiting for 6 cells" >> "$WLOG"

# up to ~12h; cells finish well before that
for i in $(seq 1 360); do
  if grep -q "TSS GRID DONE" "$DRIVER" 2>/dev/null; then break; fi
  ndone=$(grep -c "DONE " "$DRIVER" 2>/dev/null || echo 0)
  if [ "$ndone" -ge 6 ]; then break; fi
  sleep 120
done
sleep 30
echo "[$(date +%F_%H:%M:%S)] grid finished; harvesting" >> "$WLOG"
$PY scripts/harvest_tss_grid.py > "$OUT" 2>&1
echo "[$(date +%F_%H:%M:%S)] wrote comparison to $OUT" >> "$WLOG"
cat "$OUT" >> "$WLOG"
