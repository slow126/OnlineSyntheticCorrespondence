# Flow Visualization Quick Reference

## 🎯 Start Here

**Want the most interpretable visualizations?**
```bash
./run_alternative_vis.sh
```
Creates 6-panel figures showing vectors, magnitudes, directions, and stats.

---

## All Available Visualizations

### 1️⃣ Alternative Visualizations (BEST FOR UNDERSTANDING)
```bash
./run_alternative_vis.sh
```

**What you get:** 6 panels per dataset
- Vector field (quiver plot) - see actual flow vectors
- Magnitude heatmap - spatial intensity
- Direction rose - circular histogram of angles
- Magnitude distribution - with mean/median markers
- Flow space hexbin - 2D density in (dx, dy)
- Statistics summary - key numbers

**Best for:** Understanding what's happening in each dataset

---

### 2️⃣ Side-by-Side Comparison
```bash
./run_comparison_vis.sh
```

**What you get:** One figure with datasets in rows
- Compare magnitude distributions
- Compare direction patterns
- Compare flow space densities
- Compare statistics

**Best for:** Identifying differences between datasets

**Customize:** Edit `run_comparison_vis.sh` to select which datasets

---

### 3️⃣ Gaussian Splat (ARTISTIC)
```bash
./run_splat_vis.sh
```

**What you get:** Abstract splat visualizations
- Endpoint footprint (clustered Gaussians)
- Flow-space density (blurred histogram)

**Best for:** Pretty pictures, less interpretable

---

### 4️⃣ Aggregated View
```bash
./run_aggregated_splat.sh
```

**What you get:** Single figure combining ALL datasets
- Shows overall trends across entire corpus

**Best for:** Big picture view

---

## Quick Customization

### Change which datasets to visualize:

Edit the `PATTERN` variable in the scripts:
```bash
PATTERN="*_train_flow.npy"    # Only training sets
PATTERN="*_test_flow.npy"     # Only test sets
PATTERN="kitti*.npy"          # Only KITTI datasets
```

### Compare specific datasets:

Edit `run_comparison_vis.sh`:
```bash
DATASETS="dataset1.npy,dataset2.npy,dataset3.npy"
```

---

## File Locations

**Input:** `/mnt/nvme_1tb_b/coverage_vectors/`

**Output:**
- Alternative vis: `./alternative_vis/`
- Comparison: `./flow_comparison.png`
- Gaussian splats: `./output_splats/`
- Aggregated: `./aggregated_flow_splat.png`

---

## What Each Visualization Tells You

| Visualization | What It Shows | When To Use |
|--------------|---------------|-------------|
| **Quiver plot** | Actual vectors in space | Understanding spatial flow patterns |
| **Magnitude heatmap** | Where motion is strongest | Finding regions of high activity |
| **Direction rose** | Predominant flow directions | Understanding motion bias |
| **Magnitude histogram** | Distribution of speeds | Checking for outliers, typical magnitudes |
| **Flow space hexbin** | (dx,dy) distribution | Understanding displacement patterns |
| **Statistics** | Numerical summary | Quick facts, paper numbers |

---

## Troubleshooting

**Script not found?**
```bash
cd /home/spencer/Projects/OnlineSyntheticCorrespondence/gaussian_splat
```

**Want higher quality?**
- Edit script, increase `SUBSAMPLE` and `DPI`

**Too slow?**
- Edit script, decrease `SUBSAMPLE`

**Need different datasets?**
- Edit `PATTERN` variable in script
