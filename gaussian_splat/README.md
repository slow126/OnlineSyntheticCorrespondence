# Flow Visualization Toolkit

Multiple visualization methods for analyzing optical flow vector distributions. Choose the visualization style that best suits your analysis needs.

## Visualization Methods

### 1. Alternative Visualizations (RECOMMENDED - Most Interpretable)

**Script:** `visualize_flow_alternatives.py`

Creates 6-panel figures for each dataset with:
- **Vector Field (Quiver)**: Actual flow vectors with color-coded magnitude
- **Magnitude Heatmap**: Spatial distribution of average flow magnitude
- **Direction Rose**: Circular histogram showing flow directions
- **Magnitude Distribution**: Histogram with mean/median/P95 markers
- **Flow Space Hexbin**: 2D density plot in (dx, dy) space
- **Statistics Summary**: Key numerical statistics

### 2. Side-by-Side Comparison

**Script:** `visualize_flow_comparison.py`

Compare multiple datasets in a grid layout with:
- Magnitude distributions
- Direction roses
- Flow space hexbins
- Statistics for each dataset

### 3. Gaussian Splat Visualization

**Script:** `visualize_flow_splats.py`

Gaussian splatting-inspired visualizations:
- **Endpoint Footprint**: Clusters flow endpoints using K-Means, renders as elliptical Gaussian splats
- **Flow-space Density**: 2D histogram over (dx,dy) with Gaussian blur

## Quick Start

### Option 1: Alternative Visualizations (RECOMMENDED)

Most interpretable, shows multiple views per dataset:

```bash
cd gaussian_splat
./run_alternative_vis.sh
```

Creates detailed 6-panel figures in `./alternative_vis/`

### Option 2: Compare Datasets Side-by-Side

Compare 4 datasets in one figure:

```bash
cd gaussian_splat
./run_comparison_vis.sh
```

Edit the script to select which datasets to compare. Output: `./flow_comparison.png`

### Option 3: Gaussian Splat Visualizations

```bash
cd gaussian_splat
./run_splat_vis.sh
```

Generates abstract Gaussian splat visualizations in `./output_splats/`

### Option 4: Aggregated View Across All Datasets

```bash
cd gaussian_splat
./run_aggregated_splat.sh
```

Combines all datasets into one visualization: `./aggregated_flow_splat.png`

### Custom usage:

#### Alternative visualizations:
```bash
python3 visualize_flow_alternatives.py \
  --input_dir /path/to/flow_vectors \
  --pattern "*.npy" \
  --out_dir /path/to/output \
  --subsample 50000 \
  --quiver_subsample 2000
```

#### Compare specific datasets:
```bash
python3 visualize_flow_comparison.py \
  --input_dir /path/to/flow_vectors \
  --datasets "dataset1.npy,dataset2.npy,dataset3.npy" \
  --out_path comparison.png \
  --subsample 50000
```

#### Gaussian splats:
```bash
python3 visualize_flow_splats.py \
  --input_dir /path/to/flow_vectors \
  --pattern "*.npy" \
  --out_dir /path/to/output \
  --K 800 \
  --subsample 2000000 \
  --dpi 200
```

## Which Visualization Should I Use?

| Use Case | Recommended Script | Why |
|----------|-------------------|-----|
| **Understand a single dataset** | `visualize_flow_alternatives.py` | Shows 6 different views including actual vectors, distributions, and stats |
| **Compare datasets** | `visualize_flow_comparison.py` | Side-by-side comparison makes differences obvious |
| **Quick overview** | `visualize_flow_alternatives.py` | Most interpretable - shows magnitude, direction, spatial patterns |
| **Artistic/abstract view** | `visualize_flow_splats.py` | Beautiful but harder to interpret quantitatively |
| **See overall trends** | `visualize_aggregated_splats.py` | Combines all datasets into one view |

**TL;DR:** Start with `./run_alternative_vis.sh` for the most useful visualizations!

## Input Data Format

The script supports multiple formats for flow vectors, all expecting `[x, y, dx, dy]` structure:

- **`.npy`**: NumPy array, either (N,4) or ragged object array
- **`.npz`**: Compressed NumPy, looks for keys: `flows`, `xydxdy`, `data`, `arr_0`
- **`.pt`**: PyTorch tensor or dict with flow data
- **`.pkl`**: Pickled array-like object

**Ragged arrays** (object arrays of chunks) are automatically concatenated.

## Parameters

### Alternative Visualizations (`visualize_flow_alternatives.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | *required* | Directory containing flow vector files |
| `--pattern` | `*.npy` | Glob pattern to match files |
| `--out_dir` | *required* | Output directory for PNG figures |
| `--subsample` | 50000 | Max vectors for histograms/density plots |
| `--quiver_subsample` | 2000 | Max vectors for quiver plot (keep low for readability) |
| `--dpi` | 150 | Output figure DPI |
| `--seed` | 42 | Random seed |

### Comparison (`visualize_flow_comparison.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | *required* | Directory containing flow vector files |
| `--datasets` | *required* | Comma-separated list of filenames to compare |
| `--out_path` | *required* | Output PNG file path |
| `--subsample` | 50000 | Max vectors per dataset |
| `--dpi` | 150 | Output figure DPI |
| `--seed` | 42 | Random seed |

### Gaussian Splats (`visualize_flow_splats.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | *required* | Directory containing flow vector files |
| `--pattern` | `*.npy` | Glob pattern to match files |
| `--out_dir` | *required* | Output directory for PNG figures |
| `--K` | 800 | Number of Gaussian clusters for endpoint footprint |
| `--subsample` | 2000000 | Max flow vectors to use per dataset (0=no limit) |
| `--height` | *auto* | Image height in pixels (inferred if not specified) |
| `--width` | *auto* | Image width in pixels (inferred if not specified) |
| `--max_radius_px` | 64 | Maximum radius for each Gaussian splat |
| `--flow_bins` | 512 | Histogram bins for flow-space density |
| `--dpi` | 200 | Output figure DPI |
| `--seed` | 42 | Random seed for reproducibility |

## Output

### Alternative Visualizations
For each input file `dataset_name.npy`, generates:
- `{out_dir}/dataset_name_alternatives.png` - 6-panel figure

Example:
```
alternative_vis/
├── flyingthings_train_flow_alternatives.png
├── kitti2015_val_flow_alternatives.png
├── spair_test_flow_alternatives.png
└── ...
```

### Comparison
Single output file showing selected datasets:
- `flow_comparison.png` - Grid layout with N rows (one per dataset)

### Gaussian Splats
For each input file `dataset_name.npy`, generates:
- `{out_dir}/dataset_name_splat.png` - Dual-panel visualization

### Aggregated
Single combined visualization:
- `aggregated_flow_splat.png` - All datasets merged

## Dependencies

```bash
pip install numpy matplotlib scikit-learn
```

Optional (for `.pt` files):
```bash
pip install torch
```

## Implementation Details

### Endpoint Clustering
- Uses MiniBatch K-Means for efficiency on large datasets
- Computes robust 2×2 covariance matrix per cluster
- Regularizes near-singular covariances for stable rendering

### Gaussian Rasterization
- Per-splat rendering with Mahalanobis distance
- 3-sigma extent clipping to avoid excessive computation
- Weight normalization using log(1+count) to prevent cluster size dominance

### Flow-space Density
- Quantile-based range clipping (default: 99.5%) for robustness
- Simple box-blur smoothing (no scipy dependency)
- Log-scale tone mapping for visibility

## Customization

Edit `run_splat_vis.sh` to:
- Change the pattern to visualize specific datasets
- Adjust K (more clusters = finer detail, slower)
- Increase subsample for higher quality (slower)
- Set fixed HEIGHT/WIDTH for consistent aspect ratios

## Example Workflows

### Compare synthetic vs. real-world datasets:
```bash
python3 visualize_flow_comparison.py \
  --input_dir /mnt/nvme_1tb_b/coverage_vectors \
  --datasets "synthetic_train_flow.npy,flyingthings_train_flow.npy,kitti2015_val_flow.npy" \
  --out_path synthetic_vs_real.png
```

### Visualize only test sets:
```bash
python3 visualize_flow_alternatives.py \
  --input_dir /mnt/nvme_1tb_b/coverage_vectors \
  --pattern "*_test_flow.npy" \
  --out_dir ./test_alternatives
```

### High-detail quiver plot for a single dataset:
```bash
python3 visualize_flow_alternatives.py \
  --input_dir /mnt/nvme_1tb_b/coverage_vectors \
  --pattern "kitti2015_val_flow.npy" \
  --out_dir ./detailed \
  --subsample 100000 \
  --quiver_subsample 5000 \
  --dpi 200
```

### Compare all KITTI variants:
```bash
python3 visualize_flow_comparison.py \
  --input_dir /mnt/nvme_1tb_b/coverage_vectors \
  --datasets "kitti2012_val_flow.npy,kitti2015_val_flow.npy" \
  --out_path kitti_comparison.png
```

## Notes

### Alternative Visualizations
- **Most useful for understanding**: Shows actual vectors, distributions, and stats
- **Quiver plot**: Keep subsample low (2000-5000) for readability
- **Fast**: No expensive clustering, generates in ~2-5 seconds per dataset
- **Interpretable**: All 6 panels are straightforward to understand

### Comparison
- **Best for**: Identifying differences between datasets at a glance
- **Tip**: Limit to 3-5 datasets per figure for readability
- **Use case**: Compare train vs test, synthetic vs real, different domains

### Gaussian Splats
- **Artistic**: Beautiful but harder to interpret quantitatively
- **Memory**: Subsampling helps with large datasets. 2M vectors ≈ 32MB RAM
- **Speed**: K-Means is the bottleneck. ~5-10 seconds per dataset with K=800
- **Quality**: Higher K captures finer spatial structure but increases runtime

### General
- **Ragged data**: All scripts automatically handle object arrays from subsampled caches
- **Formats**: Support .npy, .npz, .pt, .pkl files

## Troubleshooting

**"No files found"**: Check `--input_dir` path and `--pattern`
**Out of memory**: Reduce `--subsample` or `--K`
**Slow rendering**: Reduce `--max_radius_px` or `--K`
**Blank images**: Data might be all zeros or NaN - check input files

---

Created for the OnlineSyntheticCorrespondence project.
