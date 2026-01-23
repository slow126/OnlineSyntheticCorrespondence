# Coverage Pipeline v2.0 - Module Documentation

This is a modular implementation of the 5-step coverage pipeline with proper caching, squared L2 distances, and multi-space decomposition.

## Pipeline Overview

```
0. Fixed Sampling Protocol (cached vectors)
   └─> Raw vectors cached on disk

1. Alpha Calibration (flow only)
   └─> Global α for balancing [x,y] and [dx,dy]

2. Define Spaces
   ├─> Flow: xy, flow, joint spaces
   └─> Features: single feature space

3. Per-Dataset Self-Radius
   └─> R_D^S for each dataset D in each space S

4. Cross-Dataset NN Distances
   └─> d_E→T and d_T→E for all train/eval pairs

5. Coverage Metrics
   ├─> Dual normalization (qnorm + rnorm)
   ├─> k=[1, 5] metrics
   └─> Optional coverage curves
```

## Module Structure

```
scripts/coverage/
├── __init__.py              # Package initialization
├── cache.py                 # Caching + preprocessing utilities
├── faiss_ops.py             # Core FAISS operations (CRITICAL: squared L2!)
├── spaces.py                # Space transformations (xy, flow, joint)
├── calibration.py           # Alpha calibration (Step 1)
├── radius.py                # Self-radius computation (Step 3)
├── metrics.py               # Coverage metrics (Steps 4-5)
└── README.md                # This file
```

## Key Design Decisions

### 1. **Squared L2 Distances Throughout**

FAISS returns **squared L2 distances** by default. We **do NOT take sqrt** anywhere.

```python
# faiss_ops.py
distances = raw_dists  # Keep squared L2
# NOT: distances = np.sqrt(raw_dists)
```

This ensures consistency between:
- Self-radius computation
- Cross-dataset distances
- Coverage metrics

### 2. **Enhanced Cache Keys**

Cache keys include ALL parameters that affect the cached value:

```python
# Radius cache key includes:
radius_{dataset}_{split}_{space}_norm2x1_sqL2_k5_q0p95[_a{alpha}].npz
                                   ^^^      ^^^^      ^^^   ^^^     ^^^
                              normalization │         k    quantile alpha
                                       distance metric
```

### 3. **Flow Normalization to [-1, 1]**

Flow vectors are normalized to `[-1, 1]` range, centered at zero:

```python
# positions: (x, y) ∈ [0, W-1] × [0, H-1] → [-1, 1] × [-1, 1]
result[:, 0] = (result[:, 0] / (W - 1)) * 2 - 1  # x
result[:, 1] = (result[:, 1] / (H - 1)) * 2 - 1  # y

# flow: (dx, dy) scaled consistently
result[:, 2] = result[:, 2] / W * 2  # dx
result[:, 3] = result[:, 3] / H * 2  # dy
```

### 4. **Dual Normalization Metrics**

Two versions of each coverage metric:

- **Query-normalized (qnorm)**: Use query set's radius
  - `eval_covered_by_train_qnorm = mean(d_E→T <= R_E)`
  - Interpretable: "How much of eval is covered, in eval's own scale?"

- **Reference-normalized (rnorm)**: Use reference set's radius
  - `eval_covered_by_train_rnorm = mean(d_E→T <= R_T)`
  - Diagnostic: "How much of eval is covered, in train's scale?"

### 5. **Coverage Curves**

Robustness check over multiple radius quantiles `{0.80, 0.90, 0.95, 0.99}`:

- Reuses the same distance arrays (no extra FAISS searches)
- Just varies the threshold radius
- Reveals sensitivity to radius choice

### 6. **Robust exclude_self for ANN**

For self-radius with approximate indices:

```python
# Query k+1 neighbors
# If distances[0] ≈ 0, drop it (self-match)
# Otherwise, keep it (ANN didn't return self)
```

## Usage

### Quick Start

Run all three representations (flow, resnet, dino):

```bash
./tmp_run_faiss.sh
```

This will:
1. Load or extract all vectors (with caching)
2. Apply preprocessing (flow normalization or PCA+L2)
3. Compute α (flow only), radii, and coverage metrics
4. Save results to CSV files

### Individual Runs

```bash
# Flow only (with xy, flow, joint decomposition)
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_full_v2.yaml

# ResNet only (with PCA 2048→256 + L2 norm)
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_resnet_v2.yaml

# DINO only (with PCA 4096→256 + L2 norm)
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml
```

## Configuration

### Flow Config Key Parameters

```yaml
representation: flow

spaces:
  enabled: ["xy", "flow", "joint"]  # Multi-space decomposition

calibration:
  enabled: true  # Compute global α
  k: 5
  aggregation: geometric_mean

flow_normalization:
  enabled: true
  scheme: norm2x1  # Map to [-1, 1]
  image_size: [512, 512]

coverage:
  self_radius_k: 5
  radius_quantile: 0.95
  k_max: 5
  k_values: [1, 5]
  compute_curves: true  # Enable coverage curves
  curve_quantiles: [0.80, 0.90, 0.95, 0.99]

faiss:
  index_factory: Flat  # Exact search
  use_gpu: true
  batch_size: 100000  # For 3090 24GB
```

### Feature Config Key Parameters (ResNet/DINO)

```yaml
representation: resnet  # or dino

spaces:
  enabled: ["features"]  # Single space only

pca:
  enabled: true  # REQUIRED for high-D features
  output_dim: 256
  whiten: false
  l2_normalize: true  # CRITICAL: Normalize to unit sphere

faiss:
  batch_size: 50000  # Smaller for 256-D
```

## Cache Structure

```
/mnt/nvme_1tb_b/coverage_vectors/
├── {dataset}_{split}_flow.npy              # Raw flow vectors
├── {dataset}_{split}_dino.npy              # Raw DINO features (4096-D)
├── {dataset}_{split}_resnet.npy            # Raw ResNet features (2048-D)
├── pca_model_dino.pkl                      # Fitted PCA model
├── pca_model_resnet.pkl
├── global_alpha_flow.npz                   # Global α for flow
└── radii/
    ├── radius_{dataset}_{split}_xy_norm2x1_sqL2_k5_q0p95.npz
    ├── radius_{dataset}_{split}_flow_norm2x1_sqL2_k5_q0p95.npz
    ├── radius_{dataset}_{split}_joint_norm2x1_sqL2_k5_q0p95_a{alpha}.npz
    └── radius_{dataset}_{split}_features_norm2x1_sqL2_k5_q0p95.npz
```

## Output

### Main Results CSV

Columns include:
- `space`: xy, flow, joint, or features
- `train_dataset`, `train_split`, `eval_dataset`, `eval_split`
- `train_n_vectors`, `eval_n_vectors`
- `train_radius`, `eval_radius`
- For each k ∈ {1, 5}:
  - `eval_covered_by_train_qnorm_k{k}` ← **Headline metric**
  - `train_covered_by_eval_qnorm_k{k}` ← **Headline metric**
  - `train_outside_eval_qnorm_k{k}` ← **Headline metric**
  - `eval_covered_by_train_rnorm_k{k}` (diagnostic)
  - `train_covered_by_eval_rnorm_k{k}` (diagnostic)
  - `train_outside_eval_rnorm_k{k}` (diagnostic)
  - `mean_nn_eval_to_train_k{k}`, `median_nn_eval_to_train_k{k}`, etc.
  - `mean_nn_train_to_eval_k{k}`, `median_nn_train_to_eval_k{k}`, etc.

### Optional Curves CSV

If `coverage.compute_curves: true`, includes coverage at multiple quantiles.

## Memory & Performance

### 3090 24GB VRAM

- **Flow (4-D)**: Very light, can handle huge batches
  - `batch_size: 100000` works well
  - All 3 spaces (xy, flow, joint) fit comfortably

- **Features (256-D after PCA)**:
  - `batch_size: 50000` recommended
  - Watch VRAM usage during index building

- **Use Flat index**: Exact search is fast enough for up to ~16M vectors in low-D

### Speedup with Caching

**First run**: ~30-60 min
- Extract all vectors
- Compute α
- Compute all radii
- Compute all coverage metrics

**Subsequent runs** (with same datasets/params): ~10-15 min
- Load cached vectors (instant)
- Load cached α (instant)
- Load cached radii (instant)
- Only recompute coverage metrics if needed

**Changing parameters**:
- Change `k` or `quantile` → Recompute radii only
- Change train set list → Recompute α + radii
- Change normalization → Recompute everything (cache keys will differ)

## Troubleshooting

### "Cache key mismatch"
- Cache keys encode all parameters
- Delete old caches if you changed: k, quantile, normalization, distance metric

### "Out of memory"
- Reduce `faiss.batch_size` in config
- Try `IVF` index instead of `Flat` for approximate search

### "Alpha values look wrong"
- Check that flow vectors are properly normalized to [-1, 1]
- Verify image_size in config matches actual image dimensions

### "Coverage values are 0 or 1 everywhere"
- Check that distances and radii are in the same units (squared L2)
- Verify you didn't accidentally take sqrt somewhere

## Testing

To test on a small subset:

```bash
# Edit config to use only 2-3 datasets
# Set sampling.max_vectors: 100000 (smaller sample)
# Run and verify:
#   1. Caching works (second run is much faster)
#   2. Multi-space output appears (for flow)
#   3. Dual normalization metrics are present
#   4. k=1 and k=5 metrics both appear
```

## Comparison to Old Pipeline

| Feature | Old | New |
|---------|-----|-----|
| Modularity | Monolithic (2894 lines) | 6 focused modules |
| Distance metric | Mixed (sometimes sqrt) | **Squared L2 everywhere** |
| Caching | Basic | Enhanced with full metadata |
| Flow spaces | Single joint space | **3 spaces: xy, flow, joint** |
| Metrics | Single "coverage" | **Dual: qnorm + rnorm** |
| k values | Single k | **Multiple: k=[1,5]** |
| Coverage curves | No | **Optional robustness check** |
| Alpha calibration | Per-train | **Global across datasets** |
| Cache keys | Simple | **Include all parameters** |

## References

- FAISS documentation: https://github.com/facebookresearch/faiss
- Squared L2 vs L2: FAISS `IndexFlatL2` returns squared distances
- PCA + L2 norm: Standard preprocessing for high-D features
