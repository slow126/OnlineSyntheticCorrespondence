# FAISS-Based Manifold Coverage Method

## Overview
We compute pairwise manifold coverage metrics between training and evaluation datasets using approximate nearest neighbors (ANN) on FAISS GPU indices. The method quantifies how well one dataset's manifold "covers" another's, with careful normalization to enable cross-dataset comparison.

---

## Flow Vector Representation
Each correspondence is represented as a 4D vector:
- **v = [x, y, dx, dy]**
  - (x, y): normalized pixel coordinates in [-1, 1]
  - (dx, dy): optical flow displacement in pixels

**Key Challenge**: Position and flow have different scales and units, making direct L2 distance misleading.

---

## Normalization: Global Block α-Balancing

### Problem with Per-Train Z-Scoring
Initial approach used per-train-set z-scoring (whitening):
- Each train set defines its own coordinate system via its covariance Σ
- Eval sets are transformed by train's statistics: z = Σ_train^(-1/2) · (v - μ_train)
- **Issue**: Cross-train comparisons become meaningless—each train set has a different metric!

### Solution: Global α Parameter
Use a **single** distance metric across all datasets:
- **v' = [x, y, α·dx, α·dy]**
- α balances position vs. flow to give them equal "energy" across the dataset collection

### Computing α (Equal-Weighted Across Datasets)
```
For each train dataset i:
  s_i^pos = MAD([x, y])    # Median Absolute Deviation of position
  s_i^flow = MAD([dx, dy])  # MAD of flow

α = mean_i(s_i^pos) / (mean_i(s_i^flow) + ε)
```

- **MAD (Median Absolute Deviation)**: Robust scale estimator = median(|v - median(v)|) / 0.6745
- **Equal weighting**: Each dataset contributes equally to α, regardless of size
- **Result**: Position and flow have comparable variance across the pooled datasets

---

## Coverage Metrics via Self-Radius

### Self-Radius Concept
Each dataset defines its own characteristic scale via k-NN distances:
- For each point, find distance to k-th nearest neighbor (k=5)
- Self-radius = 95th percentile of these k-NN distances
- Cached per dataset to avoid recomputation

### Directed Coverage (Recall & Precision)
**Recall (Train → Eval)**: How much of eval is covered by train?
```
For each eval point:
  d = distance to nearest train point (1-NN)
  covered = (d ≤ train_radius)
Recall = fraction of eval points covered
```

**Precision (Eval → Train)**: How much of train is covered by eval?
```
For each train point:
  d = distance to nearest eval point (1-NN)
  covered = (d ≤ eval_radius)
Precision = fraction of train points covered
```

**Key Insight**: Eval is evaluated against train's radius (and vice versa). This is scale-aware and asymmetric by design.

---

## FAISS Implementation Details

### Index Configuration
- **Small datasets (<164K vectors)**: `Flat` (exact search)
- **Large datasets**: `IVF4096,Flat` with nprobe=64 (99%+ recall)
  - 4096 Voronoi cells for coarse quantization
  - Flat storage within cells (no compression)

### GPU Memory Management
- Single shared `StandardGpuResources` per train/eval pair
- 18GB temp memory allocation
- Adaptive batch sizing:
  - Flow (4D): 500K vectors/batch
  - ResNet/DINO (256D): ~31K vectors/batch (scales as 1/√dim)

### Caching Strategy
- **Vectors**: Cached to disk (16GB+ savings)
- **Global α**: Cached once per representation (reused across all runs)
- **Self-radii**: Globally cached for all datasets (train & eval)
  - With `global_block_alpha`: Same α for everyone → fully reusable
  - With `train_zscore`: Eval radii train-specific → limited reuse
- **Results**: Incremental CSV checkpointing for long runs

**Key benefit of global α**: On 2nd+ runs, both train and eval self-radii are cached globally, requiring only 1-NN searches (fast!).

---

## KL Divergence (Experimental)
k-NN based Kozachenko-Leonenko estimator:
```
KL(P||Q) ≈ d/n · Σ log(ρ_Q(i) / ρ_P(i)) + log(m/n)
where ρ_X(i) = distance to k-th NN in dataset X
```

- Uses k=20 neighbors
- **Known issues**:
  - Negative KL for tiny datasets (<5K) due to estimation bias
  - High asymmetry for dissimilar datasets (expected)
  - Sensitive to manifold dimension mismatch

---

## Key Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Coverage k | 5 | 1-NN to 5th-NN for robust neighbor finding |
| Self-radius k | 5 | Robust scale estimation (less noise than k=1) |
| Radius quantile | 95% | Covers bulk of manifold, excludes outliers |
| α computation | MAD | Robust to outliers vs. std |
| KL k | 20 | Standard for k-NN density estimation |
| FAISS nprobe | 64 | 99%+ recall on IVF4096 |

---

## Interpretation Guide

### High Recall (Train → Eval)
Train manifold densely covers eval → good generalization potential

### High Precision (Eval → Train)  
Eval manifold covers train → no "out-of-distribution" eval points

### Low Recall, Low Precision
Datasets are disjoint → poor transfer expected

### High Recall, Low Precision
Train over-covers eval, but eval has novel regions → train may be too broad

---

## Theoretical Justification

1. **Global α**: Ensures cross-train metric consistency (same Mahalanobis-like scale)
2. **Self-radius normalization**: Makes coverage scale-aware and data-dependent
3. **Asymmetric coverage**: Directed metrics (train→eval ≠ eval→train) capture transfer direction
4. **k-NN robustness**: Using k=5 reduces noise from single outliers

---

## Computational Cost
- **Flow (4D)**: ~30 seconds per train/eval pair (11M × 14M vectors)
- **ResNet (256D)**: ~2-3 minutes per pair (memory-bound)
- **Total**: ~207 pairs × 30s = ~2 hours for full flow coverage matrix (on RTX 3090)
