# FAISS-Based Manifold Coverage Metrics

## Overview
We quantify how well synthetic training datasets cover real evaluation datasets in feature space using efficient GPU-accelerated nearest neighbor search (FAISS). This approach scales to millions of vectors across flow, ResNet, and DINO representations.

## Methodology

### 1. Feature Normalization (Z-scoring)
**Strategy**: Train-set standardization to prevent data leakage
- Compute mean μ and std σ from the **training set only** (per dimension)
- Apply the same (μ, σ) to normalize both train and eval sets
- Formula: `z = (x - μ_train) / (σ_train + ε)` where ε = 1e-6
- **Rationale**: Simulates deployment where eval distribution is unknown; ensures fair comparison of distances

### 2. Manifold Coverage Metrics (Recall & Precision)

#### Self-Radius Calculation
For each dataset, compute its intrinsic density scale:
1. Build k-NN index (k=5 for self-radius)
2. Find k-th nearest neighbor distance for every point
3. Self-radius R = 95th percentile of these distances

**Interpretation**: R characterizes the typical spacing between points in the dataset manifold.

#### Recall (Train → Eval Coverage)
Measures how well the training set "covers" the evaluation manifold:
```
Recall = fraction of eval points within R_eval of their nearest train point
```
- High recall (>0.9): Train densely covers eval regions
- Low recall (<0.5): Eval contains novel regions not seen in training

#### Precision (Eval → Train Coverage)  
Measures how focused the training set is on eval-relevant regions:
```
Precision = fraction of train points within R_train of their nearest eval point
```
- High precision (>0.9): Train is tightly focused on eval manifold
- Low precision (<0.3): Train contains many regions far from eval (may be useful for other tasks)

**Key insight**: Normalizing by self-radii makes metrics scale-invariant and accounts for varying dataset densities.

### 3. KL Divergence Estimation

We compute two KL divergences to measure distribution mismatch:
- `D_KL(P_eval || P_train)`: How much eval diverges from train
- `D_KL(P_train || P_eval)`: How much train diverges from eval

#### Primary Method: k-NN Estimator (Kozachenko-Leonenko)
Uses k=20 nearest neighbors to estimate local densities:
```
D_KL(P||Q) ≈ d/n · Σ log(ρ_k^Q(x) / ρ_k^P(x)) + log(m/(n-1))
```
Where ρ_k is the distance to the k-th nearest neighbor.

**Advantages**: 
- No binning required
- Adapts to local density

**Limitations**: 
- Numerically unstable for tiny datasets (<5k points)
- Can produce negative values due to finite sample effects
- Sensitive to outliers

#### Fallback Method: Histogram-based KL
For small datasets, we also compute:
```
D_KL(P||Q) = Σ P(i) · log(P(i) / Q(i))
```
Using 50 bins per dimension (with Laplace smoothing ε=1e-6).

**More stable** but loses information in high dimensions.

### 4. Implementation Details

#### FAISS Index Configuration
- **Large datasets (>160k vectors)**: `IVF4096,Flat` with nprobe=64
  - Inverted file index for speed (~100× faster than brute force)
  - Exact search within clusters
- **Small datasets (<160k vectors)**: `Flat` 
  - Automatic fallback for exact brute-force search
  - Required for datasets like pfpascal (2.4k vectors)

#### GPU Memory Management
- RTX 3090 (24GB VRAM)
- Shared GPU resources: 18GB temp memory
- Adaptive batching:
  - 4D flow features: 500k vectors/batch
  - 256D ResNet/DINO (post-PCA): 50k vectors/batch
- Prevents OOM while maximizing GPU utilization

#### Caching Strategy
- **Vectors**: Cached on disk (one-time computation)
- **Train self-radii**: Cached and reused across all eval sets
- **Eval self-radii**: Cached per (eval, train) pair due to train-specific z-scoring
- **Results**: Incremental checkpointing (resume on crash)

## Use Cases

1. **Dataset Selection**: Identify which synthetic datasets best cover target eval sets
2. **Gap Analysis**: Find eval regions not covered by training (low recall)
3. **Efficiency**: Detect redundant training data far from eval manifold (low precision)
4. **Distribution Shift**: Quantify train-eval mismatch via KL divergence

## Key Differences from Other Approaches

- **vs. MMD**: Coverage metrics are asymmetric and interpretable (fraction of points covered)
- **vs. FID**: Works with any representation (not just generative models); provides directional coverage info
- **vs. Simple NN distance**: Normalized by self-radius for scale invariance across datasets

## Interpretation Guidelines

| Recall | Precision | Interpretation |
|--------|-----------|----------------|
| High   | High      | Excellent match: train tightly covers eval |
| High   | Low       | Train covers eval but includes much more |
| Low    | High      | Train is focused but missing eval regions |
| Low    | Low       | Large distribution mismatch |

**KL Divergence**: Use histogram-based values for datasets <5k points; k-NN values can be unreliable (even negative) for small samples.

---

## References
- FAISS: Johnson et al., "Billion-scale similarity search with GPUs" (2019)
- K-NN KL estimator: Kozachenko & Leonenko (1987), Pérez-Cruz (2008)
- Manifold precision/recall: Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models" (2019)
