# Theoretical Justification for Normalization in Coverage Metrics

## The Coverage Question

You're computing **directed coverage metrics** between training and evaluation sets:

```
Coverage(Train → Eval) = P(eval point is within ε of train | eval)  [Recall]
Coverage(Eval → Train) = P(train point is within ε of eval | train) [Precision]
```

The normalization defines the **metric space** where "within ε" is measured.

---

## Manifold Learning Perspective

From a manifold perspective, your flow vectors lie on some underlying manifold M in R^4.
The coverage metric asks: "What fraction of eval manifold is covered by ε-neighborhoods of train manifold?"

### The Coordinate System Problem

Flow vectors have heterogeneous dimensions:
- **(x, y)**: Pixel coordinates [0, 512] 
- **(dx, dy)**: Flow displacements [typically -50, +50]

Without normalization, L2 distance is **not perceptually uniform**:
```
d₁ = distance between (100,200,5,10) and (101,200,5,10)  # 1 pixel apart in x
d₂ = distance between (100,200,5,10) and (100,200,6,10)  # 1 pixel apart in dx

d₁ = 1.0
d₂ = 1.0
```

But these represent **very different perceptual differences**:
- d₁: Same flow pattern, slightly different location
- d₂: Same location, different flow pattern (100% change if typical dx ≈ 1)

### Z-Score Normalization as Riemannian Metric

Z-score normalization defines a **Mahalanobis distance** with diagonal covariance:

```python
d(v₁, v₂) = √(Σᵢ ((v₁ᵢ - v₂ᵢ) / σᵢ)²)
```

where σᵢ is the standard deviation of dimension i. This makes 1 unit = 1 stddev in each dimension.

**Key question**: Whose σᵢ should we use?

---

## Six Normalization Strategies

### 0. Global Block α-Balancing (CURRENT) ✅

```python
# Compute global α from all train sets (equal-weighted)
α = mean_i(MAD([x, y])_i) / (mean_i(MAD([dx, dy])_i) + ε)

# Apply to all vectors
v' = [x, y, α·dx, α·dy]
```

**Theoretical Interpretation:**
- Single global metric for all comparisons: everyone uses the same α
- Balances position vs. flow to have equal "energy" across datasets
- Uses robust scale (MAD) to avoid outlier sensitivity
- Each dataset contributes equally regardless of size

**When to use:**
- ✅ **Primary use case**: Cross-train comparisons
- ✅ Consistent metric space for all pairs
- ✅ No per-train coordinate systems
- ✅ Simple, interpretable, and robust

**Advantages over train_zscore:**
- ✅ All train sets directly comparable (same metric)
- ✅ No "coordinate system per train" confusion
- ✅ Faster (compute α once, not per-pair)
- ✅ Self-radius caching is globally reusable

**How it differs from full z-score:**
- Only scales dimensions, doesn't shift means (keeps interpretability)
- Block structure: position [x,y] and flow [dx,dy] treated as units
- Global: one α for all, not train-specific whitening

---

### 1. Train-Only Z-Score Normalization ⚠️

```python
μ, σ = train.mean(axis=0), train.std(axis=0)
train_norm = (train - μ) / σ
eval_norm = (eval - μ) / σ
```

**Theoretical Interpretation:**
- Metric space defined by training distribution
- Measures eval in units of "training stddevs from training mean"
- **Answers**: "Is eval within the natural variation of train?"

**When to use:**
- ✅ Primary use case: Generalization evaluation
- ✅ No data leakage (clean ML evaluation)
- ✅ Each train set evaluated in its own coordinate system

**Limitations:**
- Train sets with different variances not directly comparable
- If eval has very different σ, might appear artificially close/far

---

### 2. Eval-Only Normalization ❌

```python
μ, σ = eval.mean(axis=0), eval.std(axis=0)
```

**Theoretical Interpretation:**
- Reverses the question to "Is train within eval's natural variation?"
- Doesn't match the coverage question

**When to use:** Never for coverage metrics

---

### 3. Separate Normalization (Train by Train, Eval by Eval) ❌

```python
train_norm = (train - train.mean()) / train.std()
eval_norm = (eval - eval.mean()) / eval.std()
```

**Theoretical Issue: Breaks the Metric Space!**

This is equivalent to:
1. Normalizing train in coordinate system A
2. Normalizing eval in coordinate system B
3. Measuring distances between points in **different coordinate systems**

Example:
```python
# Train has high variance in dx (σ_dx = 20)
train_point = [0, 0, 20, 0] → normalized: [0, 0, 1.0, 0]

# Eval has low variance in dx (σ_dx = 5)  
eval_point = [0, 0, 20, 0] → normalized: [0, 0, 4.0, 0]

# Same point in original space, but distance in normalized space = 3.0!
```

**Mathematical proof this is wrong:**
```
Let T: R^d → R^d be normalization T(v) = (v - μ) / σ
If T_train ≠ T_eval, then:
  d(T_train(v₁), T_eval(v₂)) ≠ any valid metric on the original manifold
```

**When to use:** Never

---

### 4. Combined Train+Eval Normalization 🔄

```python
combined = np.vstack([train, eval])
μ, σ = combined.mean(axis=0), combined.std(axis=0)
train_norm = (train - μ) / σ
eval_norm = (eval - μ) / σ
```

**Theoretical Interpretation:**
- Metric space defined by joint distribution
- Measures coverage in a "neutral" coordinate system
- **Answers**: "How similar are train and eval distributions?"

**When to use:**
- ✅ Symmetric similarity metrics (e.g., KL divergence)
- ✅ When train/eval distinction is arbitrary
- ❌ Not for generalization eval (data leakage)

---

### 5. Global Train Normalization (ALL TRAINS COMBINED) 🌐

```python
all_trains = np.vstack([flyingthings_train, sintel_train, synthetic_train, ...])
μ, σ = all_trains.mean(axis=0), all_trains.std(axis=0)

# For each train/eval pair:
train_norm = (train - μ) / σ
eval_norm = (eval - μ) / σ
```

**Theoretical Interpretation:**
- "Universal" coordinate system based on aggregate training distribution
- Measures all sets in units of "overall training stddevs"
- **Answers**: "How do different train sets compare at covering eval?"

**When to use:**
- ✅ Comparing multiple training sets on same eval
- ✅ Ranking training sets by coverage
- ✅ When you want consistent metrics across experiments

**Trade-offs:**
- More stable σ estimates (larger sample)
- Can compare train sets directly
- But: specialized train sets lose their unique scaling

---

## Recommendation for Your Use Case

### For Cross-Train Comparisons: **Global Block α-Balancing (Option 0)** ✅ RECOMMENDED

Use the current `mode: global_block_alpha`:

```yaml
normalization:
  mode: global_block_alpha
  apply_to: [flow]
```

**Rationale:**
1. **Primary question**: "Which training set provides best coverage of eval?"
2. **Consistent metric**: All train sets measured in same coordinate system
3. **Interpretable**: α balances position and flow scales
4. **Robust**: MAD-based, equal-weighted per dataset
5. **Efficient**: Compute once, cache globally

### Alternative: Per-Train Z-Score (Option 1) ⚠️ 

Use `mode: train_zscore` if you need:
- Each train set evaluated in its own natural scale
- Per-train "how many stddevs" interpretation
- But: Makes cross-train comparisons difficult!

```yaml
normalization:
  mode: train_zscore
  apply_to: [flow]
```

**When NOT to use:**
- ❌ When comparing multiple train sets on same eval
- ❌ When you need consistent metrics across experiments

---

## Implementation Status

### ✅ Currently Implemented:
- `mode: global_block_alpha` (Option 0) - **DEFAULT**
- `mode: train_zscore` (Option 1)
- `mode: none`

### 📊 Example Usage:

**Cross-training comparison (RECOMMENDED):**
```bash
# "Which train set has best coverage of test sets?"
python scripts/calculate_coverage_faiss.py \
  --config configs/coverage_faiss_flow_full.yaml
# Uses global_block_alpha → consistent metric, cross-train comparable
```

**Per-train evaluation (alternative):**
```bash
# "Does this specific train set generalize to eval?"
python scripts/calculate_coverage_faiss.py \
  --config configs/coverage_faiss_flow_pairwise.yaml
# Set mode: train_zscore in config → per-train coordinate system
```

---

## Mathematical Formalization

### Directed Coverage with Normalization

Given training set T = {t₁, ..., tₙ} and evaluation set E = {e₁, ..., eₘ}:

1. **Define metric space** via normalization N:
   ```
   N_μ,σ(v) = (v - μ) / σ
   ```

2. **Compute support radius** for train set:
   ```
   r = quantile({min_j≠i ||N(tᵢ) - N(tⱼ)||₂}, q=0.95)
   ```

3. **Coverage metric**:
   ```
   Coverage(T → E) = (1/m) Σᵢ 𝟙[min_j ||N(eᵢ) - N(tⱼ)||₂ ≤ r]
   ```

### Choice of (μ, σ) determines the metric:

- **Train-only**: (μ, σ) = moments(T)
  - Natural for: "Is E within T's support?"
  
- **Global**: (μ, σ) = moments(T₁ ∪ T₂ ∪ ... ∪ Tₖ)
  - Natural for: "Which Tᵢ has best coverage of E?"

---

## Summary Table

| Mode | Normalization | No Leakage | Comparable Across Train Sets | Use Case |
|------|---------------|------------|------------------------------|----------|
| **global_block_alpha** | α from all trains (equal-weighted MAD) | ✅ | ✅ | ✅ **Cross-train comparison** (RECOMMENDED) |
| **train_zscore** | Each train's μ, σ | ✅ | ❌ | Per-train generalization eval |
| **global_train** | All trains' μ, σ | ✅ | ✅ | Alternative global metric |
| **combined** | train+eval μ, σ | ❌ | ✅ | Similarity metrics (not coverage) |
| **separate** | Each separate μ, σ | ✅ | ❌ | ⚠️ Invalid! |

---

## References

- Mahalanobis distance and non-isotropic metrics
- Manifold learning and geodesic distances  
- Batch normalization and domain shift
- Coverage metrics in statistical learning theory
