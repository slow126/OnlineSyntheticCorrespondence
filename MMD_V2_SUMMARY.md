# MMD Pipeline v2.0 - Implementation Summary

## Overview
Created a new MMD calculation pipeline aligned with the FAISS coverage pipeline v2.0, ensuring consistent normalization and efficient computation.

## Key Features

### 1. Consistent Normalization with FAISS Coverage
- **Flow**: Uses alpha-scaled joint space `[x, y, α*dx, α*dy]` with global alpha calibration
- **ResNet/DINO**: Uses PCA (2048/4096 → 256 dims) + L2 normalization
- Same preprocessing as coverage pipeline for direct comparability

### 2. Efficient Computation
- Only computes **train→eval** MMD pairs (not train-to-train)
- Significantly reduces computation compared to all-pairs approach
- Matches the evaluation structure of the coverage pipeline

### 3. Shared Caching
- Reuses cached vectors from coverage pipeline (`/mnt/nvme_1tb_b/coverage_vectors`)
- Shares PCA models and alpha calibration
- No redundant vector extraction

### 4. Modular Design
- Reuses `scripts/coverage` modules (cache, calibration, spaces)
- Consistent with coverage pipeline implementation
- Easy to maintain and extend

## Files Created

### Main Script
- `scripts/calculate_mmd_v2.py` - Main MMD pipeline script

### Config Files
- `src/configs/mmd_configs/mmd_flow_v2.yaml` - Flow MMD config (joint space)
- `src/configs/mmd_configs/mmd_resnet_v2.yaml` - ResNet MMD config
- `src/configs/mmd_configs/mmd_dino_v2.yaml` - DINO MMD config

### Runner Script
- `tmp_run_mmd.sh` - Runs all three MMD calculations

## Usage

### Run All MMD Calculations
```bash
bash tmp_run_mmd.sh
```

### Run Individual Representations
```bash
# Flow (with alpha normalization)
python scripts/calculate_mmd_v2.py --config src/configs/mmd_configs/mmd_flow_v2.yaml

# ResNet (with PCA + L2)
python scripts/calculate_mmd_v2.py --config src/configs/mmd_configs/mmd_resnet_v2.yaml

# DINO (with PCA + L2)
python scripts/calculate_mmd_v2.py --config src/configs/mmd_configs/mmd_dino_v2.yaml
```

## Output Files
- `analysis/mmd_v2_flow_joint.csv` - Flow MMD results (joint space)
- `analysis/mmd_v2_resnet.csv` - ResNet MMD results
- `analysis/mmd_v2_dino.csv` - DINO MMD results
- `analysis/mmd_v2_logs/` - Execution logs

## Pipeline Steps

### Step 0: Load/Extract Vectors
- Load cached vectors from coverage pipeline
- Extract new vectors if not cached
- Separate into train/eval sets

### Step 1: Preprocessing
- **Flow**: Normalize to [-1, 1] range
- **Features**: Apply PCA + L2 normalization

### Step 2: Alpha Calibration (Flow Only)
- Load or compute global alpha from training sets
- Uses same calibration as coverage pipeline

### Step 3: Transform to Space
- **Flow**: Transform to joint space `[x, y, α*dx, α*dy]`
- **Features**: Already in correct space after PCA

### Step 4: Initialize MMD
- Create RFF map with appropriate dimensionality
- Initialize StreamingMMD

### Step 5: Stream Vectors
- Stream all train and eval vectors to MMD

### Step 6: Compute MMD
- Compute train→eval MMD for all pairs
- Save results to CSV

## Comparison with Old MMD Scripts

### Old Approach (`calculate_flow_mmd.py`, `calculate_feature_mmd.py`)
- ❌ No alpha normalization for flow
- ❌ No PCA for features
- ❌ Computed all-pairs MMD (including train-to-train)
- ❌ Different preprocessing than coverage

### New Approach (`calculate_mmd_v2.py`)
- ✅ Alpha normalization for flow (same as coverage)
- ✅ PCA + L2 for features (same as coverage)
- ✅ Only train→eval MMD (efficient)
- ✅ Identical preprocessing to coverage pipeline
- ✅ Shared caching with coverage

## Validation

All files validated:
- ✅ Python syntax check passed
- ✅ YAML configs validated
- ✅ All imports successful
- ✅ Bash script syntax valid
- ✅ Scripts made executable

## Next Steps

1. Run the pipeline: `bash tmp_run_mmd.sh`
2. Compare MMD results with coverage metrics
3. Analyze correlation between MMD and coverage
4. Use for dataset selection and evaluation
