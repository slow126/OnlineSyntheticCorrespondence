# PyTorch Lightning Training System

This directory contains a PyTorch Lightning-based training system that preserves all functionality from `train_cats_unified.py` while providing a cleaner, more modular architecture.

## Key Features

- **MMD Calculations**: Preserved exactly as in original implementation, computed alongside PCK metrics
- **From-Scratch Training**: Support for `pretrained_backbone=False` to test from-scratch training strength
- **Finetuning**: Full support for loading checkpoints and creating new snapshot directories
- **Multi-Benchmark Evaluation**: Evaluates on multiple benchmarks with motion-aware metrics
- **Debug Visualizations**: Flow visualizations for debugging (pre-training and per-epoch)
- **CSV Logging**: Validation results logged to CSV in same format as original
- **Config-Based Sweeps**: Generate experiment configs from sweep definitions

## Quick Start

### Basic Training

```bash
python train_lightning.py --config src/configs/CorrespondenceConfigs/pointodyssey.yaml
```

### Inspect Data

```bash
python train_lightning.py --config src/configs/CorrespondenceConfigs/pointodyssey.yaml --inspect-data --inspect-visualize
```

## Architecture

### Components

1. **Lightning Module** (`src/training/correspondence_lightning.py`)
   - Wraps model with `forward(trg_img, src_img) -> flow` interface
   - Handles training step with EPE loss
   - Configures optimizer with separate learning rates for model vs backbone

2. **Data Module** (`src/training/correspondence_datamodule.py`)
   - Manages train/val datasets using existing `create_training_dataset` and `create_validation_datasets`
   - Handles multiple validation dataloaders (one per benchmark)

3. **Callbacks** (`src/training/callbacks/`)
   - `MMDValidationCallback`: Performs multi-benchmark validation with MMD calculations
   - `CSVLoggingCallback`: Logs validation results to CSV
   - `VisualizationCallback`: Debug flow visualizations
   - `CheckpointCallback`: Saves best models per benchmark and overall best
   - `SummaryCallback`: Writes training summary text file

### Config Format

Uses the same config format as `train_cats_unified.py`:

```yaml
model:
  backbone: 'resnet101'
  freeze: true
  pretrained_backbone: false  # Set to false for from-scratch training

training:
  epochs: 50
  batch_size: 2
  lr: 3e-4
  lr_backbone: 3e-6
  # ... other training parameters

dataset:
  dataset_name: 'pointodyssey'
  # ... dataset parameters

evaluation:
  eval_benchmarks: ['kitti2012', 'kitti2015', 'pointodyssey']
  eval_alphas: [0.01, 0.01, 0.01]
  # ... evaluation parameters

paths:
  snapshots: './snapshots'
  pretrained: null  # Path to checkpoint for finetuning, or null
  start_epoch: 0
```

## Sweep Generation

### Creating Sweep Configs

Create a sweep config file (e.g., `slurm/experiment_configs/my_sweep.yaml`):

```yaml
base_config: "src/configs/CorrespondenceConfigs/pointodyssey.yaml"
output_dir: "experiment_configs/generated"

sweeps:
  - name: "lr_sweep"
    parameters:
      training.lr: [1e-4, 3e-4, 1e-3]
      training.lr_backbone: [1e-6, 3e-6]
      model.pretrained_backbone: [True, False]  # Test from-scratch vs pretrained
    name_template: "lr{training.lr}_lrbackbone{training.lr_backbone}_pretrained{model.pretrained_backbone}"
    output_dir: "experiment_configs/generated/lr_sweep"
```

### Generating Configs

```bash
python slurm/config_generator.py --sweep_config slurm/experiment_configs/my_sweep.yaml
```

### Generating SLURM Jobs

```bash
python slurm/generate_experiments.py \
    --machine_config slurm/machine_configs/local.yaml \
    --sweep_config slurm/experiment_configs/my_sweep.yaml \
    --output_dir ./slurm_jobs
```

This will:
1. Generate training configs from the sweep definition
2. Create SLURM job scripts that call `train_lightning.py --config <generated_config>`
3. Create submission scripts for easy job management

## From-Scratch Training

To train from scratch (without ImageNet pretrained backbone):

```yaml
model:
  pretrained_backbone: false  # Set to false
```

This is important for testing the strength of from-scratch training.

## Finetuning

To finetune from a checkpoint:

```yaml
paths:
  pretrained: "snapshots/previous_experiment/model_best.pth"  # Path to checkpoint
  start_epoch: 0  # Will be loaded from checkpoint if -1
```

The system will:
- Load model, optimizer, and scheduler states
- Create a new snapshot directory: `{pretrained_name}_finetune_{exp_name}`
- Preserve best performance tracking from the checkpoint

## MMD Calculations

MMD^2 values are computed alongside PCK metrics and logged to:
- TensorBoard: `val/{benchmark}/MMD2_*` scalars
- CSV: `mmd2_pred_corr_vs_pred_miss`, `mmd2_pred_corr_vs_gt`, `mmd2_pred_miss_vs_gt` columns

MMD is computed when `training.mmd_every_n_epochs > 0` and `epoch % mmd_every_n_epochs == 0`.

## Differences from train_cats_unified.py

- Uses PyTorch Lightning for training loop management
- Modular callback-based architecture
- Same config format (backward compatible)
- Same output format (CSV, checkpoints, summaries)
- All functionality preserved (MMD, visualizations, etc.)

## Migration Notes

- `train_cats_unified.py` remains untouched as reference/backup
- New system uses same config format - existing configs work as-is
- Output format matches original (same CSV columns, checkpoint structure, etc.)
- Can run both systems side-by-side for comparison
