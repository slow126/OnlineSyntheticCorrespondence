# SLURM Job Generation System

This directory contains tools for generating and managing SLURM job scripts for training experiments.

## Directory Structure

```
slurm/
├── generate_jobs.py          # Main script to generate SLURM jobs
├── machine_configs/           # Machine-specific configurations
│   ├── local.yaml            # Local machine config
│   └── remote.yaml           # Remote cluster config (update paths!)
└── experiment_configs/        # Experiment definitions
    ├── default_experiments.yaml  # Default experiment config
    └── ...                    # Add more experiment configs as needed
```

## Quick Start

### 1. Configure Your Machine

Edit `machine_configs/local.yaml` (or create a new one for your remote machine):

```yaml
machine:
  name: "local"
  project_root: "/path/to/OnlineSyntheticCorrespondence"
  conda_env: "cuda"  # or null if not using conda
  python_path: "python3"

datasets:
  flyingthings_root: "/path/to/FlyingThings3D_tiny/"
  pointodyssey_root: "/path/to/PointOdyssey"
  tss_root: "/path/to/TSS_CVPR2016"
  datapath: "./models/Datasets_CATs"
```

### 2. Define Your Experiments

Edit `experiment_configs/default_experiments.yaml` to define your experiments:

- **Base parameters**: Shared defaults for all experiments
- **Grids**: Parameter combinations (generates all combinations)
- **Individual experiments**: Specific configurations

### 3. Generate Jobs

```bash
# Generate jobs (dry run to see what will be created)
python slurm/generate_jobs.py --dry_run

# Generate jobs for real
python slurm/generate_jobs.py

# Use different configs
python slurm/generate_jobs.py \
    --machine_config slurm/machine_configs/remote.yaml \
    --experiment_config slurm/experiment_configs/my_experiments.yaml
```

### 4. Submit Jobs

```bash
# Submit all jobs
bash slurm_jobs/submit_all.sh

# Submit individual job
sbatch slurm_jobs/job_<exp_name>.sh

# Monitor jobs
squeue -u $USER
```

## Configuration Files

### Machine Config (`machine_configs/*.yaml`)

Defines machine-specific settings:
- Project root path
- Dataset paths
- Conda environment
- SLURM resource defaults

### Experiment Config (`experiment_configs/*.yaml`)

Defines experiments:
- **base**: Default parameters for all experiments
- **grids**: Parameter grids that generate combinations
- **experiments**: Individual experiment definitions
- **dataset_defaults**: Dataset-specific defaults

## Example: Creating a New Experiment Config

Create `experiment_configs/quick_tests.yaml`:

```yaml
base:
  epochs: 10
  batch_size: 8
  lr: 3e-4
  # ... other defaults

experiments:
  - name: "quick_flyingthings_test"
    train_dataset: "flyingthings"
    eval_benchmarks: ["spair"]
    eval_alphas: [0.1]
    epochs: 5
```

Then generate jobs:
```bash
python slurm/generate_jobs.py \
    --experiment_config slurm/experiment_configs/quick_tests.yaml
```

## Memory Allocation

The system automatically allocates memory based on training dataset:
- **FlyingThings** and **PointOdyssey**: 128GB (from `mem_large`)
- **Other datasets**: 64GB (from `mem_default`)

This can be configured in the machine config file.

## Tips

1. **Start small**: Test with a few experiments first using `--dry_run`
2. **Use separate configs**: Create different experiment configs for different experiment sets
3. **Monitor resources**: Check `squeue` and adjust SLURM parameters if needed
4. **Check logs**: Job outputs go to `slurm_jobs/logs/`

