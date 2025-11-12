#!/usr/bin/env python3
"""
Generate SLURM job scripts for training experiments.

This script creates individual SLURM job files for each experiment configuration,
allowing for independent scheduling and better resource management.

Usage:
    python slurm/generate_jobs.py \
        --machine_config slurm/machine_configs/local.yaml \
        --experiment_config slurm/experiment_configs/default_experiments.yaml
"""

import os
import argparse
import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from itertools import product


# Datasets that need large memory
LARGE_MEMORY_DATASETS = ['flyingthings', 'pointodyssey']


def load_machine_config(config_path: str) -> Dict[str, Any]:
    """Load machine-specific configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def expand_template(template: str, params: Dict[str, Any]) -> str:
    """Expand a name template with parameter values."""
    result = template
    
    # First, handle _join() patterns inside braces: {_join(key, 'separator')}
    for key, value in params.items():
        if isinstance(value, list):
            # Find {_join(key, 'separator')} pattern
            pattern = rf"\{{_join\({key},\s*'([^']+)'\)\}}"
            match = re.search(pattern, result)
            if match:
                separator = match.group(1)
                joined_value = separator.join(str(v) for v in value)
                result = re.sub(pattern, joined_value, result)
    
    # Then, replace remaining {key} patterns
    for key, value in params.items():
        if isinstance(value, list):
            # Skip if already handled by _join
            if f"_join({key}" not in result:
                result = result.replace(f"{{{key}}}", '_'.join(str(v) for v in value))
        else:
            result = result.replace(f"{{{key}}}", str(value))
    
    return result


def set_cpu_based_defaults(exp_config: Dict[str, Any], machine_config: Dict[str, Any]) -> None:
    """Set val_num_workers and n_threads based on CPU count from machine config."""
    slurm_config = machine_config.get('slurm', {})
    cpus_per_task = slurm_config.get('cpus_per_task', 32)
    
    # Set val_num_workers if not already set (use all CPUs for validation)
    if 'val_num_workers' not in exp_config:
        exp_config['val_num_workers'] = cpus_per_task
    
    # Set n_threads if not already set (use all CPUs for training)
    if 'n_threads' not in exp_config:
        exp_config['n_threads'] = cpus_per_task


def generate_grid_experiments(grid_config: Dict[str, Any], base_params: Dict[str, Any], 
                              dataset_defaults: Dict[str, Any], machine_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate experiments from a parameter grid."""
    experiments = []
    
    # Get parameter combinations
    param_names = list(grid_config['parameters'].keys())
    param_values = [grid_config['parameters'][name] for name in param_names]
    
    # Generate all combinations
    for combination in product(*param_values):
        # Create parameter dict for this combination
        params = dict(zip(param_names, combination))
        
        # Start with base parameters
        exp_config = base_params.copy()
        
        # Apply dataset defaults if train_dataset is in params
        train_dataset = params.get('train_dataset', exp_config.get('train_dataset'))
        if dataset_defaults and train_dataset in dataset_defaults:
            defaults = dataset_defaults[train_dataset]
            if defaults and isinstance(defaults, dict):
                exp_config.update(defaults)
        
        # Override with grid parameters
        exp_config.update(params)
        
        # Apply dataset defaults for any dataset in eval_benchmarks (for validation datasets)
        eval_benchmarks = exp_config.get('eval_benchmarks', [])
        if dataset_defaults and isinstance(eval_benchmarks, list):
            for benchmark in eval_benchmarks:
                benchmark_str = str(benchmark).lower()
                # Check if this benchmark has defaults (e.g., 'pointodyssey' in eval_benchmarks)
                for dataset_key, defaults in dataset_defaults.items():
                    if dataset_key.lower() in benchmark_str:
                        if defaults and isinstance(defaults, dict):
                            exp_config.update(defaults)
        
        # Set CPU-based defaults (val_num_workers, n_threads)
        set_cpu_based_defaults(exp_config, machine_config)
        
        # Generate experiment name
        name_template = grid_config.get('name_template', '{train_dataset}_exp')
        exp_name = expand_template(name_template, exp_config)
        exp_config['name_exp'] = exp_name
        
        # Add dataset paths from machine config
        datasets = machine_config.get('datasets', {})
        if train_dataset == 'flyingthings' and 'flyingthings_root' not in exp_config:
            exp_config['flyingthings_root'] = datasets.get('flyingthings_root', '')
        elif train_dataset == 'pointodyssey' and 'pointodyssey_root' not in exp_config:
            exp_config['pointodyssey_root'] = datasets.get('pointodyssey_root', '')
            if 'num_pts_to_track_pointodyssey' not in exp_config:
                exp_config['num_pts_to_track_pointodyssey'] = 32
        elif train_dataset in ['kitti2012', 'kitti2015'] and 'kitti_root' not in exp_config:
            exp_config['kitti_root'] = datasets.get('kitti_root', '')
        
        # Add dataset paths for validation benchmarks
        if isinstance(eval_benchmarks, list):
            eval_benchmarks_str = [str(b).lower() for b in eval_benchmarks]
            # Add pointodyssey_root if pointodyssey is in eval_benchmarks
            if any('pointodyssey' in b for b in eval_benchmarks_str) and 'pointodyssey_root' not in exp_config:
                exp_config['pointodyssey_root'] = datasets.get('pointodyssey_root', '')
            # Add flyingthings_root if flyingthings is in eval_benchmarks
            if any('flyingthings' in b for b in eval_benchmarks_str) and 'flyingthings_root' not in exp_config:
                exp_config['flyingthings_root'] = datasets.get('flyingthings_root', '')
            # Add kitti_root if kitti is in eval_benchmarks
            if any('kitti' in b for b in eval_benchmarks_str) and 'kitti_root' not in exp_config:
                exp_config['kitti_root'] = datasets.get('kitti_root', '')
        
        experiments.append(exp_config)
    
    return experiments


def load_experiments(experiment_config: Dict[str, Any], machine_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load and expand all experiments from config."""
    all_experiments = []
    
    base_params = experiment_config.get('base', {})
    grids = experiment_config.get('grids', [])
    individual_experiments = experiment_config.get('experiments', [])
    dataset_defaults = experiment_config.get('dataset_defaults', {})
    
    # Process grids
    for grid_config in grids:
        grid_experiments = generate_grid_experiments(
            grid_config, base_params, dataset_defaults, machine_config
        )
        all_experiments.extend(grid_experiments)
        print(f"Generated {len(grid_experiments)} experiments from grid: {grid_config.get('name', 'unnamed')}")
    
    # Process individual experiments
    for exp_config in individual_experiments:
        # Start with base
        final_config = base_params.copy()
        
        # Apply dataset defaults
        train_dataset = exp_config.get('train_dataset', final_config.get('train_dataset'))
        if dataset_defaults and train_dataset in dataset_defaults:
            defaults = dataset_defaults[train_dataset]
            if defaults and isinstance(defaults, dict):
                final_config.update(defaults)
        
        # Override with individual experiment params
        final_config.update(exp_config)
        
        # Apply dataset defaults for any dataset in eval_benchmarks (for validation datasets)
        eval_benchmarks = final_config.get('eval_benchmarks', [])
        if dataset_defaults and isinstance(eval_benchmarks, list):
            for benchmark in eval_benchmarks:
                benchmark_str = str(benchmark).lower()
                # Check if this benchmark has defaults (e.g., 'pointodyssey' in eval_benchmarks)
                for dataset_key, defaults in dataset_defaults.items():
                    if dataset_key.lower() in benchmark_str:
                        if defaults and isinstance(defaults, dict):
                            final_config.update(defaults)
        
        # Set CPU-based defaults (val_num_workers, n_threads)
        set_cpu_based_defaults(final_config, machine_config)
        
        # Ensure name_exp is set
        if 'name_exp' not in final_config:
            final_config['name_exp'] = final_config.get('name', 'unnamed_exp')
        
        # Add dataset paths from machine config
        datasets = machine_config.get('datasets', {})
        if train_dataset == 'flyingthings' and 'flyingthings_root' not in final_config:
            final_config['flyingthings_root'] = datasets.get('flyingthings_root', '')
        elif train_dataset == 'pointodyssey' and 'pointodyssey_root' not in final_config:
            final_config['pointodyssey_root'] = datasets.get('pointodyssey_root', '')
            if 'num_pts_to_track_pointodyssey' not in final_config:
                final_config['num_pts_to_track_pointodyssey'] = 32
        elif train_dataset in ['kitti2012', 'kitti2015'] and 'kitti_root' not in final_config:
            final_config['kitti_root'] = datasets.get('kitti_root', '')
        
        # Add dataset paths for validation benchmarks
        if isinstance(eval_benchmarks, list):
            eval_benchmarks_str = [str(b).lower() for b in eval_benchmarks]
            # Add pointodyssey_root if pointodyssey is in eval_benchmarks
            if any('pointodyssey' in b for b in eval_benchmarks_str) and 'pointodyssey_root' not in final_config:
                final_config['pointodyssey_root'] = datasets.get('pointodyssey_root', '')
            # Add flyingthings_root if flyingthings is in eval_benchmarks
            if any('flyingthings' in b for b in eval_benchmarks_str) and 'flyingthings_root' not in final_config:
                final_config['flyingthings_root'] = datasets.get('flyingthings_root', '')
            # Add kitti_root if kitti is in eval_benchmarks
            if any('kitti' in b for b in eval_benchmarks_str) and 'kitti_root' not in final_config:
                final_config['kitti_root'] = datasets.get('kitti_root', '')
        
        all_experiments.append(final_config)
    
    return all_experiments


def get_memory_for_dataset(train_dataset: str, config: Dict[str, Any]) -> str:
    """Get memory requirement based on training dataset."""
    slurm_config = config.get('slurm', {})
    mem_large = slurm_config.get('mem_large', '128g')
    mem_default = slurm_config.get('mem_default', '64g')
    return mem_large if train_dataset in LARGE_MEMORY_DATASETS else mem_default


def build_train_command(exp_config: Dict[str, Any], machine_config: Dict[str, Any]) -> str:
    """Build the training command from experiment configuration."""
    machine = machine_config.get('machine', {})
    python_path = machine.get('python_path', 'python3')
    project_root = machine.get('project_root', os.getcwd())
    
    cmd_parts = [python_path, f'{project_root}/train_cats.py']
    
    # Add dataset paths from machine config if not explicitly set in exp_config
    datasets = machine_config.get('datasets', {})
    
    # Add all arguments from config
    # Config keys should match train_cats.py argument names exactly
    for key, value in exp_config.items():
        # Skip internal keys and arguments that should use defaults
        if key in ['name', 'step']:
            continue
            
        if value is None:
            continue
        
        # Use config key directly as argument name (configs should match train_cats.py)
        if isinstance(value, bool):
            # For boolean arguments, pass the value explicitly (True/False as string)
            # This works with boolean_string type that expects 'True' or 'False'
            cmd_parts.append(f'--{key}')
            cmd_parts.append(str(value))
        elif isinstance(value, list):
            # Handle list arguments like eval_benchmarks
            cmd_parts.append(f'--{key}')
            cmd_parts.extend([str(v) for v in value])
        else:
            cmd_parts.append(f'--{key}')
            cmd_parts.append(str(value))
    
    # Add dataset paths if not already in exp_config
    # Note: These use underscores in train_cats.py, not hyphens
    train_dataset = exp_config.get('train_dataset', '')
    if train_dataset == 'flyingthings' and 'flyingthings_root' not in exp_config:
        if datasets.get('flyingthings_root'):
            cmd_parts.extend(['--flyingthings_root', datasets.get('flyingthings_root', '')])
    elif train_dataset == 'pointodyssey' and 'pointodyssey_root' not in exp_config:
        if datasets.get('pointodyssey_root'):
            cmd_parts.extend(['--pointodyssey_root', datasets.get('pointodyssey_root', '')])
    elif train_dataset in ['kitti2012', 'kitti2015'] and 'kitti_root' not in exp_config:
        if datasets.get('kitti_root'):
            cmd_parts.extend(['--kitti_root', datasets.get('kitti_root', '')])
    
    # Add dataset paths for validation benchmarks
    eval_benchmarks = exp_config.get('eval_benchmarks', [])
    if isinstance(eval_benchmarks, list):
        eval_benchmarks_str = [str(b).lower() for b in eval_benchmarks]
        # Add pointodyssey_root if pointodyssey is in eval_benchmarks (for validation)
        if any('pointodyssey' in b for b in eval_benchmarks_str) and 'pointodyssey_root' not in exp_config:
            if datasets.get('pointodyssey_root'):
                cmd_parts.extend(['--pointodyssey_root', datasets.get('pointodyssey_root', '')])
        # Add flyingthings_root if flyingthings is in eval_benchmarks (for validation)
        if any('flyingthings' in b for b in eval_benchmarks_str) and 'flyingthings_root' not in exp_config:
            if datasets.get('flyingthings_root'):
                cmd_parts.extend(['--flyingthings_root', datasets.get('flyingthings_root', '')])
        # Add kitti_root if kitti is in eval_benchmarks (for validation)
        if any('kitti' in b for b in eval_benchmarks_str) and 'kitti_root' not in exp_config:
            if datasets.get('kitti_root'):
                cmd_parts.extend(['--kitti_root', datasets.get('kitti_root', '')])
        # Add tss_root if tss is in eval_benchmarks (for validation)
        if any('tss' in b for b in eval_benchmarks_str) and 'tss_root' not in exp_config:
            if datasets.get('tss_root'):
                cmd_parts.extend(['--tss_root', datasets.get('tss_root', '')])
    
    if 'datapath' not in exp_config:
        if datasets.get('datapath'):
            cmd_parts.extend(['--datapath', datasets.get('datapath', './models/Datasets_CATs')])
    
    return ' '.join(cmd_parts)


def generate_slurm_script(
    exp_config: Dict[str, Any], 
    exp_name: str, 
    job_dir: Path, 
    machine_config: Dict[str, Any]
) -> Path:
    """Generate a SLURM job script for a single experiment."""
    train_dataset = exp_config.get('train_dataset', 'synthetic')
    memory = get_memory_for_dataset(train_dataset, machine_config)
    
    slurm_config = machine_config.get('slurm', {})
    machine = machine_config.get('machine', {})
    project_root = machine.get('project_root', os.getcwd())
    conda_env = machine.get('conda_env')
    
    # Create job filename
    job_filename = f"job_{exp_name}.sh"
    job_path = job_dir / job_filename
    
    # Convert job_dir to absolute path for SLURM output paths
    job_dir_abs = job_dir.resolve()
    
    # Build training command
    train_cmd = build_train_command(exp_config, machine_config)
    
    # Generate SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --output={job_dir_abs}/logs/{exp_name}_%j.out
#SBATCH --error={job_dir_abs}/logs/{exp_name}_%j.err
#SBATCH --time={slurm_config.get('time', '23:59:00')}
#SBATCH --nodes={slurm_config.get('nodes', 1)}
#SBATCH --ntasks={slurm_config.get('ntasks', 1)}
#SBATCH --cpus-per-task={slurm_config.get('cpus_per_task', 32)}
#SBATCH --gpus={slurm_config.get('gpus', 1)}
#SBATCH --mem={memory}
#SBATCH --qos={slurm_config.get('qos', 'cs')}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: {exp_name}"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Memory: {memory}"
echo "Training Dataset: {train_dataset}"

# Change to project directory
cd {project_root}

# Activate conda environment if specified
"""
    
    if conda_env:
        script_content += f"""# Activate conda environment
# Source .bashrc to initialize conda
source "$HOME/.bashrc"

conda activate {conda_env}
echo "Activated conda environment: {conda_env}"
"""
    
    # Extract just the arguments (remove python/python3 from command)
    # train_cmd is like "python3 /path/to/train_cats.py --arg1 val1 ..."
    cmd_parts = train_cmd.split()
    if len(cmd_parts) > 0 and cmd_parts[0] in ['python', 'python3']:
        # Remove python/python3, keep the rest
        cmd_args = ' '.join(cmd_parts[1:])
    else:
        # Already just arguments, or different format
        cmd_args = train_cmd
    
    script_content += f"""
# Run training with srun (allows sattach and real-time output)
# --ntasks=1 ensures only one instance runs (not multiple per CPU)
# -u flag disables Python buffering for real-time log output
echo "Starting training..."
srun --ntasks=1 python3 -u {cmd_args}

echo "Training completed at: $(date)"
"""
    
    # Write script
    with open(job_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(job_path, 0o755)
    
    return job_path


def main():
    parser = argparse.ArgumentParser(description='Generate SLURM job scripts for training experiments')
    parser.add_argument('-M', '--machine_config', type=str, default='slurm/machine_configs/local.yaml',
                        help='Path to machine-specific config file')
    parser.add_argument('-E', '--experiment_config', type=str, default='slurm/experiment_configs/default_experiments.yaml',
                        help='Path to experiment configuration file')
    parser.add_argument('-O', '--output_dir', type=str, default='./slurm_jobs',
                        help='Directory to save generated job scripts')
    parser.add_argument('-S', '--submit', action='store_true',
                        help='Automatically submit all generated jobs')
    parser.add_argument('-D', '--dry_run', action='store_true',
                        help='Print job commands without creating files')
    
    args = parser.parse_args()
    
    # Load machine configuration
    if not os.path.exists(args.machine_config):
        print(f"Error: Machine config file not found: {args.machine_config}")
        print(f"Please create a machine config file. See template in slurm/machine_configs/")
        return
    
    machine_config = load_machine_config(args.machine_config)
    machine_name = machine_config.get('machine', {}).get('name', 'unknown')
    project_root = machine_config.get('machine', {}).get('project_root', os.getcwd())
    
    # Load experiment configuration
    if not os.path.exists(args.experiment_config):
        print(f"Error: Experiment config file not found: {args.experiment_config}")
        print(f"Please create an experiment config file. See template in slurm/experiment_configs/")
        return
    
    experiment_config = load_experiment_config(args.experiment_config)
    
    print(f"Loaded machine config: {machine_name}")
    print(f"Project root: {project_root}")
    print(f"Machine config: {args.machine_config}")
    print(f"Experiment config: {args.experiment_config}\n")
    
    # Create output directory
    job_dir = Path(args.output_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'logs').mkdir(exist_ok=True)
    
    # Load and expand experiments
    experiments = load_experiments(experiment_config, machine_config)
    
    print(f"Generated {len(experiments)} SLURM job scripts...")
    print(f"Output directory: {job_dir}\n")
    
    job_files = []
    for i, exp_config in enumerate(experiments, 1):
        exp_name = exp_config.get('name_exp', f'exp_{i}')
        train_dataset = exp_config.get('train_dataset', 'unknown')
        memory = get_memory_for_dataset(train_dataset, machine_config)
        
        print(f"[{i}/{len(experiments)}] {exp_name}")
        print(f"  Dataset: {train_dataset}, Memory: {memory}")
        
        if args.dry_run:
            print(f"  Would create: job_{exp_name}.sh")
            continue
        
        job_path = generate_slurm_script(exp_config, exp_name, job_dir, machine_config)
        job_files.append(job_path)
        print(f"  Created: {job_path}")
    
    if args.dry_run:
        print("\nDry run complete. Use without --dry_run to generate files.")
        return
    
    # Create submission script
    submit_script = job_dir / 'submit_all.sh'
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all generated SLURM jobs\n")
        f.write("# Get the directory where this script is located (works from any location)\n")
        f.write("SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n")
        f.write("cd \"$SCRIPT_DIR\"\n\n")
        for job_file in job_files:
            f.write(f"echo 'Submitting {job_file.name}...'\n")
            f.write(f"sbatch \"$SCRIPT_DIR/{job_file.name}\"\n")
            f.write("sleep 1  # Small delay between submissions\n\n")
        f.write("echo 'All jobs submitted!'\n")
    
    os.chmod(submit_script, 0o755)
    print(f"\nCreated submission script: {submit_script}")
    
    # Create a script to submit jobs individually
    submit_individual = job_dir / 'submit_individual.sh'
    with open(submit_individual, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit individual jobs\n")
        f.write("# Usage: ./submit_individual.sh job_<exp_name>.sh\n\n")
        f.write("if [ $# -eq 0 ]; then\n")
        f.write("    echo 'Usage: ./submit_individual.sh job_<exp_name>.sh'\n")
        f.write("    echo 'Available jobs:'\n")
        for job_file in job_files:
            f.write(f"    echo '  {job_file.name}'\n")
        f.write("    exit 1\n")
        f.write("fi\n\n")
        f.write("sbatch \"$1\"\n")
    
    os.chmod(submit_individual, 0o755)
    print(f"Created individual submission script: {submit_individual}")
    
    # Auto-submit if requested
    if args.submit:
        print("\nSubmitting all jobs...")
        import subprocess
        subprocess.run(['bash', str(submit_script)])
    else:
        print(f"\nTo submit all jobs, run: bash {submit_script}")
        print(f"Or submit individually: sbatch {job_dir}/job_<exp_name>.sh")
        print(f"\nTo monitor jobs: squeue -u $USER")


if __name__ == '__main__':
    main()

