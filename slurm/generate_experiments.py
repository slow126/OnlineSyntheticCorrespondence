#!/usr/bin/env python3
"""
Generate SLURM job scripts for training experiments using config-based approach.

This script:
1. Reads sweep configs and generates full training YAML configs using config_generator
2. Creates SLURM job scripts that call train_lightning.py with generated configs
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any

# Import config generator - use absolute import for script execution
try:
    from slurm.config_generator import generate_all_configs
except ImportError:
    # Fallback for when running as module
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from slurm.config_generator import generate_all_configs


def load_machine_config(config_path: str) -> Dict[str, Any]:
    """Load machine-specific configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_memory_for_dataset(train_dataset: str, config: Dict[str, Any]) -> str:
    """Get memory requirement based on training dataset."""
    slurm_config = config.get('slurm', {})
    mem_large = slurm_config.get('mem_large', '128g')
    mem_default = slurm_config.get('mem_default', '64g')
    
    # Datasets that need large memory
    large_memory_datasets = ['flyingthings', 'pointodyssey']
    
    # Handle mixed datasets (format: "dataset1+dataset2" or "mixed")
    if '+' in train_dataset:
        # Check if any sub-dataset needs large memory
        sub_datasets = train_dataset.split('+')
        needs_large = any(ds in large_memory_datasets for ds in sub_datasets)
        return mem_large if needs_large else mem_default
    elif train_dataset == 'mixed':
        # Unknown mixed dataset - use large memory to be safe
        return mem_large
    
    return mem_large if train_dataset in large_memory_datasets else mem_default


def generate_slurm_script(
    config_path: str,
    exp_name: str,
    job_dir: Path,
    machine_config: Dict[str, Any]
) -> Path:
    """
    Generate a SLURM job script for a single experiment.
    
    Args:
        config_path: Path to generated training config YAML file
        exp_name: Experiment name
        job_dir: Directory to save job script
        machine_config: Machine configuration dict
        
    Returns:
        Path to generated job script
    """
    # Determine memory requirement from config
    config = yaml.safe_load(open(config_path, 'r'))
    dataset_config = config['dataset']
    
    # Handle mixed datasets
    is_mixed = dataset_config.get('mixed', False) or 'datasets' in dataset_config
    if is_mixed:
        datasets_list = dataset_config.get('datasets', [])
        train_dataset = '+'.join(datasets_list) if datasets_list else 'mixed'
    else:
        train_dataset = dataset_config.get('dataset_name', 'unknown')
    
    memory = get_memory_for_dataset(train_dataset, machine_config)
    
    slurm_config = machine_config.get('slurm', {})
    machine = machine_config.get('machine', {})
    datasets_config = machine_config.get('datasets', {})
    project_root = machine.get('project_root', os.getcwd())
    conda_env = machine.get('conda_env')
    hf_cache_dir = datasets_config.get('hf_cache_dir')
    
    # Create job filename
    job_filename = f"job_{exp_name}.sh"
    job_path = job_dir / job_filename
    
    # Convert job_dir to absolute path for SLURM output paths
    job_dir_abs = job_dir.resolve()
    
    # Convert config_path to absolute path
    config_path_abs = os.path.abspath(config_path)
    
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
echo "Config: {config_path_abs}"

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
    
    script_content += f"""
# Force HF offline mode for isolated clusters
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
"""
    if hf_cache_dir:
        script_content += f"export HF_DATASETS_CACHE={hf_cache_dir}\n"

    script_content += f"""
# Run training with srun (allows sattach and real-time output)
# --ntasks=1 ensures only one instance runs (not multiple per CPU)
# -u flag disables Python buffering for real-time log output
echo "Starting training..."
echo "Command: python3 -u train_lightning.py --config {config_path_abs}"
srun --ntasks=1 python3 -u train_lightning.py --config {config_path_abs}

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
    parser.add_argument('-S', '--sweep_config', type=str, required=True,
                       help='Path to sweep configuration file')
    parser.add_argument('-O', '--output_dir', type=str, default='./slurm_jobs',
                       help='Directory to save generated job scripts')
    parser.add_argument('--config_output_dir', type=str, default=None,
                       help='Directory to save generated config files (defaults to sweep config dir)')
    parser.add_argument('--submit', action='store_true',
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
    
    # Load sweep configuration
    if not os.path.exists(args.sweep_config):
        print(f"Error: Sweep config file not found: {args.sweep_config}")
        return
    
    print(f"Loaded machine config: {machine_name}")
    print(f"Project root: {project_root}")
    print(f"Machine config: {args.machine_config}")
    print(f"Sweep config: {args.sweep_config}\n")
    
    # Generate configs from sweep
    print("Generating training configs from sweep...")
    configs = generate_all_configs(args.sweep_config, args.config_output_dir)
    print(f"Generated {len(configs)} config files\n")
    
    # Create output directory
    job_dir = Path(args.output_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'logs').mkdir(exist_ok=True)
    
    print(f"Generating {len(configs)} SLURM job scripts...")
    print(f"Output directory: {job_dir}\n")
    
    job_files = []
    for i, (config_path, exp_name) in enumerate(configs, 1):
        # Load config to get dataset info
        config = yaml.safe_load(open(config_path, 'r'))
        dataset_config = config['dataset']
        
        # Handle mixed datasets
        is_mixed = dataset_config.get('mixed', False) or 'datasets' in dataset_config
        if is_mixed:
            datasets_list = dataset_config.get('datasets', [])
            train_dataset = '+'.join(datasets_list) if datasets_list else 'mixed'
        else:
            train_dataset = dataset_config.get('dataset_name', 'unknown')
        
        memory = get_memory_for_dataset(train_dataset, machine_config)
        
        print(f"[{i}/{len(configs)}] {exp_name}")
        print(f"  Config: {config_path}")
        print(f"  Dataset: {train_dataset}, Memory: {memory}")
        
        if args.dry_run:
            print(f"  Would create: job_{exp_name}.sh")
            continue
        
        job_path = generate_slurm_script(config_path, exp_name, job_dir, machine_config)
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
