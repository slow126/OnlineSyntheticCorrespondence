#!/usr/bin/env python3
"""
Generate SLURM job scripts for precomputing PointOdyssey cache.

This script creates individual SLURM job files for each parameter combination
of the precompute_pointodyssey_cache.py script.

Usage:
    python slurm/precomupte_slurm.py \
        --machine_config slurm/machine_configs/remote.yaml \
        --output_dir ./slurm_jobs_precompute
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any


def load_machine_config(config_path: str) -> Dict[str, Any]:
    """Load machine-specific configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_slurm_script(
    job_name: str,
    command_args: str,
    job_dir: Path,
    machine_config: Dict[str, Any]
) -> Path:
    """Generate a SLURM job script for a single precompute job."""
    slurm_config = machine_config.get('slurm', {})
    machine = machine_config.get('machine', {})
    project_root = machine.get('project_root', os.getcwd())
    conda_env = machine.get('conda_env')
    
    # Create job filename
    job_filename = f"job_{job_name}.sh"
    job_path = job_dir / job_filename
    
    # Convert job_dir to absolute path for SLURM output paths
    job_dir_abs = job_dir.resolve()
    
    # Use large memory for PointOdyssey
    # Precomputation doesn't need GPUs, so set to 0
    memory = slurm_config.get('mem_large', '128g')
    
    # Generate SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_dir_abs}/logs/{job_name}_%j.out
#SBATCH --error={job_dir_abs}/logs/{job_name}_%j.err
#SBATCH --time={slurm_config.get('time', '23:59:00')}
#SBATCH --nodes={slurm_config.get('nodes', 1)}
#SBATCH --ntasks={slurm_config.get('ntasks', 1)}
#SBATCH --cpus-per-task={slurm_config.get('cpus_per_task', 32)}
#SBATCH --gpus=0
#SBATCH --mem={memory}
#SBATCH --qos={slurm_config.get('qos', 'cs')}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: {job_name}"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Memory: {memory}"

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
# Run precompute script
# -u flag disables Python buffering for real-time log output
echo "Starting precompute..."
python3 -u {command_args}

echo "Precompute completed at: $(date)"
"""
    
    # Write script
    with open(job_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(job_path, 0o755)
    
    return job_path


def main():
    parser = argparse.ArgumentParser(description='Generate SLURM job scripts for PointOdyssey cache precomputation')
    parser.add_argument('-M', '--machine_config', type=str, default='slurm/machine_configs/remote.yaml',
                        help='Path to machine-specific config file')
    parser.add_argument('-O', '--output_dir', type=str, default='./slurm_jobs_precompute',
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
    datasets = machine_config.get('datasets', {})
    pointodyssey_root = datasets.get('pointodyssey_root', '/home/slow1/Data/PointOdyssey')
    
    print(f"Loaded machine config: {machine_name}")
    print(f"Project root: {project_root}")
    print(f"PointOdyssey root: {pointodyssey_root}")
    print(f"Machine config: {args.machine_config}\n")
    
    # Define all parameter combinations
    # S values: 4, 8, 16
    # strides values: 1, 2, 4
    # All combinations = 9 jobs
    jobs = []
    for S in [4, 8, 16]:
        for strides in [1, 2, 4]:
            job_name = f"precompute_S{S}_strides{strides}"
            
            # Build command arguments
            cmd_args = (
                f"scripts/precompute_pointodyssey_cache.py "
                f"--pointodyssey_root {pointodyssey_root} "
                f"--dset train "
                f"--S {S} "
                f"--N 32 "
                f"--strides {strides} "
                f"--size 512 "
                f"--feature_size 32 "
                f"--max_pts 32 "
                f"--num_workers {machine_config.get('slurm', {}).get('cpus_per_task', 32)} "
                f"--batch_size 64 "
                f"--all_points"
            )
            
            jobs.append((job_name, cmd_args))
    
    print(f"Generated {len(jobs)} SLURM job configurations...")
    print(f"Output directory: {args.output_dir}\n")
    
    # Create output directory
    job_dir = Path(args.output_dir)
    if not args.dry_run:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / 'logs').mkdir(exist_ok=True)
    
    job_files = []
    for i, (job_name, cmd_args) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {job_name}")
        print(f"  Command: python3 {cmd_args}")
        
        if args.dry_run:
            print(f"  Would create: job_{job_name}.sh")
            continue
        
        job_path = generate_slurm_script(job_name, cmd_args, job_dir, machine_config)
        job_files.append(job_path)
        print(f"  Created: {job_path}")
    
    if args.dry_run:
        print("\nDry run complete. Use without --dry_run to generate files.")
        return
    
    # Create submission script
    submit_script = job_dir / 'submit_all.sh'
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all generated SLURM jobs for PointOdyssey cache precomputation\n")
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
    
    # Auto-submit if requested
    if args.submit:
        print("\nSubmitting all jobs...")
        import subprocess
        subprocess.run(['bash', str(submit_script)])
    else:
        print(f"\nTo submit all jobs, run: bash {submit_script}")
        print(f"Or submit individually: sbatch {job_dir}/job_<job_name>.sh")
        print(f"\nTo monitor jobs: squeue -u $USER")


if __name__ == '__main__':
    main()

