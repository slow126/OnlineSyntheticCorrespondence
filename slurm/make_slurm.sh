python generate_jobs.py --machine_config slurm/machine_configs/remote.yaml --experiment_config slurm/experiment_configs/pointodyssey_experiment.yaml --output_dir PtOd_jobs
python generate_jobs.py --machine_config slurm/machine_configs/remote.yaml --experiment_config slurm/experiment_configs/synthetic.yaml --output_dir synthetic_jobs
python generate_jobs.py --machine_config slurm/machine_configs/remote.yaml --experiment_config slurm/experiment_configs/spair.yaml --output_dir spair_jobs
python generate_jobs.py --machine_config slurm/machine_configs/remote.yaml --experiment_config slurm/experiment_configs/flyingthings.yaml --output_dir flyingthings_jobs
