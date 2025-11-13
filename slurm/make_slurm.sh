#!/bin/bash

# Base experiments
python generate_jobs.py --machine_config machine_configs/remote.yaml --experiment_config experiment_configs/pointodyssey_experiment.yaml --output_dir PtOd_jobs -S
# python generate_jobs.py --machine_config machine_configs/remote.yaml --experiment_config experiment_configs/synthetic.yaml --output_dir synthetic_jobs -S
python generate_jobs.py --machine_config machine_configs/remote.yaml --experiment_config experiment_configs/spair.yaml --output_dir spair_jobs -S
python generate_jobs.py --machine_config machine_configs/remote.yaml --experiment_config experiment_configs/flyingthings.yaml --output_dir flyingthings_jobs -S

# Synthetic Views
python generate_jobs.py --machine_config machine_configs/remote.yaml --experiment_config experiment_configs/synthetic_views.yaml --output_dir synthetic_views_jobs -S
