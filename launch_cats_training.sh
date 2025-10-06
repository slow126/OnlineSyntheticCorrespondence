#!/bin/bash

# Download datasets and weights first
echo "📦 Downloading datasets and weights..."
python download_cats.py --download_all --download_weights

# Run training
echo "🚀 Starting training..."
python train_cats.py --train_dataset synthetic --epochs 50 --batch-size 8 --benchmark pfpascal --freeze False --split_to_use_for_validation test