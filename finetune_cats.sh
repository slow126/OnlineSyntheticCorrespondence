# Run training
echo "🚀 Starting training..."
python train_cats.py --train_dataset spair --epochs 50 --batch-size 8 --benchmark spair --freeze True --split_to_use_for_validation test --pretrained "snapshots/2025_10_04_01_34"