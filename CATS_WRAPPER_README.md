# CATs++ Synthetic Dataset Wrapper

This directory contains a wrapper that makes the online synthetic correspondence dataset compatible with the CATs++ model for training.

## Overview

The wrapper bridges the gap between:
- **Online Synthetic Dataset**: Generates geometry/normal maps using OpenGL shaders
- **CATs++ Model**: Expects RGB images with corresponding keypoints

## Files

### Core Components

1. **`src/data/synth/datasets/cats_wrapper.py`** - Main wrapper class
   - `CATsSyntheticDataset`: Converts synthetic data to CATs++ format
   - Handles geometry→RGB conversion
   - Generates keypoints from geometry features
   - Provides CATs++ compatible batch format

2. **`train_cats.py`** - Training script
   - Initializes CATs++ model
   - Sets up training with synthetic dataset
   - Includes full training loop with validation

3. **`test_cats_wrapper.py`** - Test script
   - Verifies wrapper functionality
   - Tests batch generation and data formats

4. **`example_usage.py`** - Usage examples
   - Basic usage patterns
   - Training setup examples
   - Custom configuration examples

## Quick Start

### 1. Test the Wrapper
```bash
python test_cats_wrapper.py
```

### 2. Run Examples
```bash
python example_usage.py
```

### 3. Start Training
```bash
python train_cats.py --num_samples 1000 --epochs 50 --batch-size 16
```

## Key Features

### Dataset Compatibility
- ✅ Returns data in exact format expected by CATs++
- ✅ Handles image normalization (ImageNet stats)
- ✅ Generates appropriate keypoints from geometry
- ✅ Supports data augmentation
- ✅ Configurable image sizes and batch processing

### Training Features
- ✅ Configurable hyperparameters
- ✅ Multiple optimizers (AdamW with different learning rates)
- ✅ Learning rate scheduling (step or cosine)
- ✅ TensorBoard logging
- ✅ Model checkpointing
- ✅ GPU support with DataParallel

### Synthetic Data Integration
- ✅ Uses existing shader-based rendering
- ✅ Generates geometry/normal pairs on-the-fly
- ✅ Converts to RGB images for CATs++ compatibility
- ✅ Maintains correspondence between source and target

## Configuration

### Basic Configuration
```python
dataset = CATsSyntheticDataset(
    num_samples=1000,
    size=256,
    split='trn',
    augmentation=True,
    seed=42,
)
```

### Custom Shaders
```python
custom_shaders = {
    'program0': 'path/to/terrain_sdf.glsl',
    'program1': 'path/to/quatJulia.c',
    'merge': 'path/to/merge.glsl',
}

dataset = CATsSyntheticDataset(
    shaders=custom_shaders,
    num_samples=1000,
    # ... other parameters
)
```

### Training Parameters
```bash
python train_cats.py \
    --num_samples 5000 \
    --epochs 100 \
    --batch-size 32 \
    --lr 3e-4 \
    --lr-backbone 3e-6 \
    --backbone resnet101
```

## Data Flow

1. **OnlineGenerationDataset** generates geometry/normal maps
2. **CATsSyntheticDataset** converts to RGB images
3. **Keypoint generation** based on geometry features
4. **CATs++ format** with proper normalization and augmentation
5. **Training loop** with validation and checkpointing

## Output Format

The wrapper returns batches with these keys:
- `src_img`: Source image tensor (3, H, W)
- `trg_img`: Target image tensor (3, H, W)
- `src_kps`: Source keypoints (2, max_pts)
- `trg_kps`: Target keypoints (2, max_pts)
- `n_pts`: Number of valid points
- `category_id`: Category identifier
- `src_imsize`: Original source image size
- `trg_imsize`: Original target image size

## Troubleshooting

### Common Issues

1. **Segmentation fault with multiprocessing**: 
   - **Cause**: OpenGL contexts cannot be shared across multiple processes
   - **Solution**: Use `--n_threads 0` for single-threaded data loading
   - **Note**: This is the default setting in the training script

2. **Shader path errors**: Ensure shader files exist at specified paths
3. **Memory issues**: Reduce batch size or image size
4. **Import errors**: Check that CATs++ model files are in the correct location

### Debug Mode
```python
dataset = CATsSyntheticDataset(
    num_samples=10,  # Small number for testing
    size=128,        # Smaller size for faster testing
    # ... other parameters
)
```

## Performance Tips

1. **Use appropriate batch sizes** for your GPU memory
2. **Single-threaded data loading** is required due to OpenGL limitations
   - Use `--n_threads 0` (default)
   - Consider larger batch sizes to compensate for slower data loading
3. **Use smaller image sizes** for faster iteration during development
4. **Enable augmentation** for better generalization
5. **GPU memory optimization**: The synthetic data generation happens on CPU, so GPU memory is primarily used for model training

## Next Steps

1. **Test the wrapper**: Run `test_cats_wrapper.py`
2. **Start training**: Use `train_cats.py` with your preferred parameters
3. **Monitor training**: Check TensorBoard logs in the `snapshots/` directory
4. **Fine-tune**: Adjust hyperparameters based on training results

The implementation is designed to be a drop-in replacement for the original CATs++ training, but using your synthetic correspondence data instead of real image pairs.
