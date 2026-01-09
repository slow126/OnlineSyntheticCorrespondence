#!/usr/bin/env python3
"""
Download script for ImageNet100 subset.

ImageNet100 is a subset of ImageNet with 100 classes.
This script helps download and organize ImageNet100 data.

Note: ImageNet requires registration and acceptance of terms.
You can download ImageNet100 using:
1. Official ImageNet download (requires registration)
2. ImageNet-Datasets-Downloader tool
3. Pre-downloaded ImageNet data

Usage:
    python scripts/download_imagenet100.py --root ~/Data/imagenet100
    python scripts/download_imagenet100.py --root ~/Data/imagenet100 --verify-only
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess


def check_directory_exists(path: Path) -> bool:
    """Check if a directory exists."""
    return path.exists() and path.is_dir()


def verify_imagenet100_structure(root_dir: Path) -> bool:
    """
    Verify that ImageNet100 directory structure is correct.
    
    Expected structure:
        root_dir/
            train/
                n01440764/  (class directories)
                    *.JPEG
                ...
            val/
                n01440764/
                    *.JPEG
                ...
    """
    print(f"\nVerifying ImageNet100 structure at: {root_dir}")
    
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    
    if not train_dir.exists():
        print(f"ERROR: Train directory not found: {train_dir}")
        return False
    
    if not val_dir.exists():
        print(f"ERROR: Val directory not found: {val_dir}")
        return False
    
    # Count classes
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    val_classes = [d for d in val_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(train_classes)} train classes")
    print(f"Found {len(val_classes)} val classes")
    
    if len(train_classes) == 0:
        print("ERROR: No train classes found")
        return False
    
    if len(val_classes) == 0:
        print("ERROR: No val classes found")
        return False
    
    # Count images
    train_images = sum(len(list(c.glob("*.JPEG"))) + len(list(c.glob("*.jpg"))) for c in train_classes)
    val_images = sum(len(list(c.glob("*.JPEG"))) + len(list(c.glob("*.jpg"))) for c in val_classes)
    
    print(f"Found {train_images} train images")
    print(f"Found {val_images} val images")
    
    if train_images == 0:
        print("ERROR: No train images found")
        return False
    
    if val_images == 0:
        print("ERROR: No val images found")
        return False
    
    print("✓ ImageNet100 structure verified")
    return True


def download_with_imagenet_downloader(root_dir: Path) -> bool:
    """
    Attempt to download using ImageNet-Datasets-Downloader.
    
    This requires the tool to be installed:
        pip install ImageNet-Datasets-Downloader
    """
    try:
        import imagenet_downloader
    except ImportError:
        print("\nImageNet-Datasets-Downloader not found.")
        print("Install it with: pip install ImageNet-Datasets-Downloader")
        return False
    
    print("\n" + "="*80)
    print("Downloading ImageNet100 using ImageNet-Datasets-Downloader")
    print("="*80)
    print("\nNote: This requires ImageNet registration and acceptance of terms.")
    print("You will need to provide your ImageNet username and access key.")
    
    # Create root directory
    root_dir.mkdir(parents=True, exist_ok=True)
    
    # ImageNet100 class list (first 100 classes from ImageNet)
    # We'll use a subset - for full ImageNet100, you'd need the complete class list
    print("\nFor ImageNet100, you need to specify 100 classes.")
    print("This script provides a framework - you may need to customize the class list.")
    
    return True


def print_instructions(root_dir: Path):
    """Print instructions for manual download."""
    print("\n" + "="*80)
    print("ImageNet100 Download Instructions")
    print("="*80)
    print(f"\nTarget directory: {root_dir}")
    print("\nImageNet requires registration and acceptance of terms.")
    print("Visit: https://www.image-net.org/download.php")
    print("\nAfter downloading ImageNet, organize it as:")
    print(f"  {root_dir}/")
    print(f"    train/")
    print(f"      n01440764/  (class directories)")
    print(f"        *.JPEG")
    print(f"      ...")
    print(f"    val/")
    print(f"      n01440764/")
    print(f"        *.JPEG")
    print(f"      ...")
    print("\nImageNet100 uses the first 100 classes from ImageNet.")
    print("You can filter the full ImageNet dataset to get ImageNet100.")
    print("\nAlternatively, use ImageNet-Datasets-Downloader:")
    print("  pip install ImageNet-Datasets-Downloader")
    print("  # Then follow the tool's instructions")


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify ImageNet100 dataset"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory for ImageNet100 dataset"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset, don't download"
    )
    parser.add_argument(
        "--use-downloader",
        action="store_true",
        help="Attempt to use ImageNet-Datasets-Downloader (requires installation)"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root).expanduser().resolve()
    
    if args.verify_only:
        if verify_imagenet100_structure(root_dir):
            print("\n✓ ImageNet100 dataset is ready")
            return 0
        else:
            print("\n✗ ImageNet100 dataset verification failed")
            return 1
    
    # Check if already exists
    if root_dir.exists() and verify_imagenet100_structure(root_dir):
        print("\n✓ ImageNet100 dataset already exists and is valid")
        return 0
    
    # Try to download
    if args.use_downloader:
        if download_with_imagenet_downloader(root_dir):
            if verify_imagenet100_structure(root_dir):
                print("\n✓ ImageNet100 download completed successfully")
                return 0
    
    # Print instructions
    print_instructions(root_dir)
    
    print("\nAfter downloading, verify with:")
    print(f"  python scripts/download_imagenet100.py --root {root_dir} --verify-only")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

