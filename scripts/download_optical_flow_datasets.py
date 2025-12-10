#!/usr/bin/env python3
"""
Download script for optical flow datasets.

Supports downloading and verifying:
- Monkaa (SceneFlow)
- Driving (SceneFlow)
- MPI Sintel
- Virtual KITTI 2
- HD1K

Usage:
    python scripts/download_optical_flow_datasets.py --dataset monkaa --root ~/Data
    python scripts/download_optical_flow_datasets.py --all --root ~/Data
    python scripts/download_optical_flow_datasets.py --dataset virtualkitti2 --root ~/Data
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess


def check_directory_exists(path: Path) -> bool:
    """Check if a directory exists."""
    return path.exists() and path.is_dir()


def run_command(cmd: str, cwd: Path = None) -> bool:
    """Run a shell command and return success status."""
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False


def download_monkaa(root_dir: Path) -> bool:
    """
    Download Monkaa dataset (part of SceneFlow).
    
    Dataset link: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    """
    print("\n" + "="*80)
    print("Downloading Monkaa Dataset (SceneFlow)")
    print("="*80)
    
    monkaa_path = root_dir / "SceneFlow" / "Monkaa"
    
    if check_directory_exists(monkaa_path):
        print(f"✓ Monkaa dataset already exists at: {monkaa_path}")
        return True
    
    print("\nMonkaa is part of the SceneFlow dataset.")
    print("Manual download required from:")
    print("https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html")
    print("\nDownload and extract:")
    print("  - Monkaa RGB images (cleanpass)")
    print("  - Monkaa optical flow")
    print(f"\nExtract to: {root_dir / 'SceneFlow' / 'Monkaa'}")
    print("\nExpected structure:")
    print("  SceneFlow/Monkaa/frames_cleanpass/")
    print("  SceneFlow/Monkaa/optical_flow/")
    
    return False


def download_driving(root_dir: Path) -> bool:
    """
    Download Driving dataset (part of SceneFlow).
    
    Dataset link: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    """
    print("\n" + "="*80)
    print("Downloading Driving Dataset (SceneFlow)")
    print("="*80)
    
    driving_path = root_dir / "SceneFlow" / "Driving"
    
    if check_directory_exists(driving_path):
        print(f"✓ Driving dataset already exists at: {driving_path}")
        return True
    
    print("\nDriving is part of the SceneFlow dataset.")
    print("Manual download required from:")
    print("https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html")
    print("\nDownload and extract:")
    print("  - Driving RGB images (cleanpass)")
    print("  - Driving optical flow")
    print(f"\nExtract to: {root_dir / 'SceneFlow' / 'Driving'}")
    print("\nExpected structure:")
    print("  SceneFlow/Driving/frames_cleanpass/")
    print("  SceneFlow/Driving/optical_flow/")
    
    return False


def download_sintel(root_dir: Path) -> bool:
    """
    Download MPI Sintel dataset.
    
    Dataset link: http://sintel.is.tue.mpg.de/
    Direct download: http://sintel.cs.washington.edu/MPI-Sintel-complete.zip
    """
    print("\n" + "="*80)
    print("Downloading MPI Sintel Dataset")
    print("="*80)
    
    sintel_path = root_dir / "Sintel"
    
    if check_directory_exists(sintel_path / "training"):
        print(f"✓ Sintel dataset already exists at: {sintel_path}")
        return True
    
    print("\nMPI Sintel dataset - attempting automatic download...")
    
    # Create directory
    root_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file = root_dir / "MPI-Sintel-complete.zip"
    
    if not zip_file.exists():
        print(f"\nDownloading to {zip_file}...")
        cmd = f"wget -O {zip_file} http://sintel.cs.washington.edu/MPI-Sintel-complete.zip"
        if not run_command(cmd, cwd=root_dir):
            print("\n✗ Download failed. You can download manually:")
            print("  http://sintel.cs.washington.edu/MPI-Sintel-complete.zip")
            return False
    
    print(f"\nExtracting to {sintel_path}...")
    cmd = f"unzip -q {zip_file} -d {root_dir}"
    if not run_command(cmd, cwd=root_dir):
        print("\n✗ Extraction failed.")
        return False
    
    # Check if extraction succeeded
    if check_directory_exists(sintel_path / "training"):
        print(f"\n✓ Sintel dataset successfully downloaded and extracted to: {sintel_path}")
        print("\nExpected structure:")
        print("  Sintel/training/clean/")
        print("  Sintel/training/final/")
        print("  Sintel/training/flow/")
        return True
    else:
        print("\n✗ Extraction succeeded but expected structure not found.")
        return False


def download_virtual_kitti2(root_dir: Path) -> bool:
    """
    Download Virtual KITTI 2 dataset.
    
    Dataset link: https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
    Direct downloads available.
    """
    print("\n" + "="*80)
    print("Downloading Virtual KITTI 2 Dataset")
    print("="*80)
    
    vkitti2_path = root_dir / "vkitti_2.0.3"
    
    if check_directory_exists(vkitti2_path):
        print(f"✓ Virtual KITTI 2 dataset already exists at: {vkitti2_path}")
        return True
    
    print("\nVirtual KITTI 2 dataset - attempting automatic download...")
    print("This will download RGB images and forward optical flow.")
    print("Total size: ~30GB")
    
    # Create directory
    root_dir.mkdir(parents=True, exist_ok=True)
    
    # Download URLs
    downloads = [
        ("vkitti_2.0.3_rgb.tar", "https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar"),
        ("vkitti_2.0.3_forwardFlow.tar", "https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_forwardFlow.tar"),
    ]
    
    # Optional downloads (commented out by default)
    optional_downloads = [
        ("vkitti_2.0.3_backwardFlow.tar", "https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_backwardFlow.tar"),
        ("vkitti_2.0.3_forwardSceneFlow.tar", "https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_forwardSceneFlow.tar"),
        ("vkitti_2.0.3_backwardSceneFlow.tar", "https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_backwardSceneFlow.tar"),
    ]
    
    print("\nOptional: To download backward flow and scene flow, uncomment in the script.")
    print("\nDownloading required files (RGB + forward flow)...")
    
    all_success = True
    for filename, url in downloads:
        tar_file = root_dir / filename
        
        if tar_file.exists():
            print(f"\n✓ {filename} already downloaded, skipping...")
        else:
            print(f"\nDownloading {filename}...")
            print(f"  URL: {url}")
            cmd = f"wget -O {tar_file} {url}"
            if not run_command(cmd, cwd=root_dir):
                print(f"✗ Failed to download {filename}")
                all_success = False
                continue
        
        # Extract
        print(f"Extracting {filename}...")
        cmd = f"tar -xf {tar_file} -C {root_dir}"
        if not run_command(cmd, cwd=root_dir):
            print(f"✗ Failed to extract {filename}")
            all_success = False
        else:
            print(f"✓ {filename} extracted successfully")
    
    if all_success and check_directory_exists(vkitti2_path):
        print(f"\n✓ Virtual KITTI 2 dataset successfully downloaded to: {vkitti2_path}")
        print("\nExpected structure:")
        print("  vkitti_2.0.3/Scene01/clone/frames/rgb/Camera_0/")
        print("  vkitti_2.0.3/Scene01/clone/frames/forwardFlow/Camera_0/")
        return True
    else:
        print("\n✗ Download incomplete or failed.")
        return False


def download_hd1k(root_dir: Path) -> bool:
    """
    Download HD1K dataset.
    
    Dataset link: http://hci-benchmark.org/
    """
    print("\n" + "="*80)
    print("Downloading HD1K Dataset")
    print("="*80)
    
    hd1k_path = root_dir / "HD1K"
    
    if check_directory_exists(hd1k_path):
        print(f"✓ HD1K dataset already exists at: {hd1k_path}")
        return True
    
    print("\nHD1K dataset download:")
    print("Official website: http://hci-benchmark.org/")
    print("Alternative: https://hci.iwr.uni-heidelberg.de/content/hd1k-benchmark-suite")
    print("\nDownload:")
    print("  - HD1K training images")
    print("  - HD1K optical flow")
    print(f"\nExtract to: {root_dir / 'HD1K'}")
    print("\nExpected structure:")
    print("  HD1K/hd1k_input/")
    print("  HD1K/hd1k_flow_gt/")
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download optical flow datasets for CATs++ training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Sintel dataset (automatic)
  python scripts/download_optical_flow_datasets.py --dataset sintel --root ~/Data
  
  # Download Virtual KITTI 2 (automatic)
  python scripts/download_optical_flow_datasets.py --dataset virtualkitti2 --root ~/Data
  
  # Check which datasets are already downloaded
  python scripts/download_optical_flow_datasets.py --check --root ~/Data

Supported datasets:
  - monkaa: Monkaa (SceneFlow) - manual download
  - driving: Driving (SceneFlow) - manual download
  - sintel: MPI Sintel - automatic download
  - virtualkitti2: Virtual KITTI 2 - automatic download
  - hd1k: HD1K - manual download
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["monkaa", "driving", "sintel", "virtualkitti2", "hd1k"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets (or show instructions)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which datasets are already downloaded"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory for datasets (e.g., ~/Data)"
    )
    
    args = parser.parse_args()
    
    # Expand ~ in path
    root_dir = Path(args.root).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDataset root directory: {root_dir}")
    
    # Define dataset download functions
    datasets = {
        "monkaa": download_monkaa,
        "driving": download_driving,
        "sintel": download_sintel,
        "virtualkitti2": download_virtual_kitti2,
        "hd1k": download_hd1k,
    }
    
    if args.check:
        print("\n" + "="*80)
        print("Checking dataset availability")
        print("="*80)
        for name, download_fn in datasets.items():
            download_fn(root_dir)
        return
    
    if args.all:
        print("\nDownloading/checking all datasets...")
        results = {}
        for name, download_fn in datasets.items():
            results[name] = download_fn(root_dir)
        
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        for name, success in results.items():
            status = "✓ Ready" if success else "✗ Manual download required"
            print(f"{name:20s}: {status}")
    
    elif args.dataset:
        download_fn = datasets[args.dataset]
        success = download_fn(root_dir)
        if not success:
            print("\n⚠ Manual steps required. See instructions above.")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
