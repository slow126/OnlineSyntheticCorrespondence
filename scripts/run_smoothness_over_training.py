#!/usr/bin/env python3
"""
Run smoothness analysis across training checkpoints and plot trends over time.

This script:
1. Scans snapshot directories for checkpoint files
2. Runs smoothness evaluation on each checkpoint
3. Merges smoothness with validation PCK by training step
4. Produces line plots for smoothness and PCK over training steps
"""

import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def parse_checkpoint_metadata(checkpoint_path: Path):
    name = checkpoint_path.name
    step_match = re.search(r"(?:step|steps|iter|iteration)[= _-]?(\d+)", name)
    epoch_match = re.search(r"epoch[= _-]?(\d+)", name)
    step = int(step_match.group(1)) if step_match else None
    epoch = int(epoch_match.group(1)) if epoch_match else None
    return step, epoch


def collect_snapshot_dirs(snapshot_roots):
    snapshot_dirs = []
    for root in snapshot_roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"Warning: Directory not found: {root_path}")
            continue
        if (root_path / "validation_results.csv").exists() or (root_path / "checkpoints").exists():
            snapshot_dirs.append(root_path)
            continue
        for child in root_path.iterdir():
            if child.is_dir():
                snapshot_dirs.append(child)
    return snapshot_dirs


def collect_checkpoints(snapshot_dir: Path, patterns):
    checkpoint_paths = []
    for pattern in patterns:
        checkpoint_paths.extend(snapshot_dir.glob(pattern))
    # De-duplicate and filter invalid files
    unique_paths = []
    seen = set()
    for path in checkpoint_paths:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file() and path.stat().st_size > 0:
            unique_paths.append(path)

    def sort_key(path):
        step, epoch = parse_checkpoint_metadata(path)
        if step is not None:
            return (0, step)
        if epoch is not None:
            return (1, epoch)
        return (2, path.stat().st_mtime)

    unique_paths.sort(key=sort_key)
    return unique_paths


def load_pck_results(snapshot_dir: Path):
    val_path = snapshot_dir / "validation_results.csv"
    if not val_path.exists():
        return pd.DataFrame()

    try:
        val_df = pd.read_csv(val_path)
    except Exception as exc:
        print(f"Warning: Could not read {val_path}: {exc}")
        return pd.DataFrame()

    required_cols = {"benchmark", "training_steps", "pck"}
    if not required_cols.issubset(set(val_df.columns)):
        print(f"Warning: Missing columns in {val_path}: {sorted(required_cols)}")
        return pd.DataFrame()

    pck_df = val_df[["benchmark", "training_steps", "pck"]].copy()
    pck_df["training_steps"] = pd.to_numeric(pck_df["training_steps"], errors="coerce")
    pck_df = pck_df.dropna(subset=["training_steps"])
    pck_df["training_steps"] = pck_df["training_steps"].astype("Int64")
    pck_df = pck_df.groupby(["benchmark", "training_steps"], as_index=False)["pck"].max()
    return pck_df


def run_smoothness(checkpoint_csv, output_csv, args):
    cmd = [
        "python",
        "scripts/calculate_flow_smoothness.py",
        "--checkpoints", str(checkpoint_csv),
        "--benchmarks",
    ] + args.benchmarks + [
        "--output", str(output_csv),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--use-checkpoint-paths",
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.include_gt:
        cmd.append("--include-gt")
    if args.mask_by_gt:
        cmd.append("--mask-by-gt")
    if args.tss_root:
        cmd += ["--tss-root", str(args.tss_root)]

    print(f"\nRunning smoothness evaluation on {checkpoint_csv}...")
    print(f"Command: {' '.join(cmd)}\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def plot_smoothness(bench_df, output_path, snapshot_name, benchmark):
    plot_df = bench_df.copy()
    plot_df["training_steps"] = pd.to_numeric(plot_df["training_steps"], errors="coerce")
    plot_df = plot_df.dropna(subset=["training_steps"]).sort_values("training_steps")

    if plot_df.empty:
        print(f"Warning: No training steps for {snapshot_name} ({benchmark})")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{snapshot_name} - {benchmark} Smoothness", fontsize=14, fontweight="bold")

    axes[0].plot(plot_df["training_steps"], plot_df["mean_tv"], marker="o", linewidth=2)
    axes[0].set_ylabel("Total Variation (lower = smoother)")
    axes[0].grid(True, alpha=0.3)
    gt_tv = plot_df["mean_tv_gt"].dropna().iloc[0] if "mean_tv_gt" in plot_df.columns and plot_df["mean_tv_gt"].notna().any() else None
    if gt_tv is not None:
        axes[0].axhline(gt_tv, color="gray", linestyle="--", linewidth=1.5, label="GT")
        axes[0].legend()

    axes[1].plot(plot_df["training_steps"], plot_df["mean_laplacian"], marker="o", linewidth=2, color="#ff7f0e")
    axes[1].set_ylabel("Laplacian (lower = smoother)")
    axes[1].set_xlabel("Training steps")
    axes[1].grid(True, alpha=0.3)
    gt_lap = plot_df["mean_laplacian_gt"].dropna().iloc[0] if "mean_laplacian_gt" in plot_df.columns and plot_df["mean_laplacian_gt"].notna().any() else None
    if gt_lap is not None:
        axes[1].axhline(gt_lap, color="gray", linestyle="--", linewidth=1.5, label="GT")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pck(bench_df, output_path, snapshot_name, benchmark):
    plot_df = bench_df.copy()
    plot_df["training_steps"] = pd.to_numeric(plot_df["training_steps"], errors="coerce")
    plot_df = plot_df.dropna(subset=["training_steps", "pck"]).sort_values("training_steps")

    if plot_df.empty:
        print(f"Warning: No PCK values for {snapshot_name} ({benchmark})")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(plot_df["training_steps"], plot_df["pck"], marker="o", linewidth=2, color="#2ca02c")
    ax.set_title(f"{snapshot_name} - {benchmark} PCK", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("PCK (higher = better)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run smoothness-over-training analysis with line plots."
    )
    parser.add_argument("--snapshot-dirs", type=str, nargs="+", required=True,
                        help="Directories containing snapshot subdirectories")
    parser.add_argument("--output-dir", type=str, default="analysis/smoothness_over_training",
                        help="Output directory for results")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["spair"],
                        help="Benchmarks to evaluate (default: spair)")
    parser.add_argument("--checkpoint-glob", type=str,
                        default="checkpoints/*.ckpt,checkpoints/*.pth",
                        help="Comma-separated checkpoint glob patterns (relative to snapshot dir)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (optional)")
    parser.add_argument("--include-gt", action="store_true",
                        help="Also compute smoothness on ground-truth flow when available.")
    parser.add_argument("--mask-by-gt", action="store_true",
                        help="Compute smoothness only over valid GT pixels (mask invalid regions).")
    parser.add_argument("--tss-root", type=str, default=None,
                        help="Path to TSS dataset root (overrides config).")
    parser.add_argument("--skip-calculation", action="store_true",
                        help="Skip smoothness calculation (use existing results)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dirs = collect_snapshot_dirs(args.snapshot_dirs)
    if not snapshot_dirs:
        raise ValueError("No snapshot directories found")

    patterns = [p.strip() for p in args.checkpoint_glob.split(",") if p.strip()]
    checkpoint_rows = []

    print(f"Scanning {len(snapshot_dirs)} snapshot directories...")
    for snapshot_dir in snapshot_dirs:
        checkpoints = collect_checkpoints(snapshot_dir, patterns)
        if not checkpoints:
            print(f"Warning: No checkpoints found in {snapshot_dir}")
            continue
        for checkpoint_path in checkpoints:
            training_steps, epoch = parse_checkpoint_metadata(checkpoint_path)
            checkpoint_rows.append({
                "checkpoint_path": str(checkpoint_path),
                "snapshot_dir": str(snapshot_dir),
                "snapshot_name": snapshot_dir.name,
                "checkpoint_file": checkpoint_path.name,
                "training_steps": training_steps,
                "epoch": epoch,
            })

    if not checkpoint_rows:
        raise ValueError("No checkpoints found to evaluate")

    checkpoints_df = pd.DataFrame(checkpoint_rows)
    checkpoints_csv = output_dir / "checkpoints_over_training.csv"
    checkpoints_df.to_csv(checkpoints_csv, index=False)
    print(f"Saved checkpoint list to: {checkpoints_csv}")

    smoothness_csv = output_dir / "smoothness_raw_results_over_training.csv"
    if not args.skip_calculation:
        run_smoothness(checkpoints_csv, smoothness_csv, args)
    else:
        if not smoothness_csv.exists():
            raise FileNotFoundError(f"No existing results found at {smoothness_csv}")

    smoothness_df = pd.read_csv(smoothness_csv)
    merged_df = smoothness_df.merge(checkpoints_df, on="checkpoint_path", how="left")
    merged_csv = output_dir / "smoothness_over_training_all.csv"
    merged_df.to_csv(merged_csv, index=False)
    print(f"Saved merged results to: {merged_csv}")

    for snapshot_name in sorted(merged_df["snapshot_name"].dropna().unique()):
        snapshot_df = merged_df[merged_df["snapshot_name"] == snapshot_name].copy()
        snapshot_df["training_steps"] = pd.to_numeric(snapshot_df["training_steps"], errors="coerce").astype("Int64")
        snapshot_dir = Path(snapshot_df["snapshot_dir"].dropna().iloc[0])
        snapshot_out = output_dir / snapshot_name
        snapshot_out.mkdir(parents=True, exist_ok=True)

        pck_df = load_pck_results(snapshot_dir)
        if not pck_df.empty:
            snapshot_df = snapshot_df.merge(
                pck_df, on=["benchmark", "training_steps"], how="left"
            )

        snapshot_csv = snapshot_out / "smoothness_over_training.csv"
        snapshot_df.to_csv(snapshot_csv, index=False)

        for benchmark in args.benchmarks:
            bench_df = snapshot_df[snapshot_df["benchmark"] == benchmark].copy()
            if bench_df.empty:
                continue

            smoothness_plot = snapshot_out / f"smoothness_over_training_{benchmark}.png"
            plot_smoothness(bench_df, smoothness_plot, snapshot_name, benchmark)

            pck_plot = snapshot_out / f"pck_over_training_{benchmark}.png"
            plot_pck(bench_df, pck_plot, snapshot_name, benchmark)

    print(f"\nDone. Results saved under: {output_dir}")


if __name__ == "__main__":
    main()
