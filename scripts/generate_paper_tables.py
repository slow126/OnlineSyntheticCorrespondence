#!/usr/bin/env python3
"""Generate publication-ready tables from clean analysis."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def generate_table1_family_comparison(stage1_dir):
    """Table 1: Univariate predictor family comparison."""
    stage1_dir = Path(stage1_dir)
    families = ['flow_only', 'dino_only', 'mmd_only']
    rows = []
    
    print("\n" + "="*80)
    print("TABLE 1: Univariate Predictor Family Comparison")
    print("="*80)
    
    for family in families:
        family_dir = stage1_dir / family
        summary_file = family_dir / "prediction_lobo_summary.csv"
        rank_summary_file = family_dir / "prediction_lobo_rank_summary.csv"
        
        if not summary_file.exists():
            print(f"Warning: {summary_file} not found, skipping {family}")
            continue
        
        summary = pd.read_csv(summary_file)
        overall = summary[summary['benchmark'] == '__overall__']
        
        if overall.empty:
            print(f"Warning: No overall results in {summary_file}, skipping {family}")
            continue
        
        overall = overall.iloc[0]
        
        row = {
            'Predictor Family': family.replace('_only', '').upper(),
            'LOBO Pearson': f"{overall['pearson']:.3f}",
            'LOBO Spearman': f"{overall['spearman']:.3f}",
            'RMSE': f"{overall['rmse']:.3f}",
            'N Test': int(overall['n_test']),
        }
        
        # Add ranking metrics if available
        if rank_summary_file.exists():
            rank_summary = pd.read_csv(rank_summary_file)
            overall_rank = rank_summary[rank_summary['benchmark'] == '__overall__']
            
            if not overall_rank.empty:
                overall_rank = overall_rank.iloc[0]
                row['Top-1 Accuracy'] = f"{overall_rank['top1']:.2f}"
                row['Top-3 Accuracy'] = f"{overall_rank['top3']:.2f}"
                row['Rank Spearman'] = f"{overall_rank['spearman']:.3f}"
        
        rows.append(row)
    
    if not rows:
        print("No results found. Make sure Stage 1 has been run.")
        return None
    
    table1 = pd.DataFrame(rows)
    print("\n" + table1.to_markdown(index=False))
    
    output_file = stage1_dir / "table1_family_comparison.csv"
    table1.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    return table1


def generate_table2_stable_predictors(stage2_dir):
    """Table 2: Stable predictors from each family."""
    stage2_dir = Path(stage2_dir)
    
    print("\n" + "="*80)
    print("TABLE 2: Stability Selection Results")
    print("="*80)
    
    for family_name in ['flow', 'dino']:
        family_dir = stage2_dir / family_name
        stability_file = family_dir / "stability_selection_results.csv"
        
        if not stability_file.exists():
            print(f"\nWarning: {stability_file} not found, skipping {family_name}")
            continue
        
        stability_df = pd.read_csv(stability_file)
        
        print(f"\n{family_name.upper()} Predictors:")
        print("-" * 80)
        
        # Show top 5 by stability score
        top_df = stability_df[['predictor', 'stability_score', 'stable', 'selection_count']].head(5)
        print(top_df.to_markdown(index=False))
        
        # Summary statistics
        n_stable = stability_df['stable'].sum()
        n_total = len(stability_df)
        print(f"\nStable predictors (threshold ≥ 0.7): {n_stable}/{n_total}")
        
        if n_stable > 0:
            stable_preds = stability_df[stability_df['stable']]['predictor'].tolist()
            print(f"Stable predictor list: {', '.join(stable_preds)}")


def generate_table3_universality(paper_dir):
    """Table 3: Effect consistency across encoder configs."""
    paper_dir = Path(paper_dir)
    by_encoder_dir = paper_dir / "by_encoder"
    
    print("\n" + "="*80)
    print("TABLE 3: Universality Across Encoder Configs")
    print("="*80)
    
    if not by_encoder_dir.exists():
        print(f"Warning: {by_encoder_dir} not found. Run with per_encoder=True.")
        return None
    
    encoder_configs = list(by_encoder_dir.glob("*"))
    if not encoder_configs:
        print(f"Warning: No encoder subdirectories found in {by_encoder_dir}")
        return None
    
    rows = []
    for enc_dir in sorted(encoder_configs):
        summary_file = enc_dir / "prediction_lobo_summary.csv"
        if not summary_file.exists():
            continue
        
        summary = pd.read_csv(summary_file)
        overall = summary[summary['benchmark'] == '__overall__']
        
        if overall.empty:
            continue
        
        overall = overall.iloc[0]
        
        # Parse config name
        config_name = enc_dir.name.replace('pretrained', 'P').replace('freeze', 'F')
        config_name = config_name.replace('True', 'T').replace('False', 'F').replace('_', '')
        
        rows.append({
            'Encoder Config': config_name,
            'N Test': int(overall['n_test']),
            'LOBO Pearson': f"{overall['pearson']:.3f}",
            'LOBO Spearman': f"{overall['spearman']:.3f}",
            'RMSE': f"{overall['rmse']:.3f}",
        })
    
    if not rows:
        print("No results found.")
        return None
    
    table3 = pd.DataFrame(rows)
    print("\n" + table3.to_markdown(index=False))
    
    # Calculate statistics
    pearsons = [float(r['LOBO Pearson']) for r in rows]
    spearmans = [float(r['LOBO Spearman']) for r in rows]
    
    print("\nConsistency Statistics:")
    print(f"  Pearson range: [{min(pearsons):.3f}, {max(pearsons):.3f}] (Δ={max(pearsons)-min(pearsons):.3f})")
    print(f"  Spearman range: [{min(spearmans):.3f}, {max(spearmans):.3f}] (Δ={max(spearmans)-min(spearmans):.3f})")
    
    all_positive_pearson = all(p > 0 for p in pearsons)
    all_positive_spearman = all(s > 0 for s in spearmans)
    
    if all_positive_pearson and all_positive_spearman:
        print("  ✓ All correlations positive across configs")
    else:
        print("  ⚠️  Warning: Correlations change sign across configs")
    
    output_file = paper_dir / "table3_universality.csv"
    table3.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    return table3


def generate_all_tables(base_dir):
    """Generate all publication tables."""
    base_dir = Path(base_dir)
    
    # Table 1: Family comparison
    stage1_dir = base_dir / "clean" / "stage1_univariate"
    if stage1_dir.exists():
        generate_table1_family_comparison(stage1_dir)
    else:
        print(f"Stage 1 directory not found: {stage1_dir}")
    
    # Table 2: Stability selection
    stage2_dir = base_dir / "clean" / "stage2_stability"
    if stage2_dir.exists():
        generate_table2_stable_predictors(stage2_dir)
    else:
        print(f"Stage 2 directory not found: {stage2_dir}")
    
    # Table 3: Universality
    paper_dir = base_dir / "paper_main"
    if paper_dir.exists():
        generate_table3_universality(paper_dir)
    else:
        print(f"Paper directory not found: {paper_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_paper_tables.py <analysis_base_dir>")
        print("\nExample:")
        print("  python generate_paper_tables.py analysis")
        print("\nThis will look for:")
        print("  - analysis/clean/stage1_univariate/*/")
        print("  - analysis/clean/stage2_stability/*/")
        print("  - analysis/paper_main/")
        sys.exit(1)
    
    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)
    
    generate_all_tables(base_dir)
