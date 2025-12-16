"""
Plot histogram coverage results and compare with precision/recall metrics.

Computes correlation to check colinearity between histogram coverage and
existing soft k-NN precision/recall metrics.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_and_merge_data(histogram_csv: str, coverage_csv: str) -> pd.DataFrame:
    """
    Load histogram coverage and precision/recall CSVs, merge on dataset pairs.
    """
    df_hist = pd.read_csv(histogram_csv)
    df_cov = pd.read_csv(coverage_csv)
    
    # Create merge keys
    df_hist['pair_key'] = df_hist['dataset1'] + '_' + df_hist['split1'] + '__' + \
                          df_hist['dataset2'] + '_' + df_hist['split2']
    df_cov['pair_key'] = df_cov['dataset1'] + '_' + df_cov['split1'] + '__' + \
                         df_cov['dataset2'] + '_' + df_cov['split2']
    
    # Merge
    df = pd.merge(
        df_hist[['pair_key', 'dataset1', 'split1', 'dataset2', 'split2', 'histogram_coverage']],
        df_cov[['pair_key', 'recall', 'precision']],
        on='pair_key',
        how='inner'
    )
    
    return df


def plot_colinearity(df: pd.DataFrame, output_prefix: str):
    """
    Create scatter plots comparing histogram coverage to recall and precision.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram coverage vs Recall
    ax = axes[0]
    ax.scatter(df['histogram_coverage'], df['recall'], alpha=0.7, edgecolor='k', linewidth=0.5)
    
    # Fit line
    if len(df) > 2:
        slope, intercept, r, p, se = stats.linregress(df['histogram_coverage'], df['recall'])
        x_line = np.linspace(df['histogram_coverage'].min(), df['histogram_coverage'].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r--', 
                label=f'r={r:.3f}, p={p:.3e}')
        ax.legend()
    
    ax.set_xlabel('Histogram Coverage')
    ax.set_ylabel('Recall (soft k-NN)')
    ax.set_title('Histogram Coverage vs Recall')
    ax.grid(True, alpha=0.3)
    
    # Histogram coverage vs Precision
    ax = axes[1]
    ax.scatter(df['histogram_coverage'], df['precision'], alpha=0.7, edgecolor='k', linewidth=0.5)
    
    if len(df) > 2:
        slope, intercept, r, p, se = stats.linregress(df['histogram_coverage'], df['precision'])
        x_line = np.linspace(df['histogram_coverage'].min(), df['histogram_coverage'].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r--',
                label=f'r={r:.3f}, p={p:.3e}')
        ax.legend()
    
    ax.set_xlabel('Histogram Coverage')
    ax.set_ylabel('Precision (soft k-NN)')
    ax.set_title('Histogram Coverage vs Precision')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_colinearity.png', dpi=150)
    plt.savefig(f'{output_prefix}_colinearity.pdf')
    print(f"Saved: {output_prefix}_colinearity.png/pdf")
    plt.close()


def print_correlation_summary(df: pd.DataFrame):
    """Print correlation statistics."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    if len(df) < 3:
        print("Not enough data points for correlation analysis")
        return
    
    # Pearson correlation
    r_recall, p_recall = stats.pearsonr(df['histogram_coverage'], df['recall'])
    r_prec, p_prec = stats.pearsonr(df['histogram_coverage'], df['precision'])
    
    print(f"\nPearson correlation (histogram_coverage vs recall):")
    print(f"  r = {r_recall:.4f}, p = {p_recall:.4e}")
    
    print(f"\nPearson correlation (histogram_coverage vs precision):")
    print(f"  r = {r_prec:.4f}, p = {p_prec:.4e}")
    
    # Spearman (rank) correlation
    rho_recall, p_rho_recall = stats.spearmanr(df['histogram_coverage'], df['recall'])
    rho_prec, p_rho_prec = stats.spearmanr(df['histogram_coverage'], df['precision'])
    
    print(f"\nSpearman correlation (histogram_coverage vs recall):")
    print(f"  rho = {rho_recall:.4f}, p = {p_rho_recall:.4e}")
    
    print(f"\nSpearman correlation (histogram_coverage vs precision):")
    print(f"  rho = {rho_prec:.4f}, p = {p_rho_prec:.4e}")
    
    print("\n" + "="*60)
    if abs(r_recall) > 0.8 or abs(r_prec) > 0.8:
        print("HIGH COLINEARITY DETECTED - histogram coverage may be redundant")
    elif abs(r_recall) > 0.5 or abs(r_prec) > 0.5:
        print("MODERATE COLINEARITY - some overlap with existing metrics")
    else:
        print("LOW COLINEARITY - histogram coverage captures different information")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Plot histogram coverage vs precision/recall for colinearity analysis'
    )
    parser.add_argument(
        '--histogram-csv', type=str, default='histogram_coverage_results.csv',
        help='CSV file with histogram coverage results'
    )
    parser.add_argument(
        '--coverage-csv', type=str, default='coverage_results.csv',
        help='CSV file with precision/recall results'
    )
    parser.add_argument(
        '--output-prefix', type=str, default='histogram_analysis',
        help='Prefix for output plot files'
    )
    args = parser.parse_args()
    
    print(f"Loading histogram coverage from: {args.histogram_csv}")
    print(f"Loading precision/recall from: {args.coverage_csv}")
    
    df = load_and_merge_data(args.histogram_csv, args.coverage_csv)
    
    print(f"\nMerged {len(df)} dataset pairs")
    
    if len(df) == 0:
        print("ERROR: No matching pairs found between CSVs")
        return
    
    print("\nMatched pairs:")
    for _, row in df.iterrows():
        print(f"  {row['dataset1']}_{row['split1']} -> {row['dataset2']}_{row['split2']}: "
              f"hist_cov={row['histogram_coverage']:.4f}, "
              f"recall={row['recall']:.4f}, precision={row['precision']:.4f}")
    
    print_correlation_summary(df)
    plot_colinearity(df, args.output_prefix)


if __name__ == "__main__":
    main()
