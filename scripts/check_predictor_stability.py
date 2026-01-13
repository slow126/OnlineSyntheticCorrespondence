#!/usr/bin/env python3
"""
Check sign stability and effect size consistency across experimental conditions.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def check_sign_consistency(analysis_dir):
    """Check if predictor signs are consistent across encoder configs."""
    analysis_dir = Path(analysis_dir)
    by_encoder_dir = analysis_dir / "by_encoder"
    
    if not by_encoder_dir.exists():
        print(f"No by_encoder directory found at {by_encoder_dir}")
        print("Run analysis with per_encoder=True to generate encoder-specific results")
        return
    
    encoder_dirs = list(by_encoder_dir.glob("*"))
    if not encoder_dirs:
        print(f"No encoder subdirectories found in {by_encoder_dir}")
        return
    
    results = []
    for enc_dir in encoder_dirs:
        slopes_file = enc_dir / "within_benchmark_slopes.csv"
        if not slopes_file.exists():
            continue
        
        slopes_df = pd.read_csv(slopes_file)
        encoder_config = enc_dir.name
        
        for col in slopes_df.columns:
            if col in ["benchmark", "n", "r2", "mode"]:
                continue
            
            signs = np.sign(slopes_df[col].dropna())
            if len(signs) == 0:
                continue
            
            pos_frac = (signs > 0).mean()
            median_coef = slopes_df[col].median()
            
            results.append({
                'encoder_config': encoder_config,
                'predictor': col,
                'positive_fraction': pos_frac,
                'median_coefficient': median_coef,
                'sign_stable': pos_frac > 0.8 or pos_frac < 0.2,
            })
    
    if not results:
        print("No results found. Make sure within_benchmark_slopes.csv files exist.")
        return
    
    result_df = pd.DataFrame(results)
    
    # Check universal consistency
    pivot = result_df.pivot(index='predictor', columns='encoder_config', values='median_coefficient')
    
    print("\n" + "="*80)
    print("PREDICTOR SIGN CONSISTENCY ACROSS ENCODER CONFIGS")
    print("="*80)
    
    for predictor in pivot.index:
        coefs = pivot.loc[predictor].dropna()
        if len(coefs) < 2:
            continue
        
        all_same_sign = (coefs > 0).all() or (coefs < 0).all()
        cv = coefs.std() / abs(coefs.mean()) if coefs.mean() != 0 else np.inf
        
        print(f"\n{predictor}:")
        print(f"  Consistent sign across configs: {all_same_sign}")
        print(f"  Coefficient of variation: {cv:.2f}")
        print(f"  Median coefficients by config:")
        for config, val in coefs.items():
            print(f"    {config}: {val:.4f}")
        
        if not all_same_sign:
            print(f"  ⚠️  WARNING: Sign flips across encoder configs!")
        elif cv > 1.0:
            print(f"  ⚠️  WARNING: High variance in effect size (CV > 1.0)!")
        elif cv < 0.5 and all_same_sign:
            print(f"  ✓ STABLE: Consistent sign and low variance")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Count predictors by stability
    sign_consistent = []
    for predictor in pivot.index:
        coefs = pivot.loc[predictor].dropna()
        if len(coefs) >= 2:
            all_same_sign = (coefs > 0).all() or (coefs < 0).all()
            sign_consistent.append(all_same_sign)
    
    if sign_consistent:
        n_stable = sum(sign_consistent)
        n_total = len(sign_consistent)
        print(f"\nSign-consistent predictors: {n_stable}/{n_total} ({100*n_stable/n_total:.1f}%)")
    
    # Save detailed results
    output_file = analysis_dir / "predictor_stability_report.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return result_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_predictor_stability.py <analysis_dir>")
        print("\nExample:")
        print("  python check_predictor_stability.py analysis/paper_main")
        sys.exit(1)
    
    analysis_dir = Path(sys.argv[1])
    if not analysis_dir.exists():
        print(f"Error: Directory not found: {analysis_dir}")
        sys.exit(1)
    
    check_sign_consistency(analysis_dir)
