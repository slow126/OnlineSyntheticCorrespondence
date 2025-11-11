#!/usr/bin/env python3
"""
Quick script to find all pointodyssey experiments and extract max PCK for pointodyssey and spair.
"""

import os
import csv
from pathlib import Path
from collections import defaultdict

snapshots_dir = Path("/home/spencer/Correspondence_logs/rc/snapshots")

# Find all pointodyssey experiments
pointodyssey_experiments = []
for exp_dir in snapshots_dir.iterdir():
    if not exp_dir.is_dir():
        continue
    exp_name = exp_dir.name
    # Check if it's a pointodyssey experiment (name contains pointodyssey)
    if 'pointodyssey' in exp_name.lower():
        pointodyssey_experiments.append(exp_dir)

print(f"Found {len(pointodyssey_experiments)} pointodyssey experiments\n")

# Extract max PCK for each experiment
results = []
for exp_dir in sorted(pointodyssey_experiments):
    exp_name = exp_dir.name
    csv_path = exp_dir / "validation_results.csv"
    
    if not csv_path.exists():
        print(f"Warning: {exp_name} - no validation_results.csv found")
        results.append({
            'name': exp_name,
            'pointodyssey_pck': None,
            'spair_pck': None
        })
        continue
    
    # Parse CSV to find max PCK for pointodyssey and spair
    max_pck_pointodyssey = None
    max_pck_spair = None
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                benchmark = row.get('benchmark', '').strip()
                pck_str = row.get('pck', '').strip()
                
                if not pck_str:
                    continue
                
                try:
                    pck = float(pck_str)
                    
                    if benchmark == 'pointodyssey':
                        if max_pck_pointodyssey is None or pck > max_pck_pointodyssey:
                            max_pck_pointodyssey = pck
                    elif benchmark == 'spair':
                        if max_pck_spair is None or pck > max_pck_spair:
                            max_pck_spair = pck
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        max_pck_pointodyssey = None
        max_pck_spair = None
    
    results.append({
        'name': exp_name,
        'pointodyssey_pck': max_pck_pointodyssey,
        'spair_pck': max_pck_spair
    })

# Print table sorted by experiment name
print("=" * 120)
print(f"{'Experiment Name':<80} {'PointOdyssey PCK':<20} {'SPair PCK':<20}")
print("=" * 120)

for result in results:
    name = result['name']
    ptod_pck = f"{result['pointodyssey_pck']:.2f}%" if result['pointodyssey_pck'] is not None else "N/A"
    spair_pck = f"{result['spair_pck']:.2f}%" if result['spair_pck'] is not None else "N/A"
    
    print(f"{name:<80} {ptod_pck:<20} {spair_pck:<20}")

print("=" * 120)

# Print table sorted by PointOdyssey PCK (descending)
print("\n" + "=" * 120)
print("SORTED BY POINTODYSSEY PCK (DESCENDING)")
print("=" * 120)
print(f"{'Experiment Name':<80} {'PointOdyssey PCK':<20} {'SPair PCK':<20}")
print("=" * 120)

# Sort by PointOdyssey PCK (descending), handling None values
sorted_results = sorted(
    results, 
    key=lambda x: x['pointodyssey_pck'] if x['pointodyssey_pck'] is not None else -1,
    reverse=True
)

for result in sorted_results:
    name = result['name']
    ptod_pck = f"{result['pointodyssey_pck']:.2f}%" if result['pointodyssey_pck'] is not None else "N/A"
    spair_pck = f"{result['spair_pck']:.2f}%" if result['spair_pck'] is not None else "N/A"
    
    print(f"{name:<80} {ptod_pck:<20} {spair_pck:<20}")

print("=" * 120)

