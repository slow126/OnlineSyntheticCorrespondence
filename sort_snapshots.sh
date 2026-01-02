#!/bin/bash

# Script to sort snapshots by pretrained and freeze configurations, then by dataset

# Base directories
SNAPSHOTS_DIR="snapshots_mixed"
OUTPUT_BASE="sorted_snapshots"

# Function to extract dataset mix info from snapshot name
extract_dataset_info() {
    local name=$1
    local dataset_info=""
    
    # Check for mixed dataset pattern: dataset1_dataset2 or dataset1_dataset2_X_Y
    # Pattern: ends with _X_Y where X and Y are numbers (percentages)
    if [[ $name =~ _([0-9]+)_([0-9]+)$ ]]; then
        # Has explicit percentages: e.g., spair_synthetic_70_30
        dataset_info="${name%_${BASH_REMATCH[1]}_${BASH_REMATCH[2]}}"
        mix_percent="${BASH_REMATCH[1]}_${BASH_REMATCH[2]}"
        dataset_info="${dataset_info}_${mix_percent}"
    elif [[ $name =~ ^([a-zA-Z]+_[a-zA-Z]+)(_|$) ]]; then
        # Mixed dataset without explicit percentages (assumed 50/50): e.g., spair_synthetic
        dataset_info="${BASH_REMATCH[1]}"
    else
        # Single dataset: extract first part before any config flags
        # Remove common config suffixes to get base dataset name
        dataset_info=$(echo "$name" | sed -E 's/_(pretrained|freeze)[A-Za-z]*.*$//' | sed -E 's/_.*$//')
    fi
    
    echo "$dataset_info"
}

echo "Starting to sort snapshots..."
echo "================================"

# Counters for tracking
declare -A counts
declare -A dataset_counts

# Initialize counters
counts[ff]=0
counts[ft]=0
counts[tf]=0
counts[tt]=0
counts[pretrained_false]=0
counts[pretrained_true]=0

# Loop through all directories in snapshots
for snapshot_dir in ${SNAPSHOTS_DIR}/*; do
    if [ -d "$snapshot_dir" ]; then
        snapshot_name=$(basename "$snapshot_dir")
        
        # Extract dataset information
        dataset_info=$(extract_dataset_info "$snapshot_name")
        
        # Determine pretrained and freeze configuration
        if [[ $snapshot_name == *"pretrainedFalse"*"freezeFalse"* ]]; then
            pretrained="False"
            freeze="False"
            config_key="ff"
            ((counts[ff]++))
            ((counts[pretrained_false]++))
        elif [[ $snapshot_name == *"pretrainedFalse"*"freezeTrue"* ]]; then
            pretrained="False"
            freeze="True"
            config_key="ft"
            ((counts[ft]++))
            ((counts[pretrained_false]++))
        elif [[ $snapshot_name == *"pretrainedTrue"*"freezeFalse"* ]]; then
            pretrained="True"
            freeze="False"
            config_key="tf"
            ((counts[tf]++))
            ((counts[pretrained_true]++))
        elif [[ $snapshot_name == *"pretrainedTrue"*"freezeTrue"* ]]; then
            pretrained="True"
            freeze="True"
            config_key="tt"
            ((counts[tt]++))
            ((counts[pretrained_true]++))
        else
            echo "Warning: Could not categorize pretrained/freeze for $snapshot_name"
            continue
        fi
        
        # Create directory structure: pretrained_freeze/dataset/
        # Primary organization by pretrained/freeze, then by dataset
        target_dir="${OUTPUT_BASE}/pretrained${pretrained}_freeze${freeze}/${dataset_info}"
        pretrained_dir="${OUTPUT_BASE}/pretrained${pretrained}/${dataset_info}"
        
        # Create directories
        mkdir -p "$target_dir"
        mkdir -p "$pretrained_dir"
        
        # Track dataset counts per configuration
        config_dataset_key="${config_key}_${dataset_info}"
        if [ -z "${dataset_counts[$config_dataset_key]}" ]; then
            dataset_counts[$config_dataset_key]=0
        fi
        ((dataset_counts[$config_dataset_key]++))
        
        # Copy the snapshot directory to both the specific config folder and the pretrained-only folder
        echo "Copying $snapshot_name -> $target_dir/"
        cp -r "$snapshot_dir" "$target_dir/"
        echo "Copying $snapshot_name -> $pretrained_dir/"
        cp -r "$snapshot_dir" "$pretrained_dir/"
    fi
done

echo "================================"
echo "Sorting complete!"
echo ""
echo "Summary (by pretrained and freeze):"
echo "  pretrainedFalse_freezeFalse: ${counts[ff]} snapshots"
echo "  pretrainedFalse_freezeTrue:  ${counts[ft]} snapshots"
echo "  pretrainedTrue_freezeFalse:  ${counts[tf]} snapshots"
echo "  pretrainedTrue_freezeTrue:   ${counts[tt]} snapshots"
echo ""
echo "Summary (by pretrained only, combines all freeze models):"
echo "  pretrainedFalse: ${counts[pretrained_false]} snapshots"
echo "  pretrainedTrue:  ${counts[pretrained_true]} snapshots"
echo ""
echo "Summary (by dataset within each configuration):"
for key in "${!dataset_counts[@]}"; do
    echo "  $key: ${dataset_counts[$key]} snapshots"
done
echo ""
echo "Total: $((${counts[ff]} + ${counts[ft]} + ${counts[tf]} + ${counts[tt]})) snapshots sorted"
echo ""
echo "Output directory: ${OUTPUT_BASE}/"
echo "  Structure: ${OUTPUT_BASE}/pretrained<True|False>_freeze<True|False>/<dataset>/"
echo "            ${OUTPUT_BASE}/pretrained<True|False>/<dataset>/"
