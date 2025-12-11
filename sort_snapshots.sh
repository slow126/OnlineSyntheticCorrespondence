#!/bin/bash

# Script to sort snapshots by pretrained and freeze configurations

# Base directories
SNAPSHOTS_DIR="snapshots"
OUTPUT_BASE="sorted_snapshots"

# Create output directories for each configuration
mkdir -p "${OUTPUT_BASE}/pretrainedFalse_freezeFalse"
mkdir -p "${OUTPUT_BASE}/pretrainedFalse_freezeTrue"
mkdir -p "${OUTPUT_BASE}/pretrainedTrue_freezeFalse"
mkdir -p "${OUTPUT_BASE}/pretrainedTrue_freezeTrue"

echo "Starting to sort snapshots..."
echo "================================"

# Counter for tracking
count_ff=0
count_ft=0
count_tf=0
count_tt=0

# Loop through all directories in snapshots
for snapshot_dir in ${SNAPSHOTS_DIR}/*; do
    if [ -d "$snapshot_dir" ]; then
        snapshot_name=$(basename "$snapshot_dir")
        
        # Determine the configuration
        if [[ $snapshot_name == *"pretrainedFalse"*"freezeFalse"* ]]; then
            target_dir="${OUTPUT_BASE}/pretrainedFalse_freezeFalse"
            ((count_ff++))
        elif [[ $snapshot_name == *"pretrainedFalse"*"freezeTrue"* ]]; then
            target_dir="${OUTPUT_BASE}/pretrainedFalse_freezeTrue"
            ((count_ft++))
        elif [[ $snapshot_name == *"pretrainedTrue"*"freezeFalse"* ]]; then
            target_dir="${OUTPUT_BASE}/pretrainedTrue_freezeFalse"
            ((count_tf++))
        elif [[ $snapshot_name == *"pretrainedTrue"*"freezeTrue"* ]]; then
            target_dir="${OUTPUT_BASE}/pretrainedTrue_freezeTrue"
            ((count_tt++))
        else
            echo "Warning: Could not categorize $snapshot_name"
            continue
        fi
        
        # Copy the snapshot directory
        echo "Copying $snapshot_name -> $target_dir/"
        cp -r "$snapshot_dir" "$target_dir/"
    fi
done

echo "================================"
echo "Sorting complete!"
echo ""
echo "Summary:"
echo "  pretrainedFalse_freezeFalse: $count_ff snapshots"
echo "  pretrainedFalse_freezeTrue:  $count_ft snapshots"
echo "  pretrainedTrue_freezeFalse:  $count_tf snapshots"
echo "  pretrainedTrue_freezeTrue:   $count_tt snapshots"
echo ""
echo "Total: $((count_ff + count_ft + count_tf + count_tt)) snapshots sorted"
echo ""
echo "Output directory: ${OUTPUT_BASE}/"
