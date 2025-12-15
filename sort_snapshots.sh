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
# Create directories for pretrained-only sorting (combines all freeze models)
mkdir -p "${OUTPUT_BASE}/pretrainedFalse"
mkdir -p "${OUTPUT_BASE}/pretrainedTrue"

echo "Starting to sort snapshots..."
echo "================================"

# Counter for tracking
count_ff=0
count_ft=0
count_tf=0
count_tt=0
count_pretrained_false=0
count_pretrained_true=0

# Loop through all directories in snapshots
for snapshot_dir in ${SNAPSHOTS_DIR}/*; do
    if [ -d "$snapshot_dir" ]; then
        snapshot_name=$(basename "$snapshot_dir")
        
        # Determine the configuration
        if [[ $snapshot_name == *"pretrainedFalse"*"freezeFalse"* ]]; then
            target_dir="${OUTPUT_BASE}/pretrainedFalse_freezeFalse"
            pretrained_dir="${OUTPUT_BASE}/pretrainedFalse"
            ((count_ff++))
            ((count_pretrained_false++))
        elif [[ $snapshot_name == *"pretrainedFalse"*"freezeTrue"* ]]; then
            target_dir="${OUTPUT_BASE}/pretrainedFalse_freezeTrue"
            pretrained_dir="${OUTPUT_BASE}/pretrainedFalse"
            ((count_ft++))
            ((count_pretrained_false++))
        elif [[ $snapshot_name == *"pretrainedTrue"*"freezeFalse"* ]]; then
            target_dir="${OUTPUT_BASE}/pretrainedTrue_freezeFalse"
            pretrained_dir="${OUTPUT_BASE}/pretrainedTrue"
            ((count_tf++))
            ((count_pretrained_true++))
        elif [[ $snapshot_name == *"pretrainedTrue"*"freezeTrue"* ]]; then
            target_dir="${OUTPUT_BASE}/pretrainedTrue_freezeTrue"
            pretrained_dir="${OUTPUT_BASE}/pretrainedTrue"
            ((count_tt++))
            ((count_pretrained_true++))
        else
            echo "Warning: Could not categorize $snapshot_name"
            continue
        fi
        
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
echo "  pretrainedFalse_freezeFalse: $count_ff snapshots"
echo "  pretrainedFalse_freezeTrue:  $count_ft snapshots"
echo "  pretrainedTrue_freezeFalse:  $count_tf snapshots"
echo "  pretrainedTrue_freezeTrue:   $count_tt snapshots"
echo ""
echo "Summary (by pretrained only, combines all freeze models):"
echo "  pretrainedFalse: $count_pretrained_false snapshots"
echo "  pretrainedTrue:  $count_pretrained_true snapshots"
echo ""
echo "Total: $((count_ff + count_ft + count_tf + count_tt)) snapshots sorted"
echo ""
echo "Output directory: ${OUTPUT_BASE}/"
