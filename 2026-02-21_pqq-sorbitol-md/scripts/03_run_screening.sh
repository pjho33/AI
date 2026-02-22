#!/bin/bash
#
# Batch MD screening launcher
# Runs multiple poses in parallel or sequential
#

set -e

SCREENING_DIR="../md_runs/screening"
N_PARALLEL=4  # Number of parallel jobs (adjust based on GPU availability)

echo "============================================================"
echo "Batch MD Screening Launcher"
echo "============================================================"

if [ ! -d "$SCREENING_DIR" ]; then
    echo "Error: Screening directory not found: $SCREENING_DIR"
    echo "Run 02_insert_poses.py first"
    exit 1
fi

# Count available poses
N_POSES=$(find $SCREENING_DIR -maxdepth 1 -type d -name "pose_*" | wc -l)

echo ""
echo "Found $N_POSES pose directories"
echo "Parallel jobs: $N_PARALLEL"
echo ""

# Function to run MD for one pose
run_pose() {
    POSE_DIR=$1
    POSE_NAME=$(basename $POSE_DIR)
    
    echo "[$POSE_NAME] Starting MD screening..."
    
    cd $POSE_DIR
    
    # Check if already completed
    if [ -f "prod/prod.gro" ]; then
        echo "[$POSE_NAME] Already completed, skipping"
        cd - > /dev/null
        return 0
    fi
    
    # Run MD script
    if [ -f "run_md.sh" ]; then
        bash run_md.sh > md_screening.log 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[$POSE_NAME] ✓ Complete"
        else
            echo "[$POSE_NAME] ✗ Failed (check md_screening.log)"
        fi
    else
        echo "[$POSE_NAME] ✗ No run_md.sh found"
    fi
    
    cd - > /dev/null
}

export -f run_pose

# Get list of pose directories
POSE_DIRS=$(find $SCREENING_DIR -maxdepth 1 -type d -name "pose_*" | sort)

# Run in parallel using GNU parallel if available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for $N_PARALLEL concurrent jobs"
    echo ""
    echo "$POSE_DIRS" | parallel -j $N_PARALLEL run_pose {}
else
    echo "GNU parallel not found, running sequentially"
    echo ""
    for POSE_DIR in $POSE_DIRS; do
        run_pose $POSE_DIR
    done
fi

echo ""
echo "============================================================"
echo "Screening Complete"
echo "============================================================"
echo ""
echo "Next: Analyze trajectories to select top poses"
echo "  python scripts/05_rank_poses.py --screening_dir $SCREENING_DIR"
