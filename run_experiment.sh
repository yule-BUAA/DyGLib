#!/bin/bash

# Complete experiment script for any dataset and model
# Usage: ./run_experiment.sh <dataset_name> <model_name> [options]
#
# Example:
#   ./run_experiment.sh wikipedia TGN
#   ./run_experiment.sh reddit DyGFormer --quick
#   ./run_experiment.sh mooc TGAT --epochs 30 --runs 3

set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default parameters
EPOCHS=50
RUNS=5
BATCH_SIZE=200
NUM_NEIGHBORS=10
GPU=0
QUICK_MODE=false
SKIP_DOWNLOAD=false
MODES="full random_10 compressed_10"

# Function to print usage
usage() {
    echo "Usage: $0 <dataset_name> <model_name> [options]"
    echo ""
    echo "Required arguments:"
    echo "  dataset_name    Dataset to use (wikipedia, reddit, mooc, lastfm, etc.)"
    echo "  model_name      Model to train (TGN, DyGFormer, TGAT, etc.)"
    echo ""
    echo "Options:"
    echo "  --epochs N      Number of epochs (default: 50)"
    echo "  --runs N        Number of runs (default: 5)"
    echo "  --batch N       Batch size (default: 200)"
    echo "  --neighbors N   Number of neighbors (default: 10)"
    echo "  --gpu N         GPU device (default: 0)"
    echo "  --quick         Quick mode: 5 epochs, 1 run"
    echo "  --skip-download Skip dataset download"
    echo "  --modes \"...\"   Train modes to run (default: \"full random_10 compressed_10\")"
    echo "  --best-configs  Use best configurations from paper"
    echo ""
    echo "Examples:"
    echo "  $0 wikipedia TGN"
    echo "  $0 reddit DyGFormer --quick"
    echo "  $0 mooc TGAT --epochs 30 --runs 3 --gpu 1"
    echo "  $0 wikipedia TGN --modes \"full compressed_10\" --best-configs"
    echo ""
    exit 1
}

# Check if at least 2 arguments provided
if [ $# -lt 2 ]; then
    usage
fi

DATASET=$1
MODEL=$2
shift 2

# Additional arguments
EXTRA_ARGS=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --neighbors)
            NUM_NEIGHBORS="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            EPOCHS=5
            RUNS=1
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --best-configs)
            EXTRA_ARGS="$EXTRA_ARGS --load_best_configs"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Print configuration
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}DyGLib Complete Experiment${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Dataset:       $DATASET"
echo "  Model:         $MODEL"
echo "  Train modes:   $MODES"
echo "  Epochs:        $EPOCHS"
echo "  Runs:          $RUNS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Neighbors:     $NUM_NEIGHBORS"
echo "  GPU:           $GPU"
echo "  Extra args:    $EXTRA_ARGS"
echo ""

# Step 1: Download and preprocess dataset
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo -e "${BLUE}Step 1: Checking dataset...${NC}"

    if [ ! -f "processed_data/$DATASET/ml_$DATASET.csv" ]; then
        echo -e "${YELLOW}Dataset not found. Downloading and preprocessing...${NC}"
        python download_dataset.py "$DATASET" --preprocess
    else
        echo -e "${GREEN}✓ Dataset already processed${NC}"
    fi
else
    echo -e "${YELLOW}Skipping download (--skip-download)${NC}"
fi

echo ""

# Step 2: Train with each mode
STEP_NUM=2
for mode in $MODES; do
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}Step $STEP_NUM: Training with $mode mode${NC}"
    echo -e "${BLUE}======================================${NC}"

    python train_link_prediction.py \
        --dataset_name "$DATASET" \
        --model_name "$MODEL" \
        --train_mode "$mode" \
        --batch_size "$BATCH_SIZE" \
        --num_neighbors "$NUM_NEIGHBORS" \
        --num_epochs "$EPOCHS" \
        --num_runs "$RUNS" \
        --gpu "$GPU" \
        $EXTRA_ARGS

    echo ""
    ((STEP_NUM++))
done

# Results summary
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}✅ All training completed!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${BLUE}Results saved to:${NC}"
echo "  Logs:          ./logs/$MODEL/$DATASET/"
echo "  Models:        ./saved_models/$MODEL/$DATASET/"
echo "  Results:       ./saved_results/$MODEL/$DATASET/"
echo ""

# Quick results display
if [ -d "logs/$MODEL/$DATASET" ]; then
    echo -e "${BLUE}Latest test results:${NC}"
    for mode in $MODES; do
        echo ""
        echo -e "${YELLOW}$mode mode:${NC}"
        LOG_FILE=$(ls -t logs/$MODEL/$DATASET/*/$(date +%Y%m%d)*.log 2>/dev/null | head -1)
        if [ -f "$LOG_FILE" ]; then
            grep "average test" "$LOG_FILE" | tail -3 || echo "No results found"
        else
            # Try finding any recent log
            LOG_FILE=$(ls -t logs/$MODEL/$DATASET/*/*.log 2>/dev/null | head -1)
            if [ -f "$LOG_FILE" ]; then
                grep "average test" "$LOG_FILE" | tail -3 || echo "No results found"
            fi
        fi
    done
fi

echo ""
echo -e "${GREEN}Experiment complete! 🎉${NC}"
