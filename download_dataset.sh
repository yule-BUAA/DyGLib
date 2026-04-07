#!/bin/bash

# Download and Setup Script for DyGLib Datasets
# Downloads datasets from Zenodo and prepares them for preprocessing

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== DyGLib Dataset Download and Setup ===${NC}\n"

# Dataset URLs from Zenodo
declare -A DATASET_URLS=(
    ["wikipedia"]="https://zenodo.org/records/7213796/files/wikipedia.zip?download=1"
    ["reddit"]="https://zenodo.org/records/7213796/files/reddit.zip?download=1"
    ["mooc"]="https://zenodo.org/records/7213796/files/mooc.zip?download=1"
    ["lastfm"]="https://zenodo.org/records/7213796/files/lastfm.zip?download=1"
    ["enron"]="https://zenodo.org/records/7213796/files/enron.zip?download=1"
    ["uci"]="https://zenodo.org/records/7213796/files/uci.zip?download=1"
    ["Flights"]="https://zenodo.org/records/7213796/files/Flights.zip?download=1"
    ["CanParl"]="https://zenodo.org/records/7213796/files/CanParl.zip?download=1"
    ["USLegis"]="https://zenodo.org/records/7213796/files/USLegis.zip?download=1"
    ["UNtrade"]="https://zenodo.org/records/7213796/files/UNtrade.zip?download=1"
    ["UNvote"]="https://zenodo.org/records/7213796/files/UNvote.zip?download=1"
    ["Contacts"]="https://zenodo.org/records/7213796/files/Contacts.zip?download=1"
    ["SocialEvo"]="https://zenodo.org/records/7213796/files/SocialEvo.zip?download=1"
)

# Function to download and setup a dataset
download_dataset() {
    local dataset_name=$1
    local url=${DATASET_URLS[$dataset_name]}

    if [ -z "$url" ]; then
        echo -e "${RED}Error: Unknown dataset '$dataset_name'${NC}"
        echo "Available datasets: ${!DATASET_URLS[@]}"
        return 1
    fi

    echo -e "${GREEN}Downloading $dataset_name...${NC}"

    # Create dataset folder
    mkdir -p "DG_data/${dataset_name}"

    # Download the zip file
    wget -q --show-progress "$url" -O "DG_data/${dataset_name}/${dataset_name}.zip"

    # Unzip
    echo -e "${GREEN}Unzipping $dataset_name...${NC}"
    unzip -q "DG_data/${dataset_name}/${dataset_name}.zip" -d "DG_data/${dataset_name}/"

    # Remove zip file
    rm "DG_data/${dataset_name}/${dataset_name}.zip"

    # Check if CSV exists
    if [ -f "DG_data/${dataset_name}/${dataset_name}.csv" ]; then
        echo -e "${GREEN}✓ ${dataset_name}.csv downloaded successfully${NC}"
    else
        echo -e "${RED}✗ Warning: ${dataset_name}.csv not found after extraction${NC}"
    fi

    echo ""
}

# Function to preprocess a dataset
preprocess_dataset() {
    local dataset_name=$1

    echo -e "${GREEN}Preprocessing $dataset_name...${NC}"
    cd preprocess_data

    # Determine node feature dimension (172 is default for most datasets)
    local node_feat_dim=172

    python preprocess_data.py --dataset_name "$dataset_name" --node_feat_dim $node_feat_dim

    cd ..
    echo -e "${GREEN}✓ $dataset_name preprocessed successfully${NC}\n"
}

# Main logic
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_name> [--preprocess]"
    echo ""
    echo "Available datasets:"
    for dataset in "${!DATASET_URLS[@]}"; do
        echo "  - $dataset"
    done
    echo ""
    echo "Examples:"
    echo "  $0 wikipedia              # Download only"
    echo "  $0 wikipedia --preprocess # Download and preprocess"
    echo "  $0 reddit --preprocess"
    echo ""
    exit 1
fi

DATASET=$1
PREPROCESS=false

# Check for --preprocess flag
if [ "$2" == "--preprocess" ]; then
    PREPROCESS=true
fi

# Check if dataset already exists
if [ -f "DG_data/${DATASET}/${DATASET}.csv" ]; then
    echo -e "${BLUE}Dataset $DATASET already exists in DG_data/${DATASET}/${NC}"
    read -p "Do you want to re-download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        if [ "$PREPROCESS" = true ]; then
            preprocess_dataset "$DATASET"
        fi
        exit 0
    fi
fi

# Download the dataset
download_dataset "$DATASET"

# Preprocess if requested
if [ "$PREPROCESS" = true ]; then
    preprocess_dataset "$DATASET"
    echo -e "${GREEN}=== Setup Complete! ===${NC}"
    echo -e "Dataset ready at: ${BLUE}processed_data/${DATASET}/${NC}"
else
    echo -e "${GREEN}=== Download Complete! ===${NC}"
    echo -e "Raw dataset at: ${BLUE}DG_data/${DATASET}/${DATASET}.csv${NC}"
    echo -e "To preprocess, run: ${BLUE}cd preprocess_data && python preprocess_data.py --dataset_name ${DATASET}${NC}"
fi
