# Training Data Modes and MRR Metric

This document describes the new training data modes and MRR metric integration added to DyGLib.

## Overview

DyGLib now supports three training data modes:
1. **full** (default): Use the complete 70% training split
2. **random_10**: Use a random 10% sample of the training split
3. **compressed_10**: Use a binary search window compressed 10% of the training split

All modes train on their respective training sets and evaluate on the **full validation and test sets** (inductive setting).

## New Metrics

### Mean Reciprocal Rank (MRR)
- Measures the ranking quality of positive edges against negative edges
- Uses optimistic and pessimistic ranking for tie-breaking
- Automatically computed and displayed alongside AUC and AP metrics

## Usage

### Training

```bash
# Full training set (default)
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --num_runs 5 --gpu 0

# Random 10% sample of training set
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --train_mode random_10 --num_runs 5 --gpu 0

# Compressed 10% of training set (binary search window compression)
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --train_mode compressed_10 --num_runs 5 --gpu 0
```

### Evaluation

```bash
# Full training set (default)
python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --negative_sample_strategy random --num_runs 5 --gpu 0

# Random 10% sample
python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --train_mode random_10 --negative_sample_strategy random --num_runs 5 --gpu 0

# Compressed 10%
python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --train_mode compressed_10 --negative_sample_strategy random --num_runs 5 --gpu 0
```

### With Best Configurations

```bash
# Training with best configs and compressed 10%
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --train_mode compressed_10 --load_best_configs --num_runs 5 --gpu 0

# Evaluation with best configs and compressed 10%
python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --train_mode compressed_10 --negative_sample_strategy random --load_best_configs --num_runs 5 --gpu 0
```

## Compression Details

### Random Sampling (`random_10`)
- Uniformly samples 10% of training interactions
- Maintains temporal order
- Fast and simple
- Seed: 42 (reproducible)

### Binary Search Window Compression (`compressed_10`)
- Per-(src, dst)-pair compression using binary search
- Finds optimal time window size to achieve ~10% compression (±2% tolerance)
- Events within windows are aggregated (last event timestamp per window)
- Preserves temporal patterns and burst structures
- Seed: 42 (reproducible)
- Parameters:
  - `ratio`: 0.1 (target 10%)
  - `tol`: 0.02 (±2% tolerance)
  - `w_lo`: 1.0 (minimum window size)
  - `w_hi`: 1e7 (maximum window size)
  - `max_iter`: 30 (max binary search iterations)

## Output Metrics

All evaluation scripts now report three metrics:
1. **average_precision** (AP): Area under the precision-recall curve
2. **roc_auc** (AUC): Area under the ROC curve
3. **mrr** (MRR): Mean Reciprocal Rank

Example output:
```
average test average_precision, 0.9524 ± 0.0032
average test roc_auc, 0.9456 ± 0.0041
average test mrr, 0.8234 ± 0.0067
```

## Implementation Files

- `utils/metrics.py`: Added `compute_mrr()` and updated `get_link_prediction_metrics()`
- `utils/compression.py`: New file with `random_compress_data()` and `binary_search_window_compress_data()`
- `utils/DataLoader.py`: Modified `get_link_prediction_data()` to support `train_mode` parameter
- `utils/load_configs.py`: Added `--train_mode` argument
- `train_link_prediction.py`: Updated to pass `train_mode` to data loader
- `evaluate_link_prediction.py`: Updated to pass `train_mode` to data loader

## Notes

- All modes use the same validation and test sets (full, not compressed)
- Compression is applied only to the training split
- The inductive setting is maintained: new nodes in test set are handled the same way
- Random seed is fixed (42) for reproducibility across runs
- MRR calculation uses separate positive/negative scores for ranking
