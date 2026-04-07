# Quick Start Guide for DyGLib with Train Modes & MRR

This guide shows you how to download datasets, preprocess them, and run experiments with the three training modes (full, random_10, compressed_10) reporting AUC, AP, and MRR metrics.

## 📋 Quick Steps

### Step 1: Download and Preprocess a Dataset

**Option A: Using Python script (recommended)**
```bash
# Download Wikipedia dataset and preprocess it
python download_dataset.py wikipedia --preprocess

# Or download Reddit
python download_dataset.py reddit --preprocess
```

**Option B: Using bash script**
```bash
chmod +x download_dataset.sh
./download_dataset.sh wikipedia --preprocess
```

**Option C: Manual download (Kaggle-style)**
```python
import os
import zipfile
import urllib.request

# Create folder
os.makedirs("DG_data/wikipedia", exist_ok=True)

# Download
url = "https://zenodo.org/records/7213796/files/wikipedia.zip?download=1"
urllib.request.urlretrieve(url, "DG_data/wikipedia/wikipedia.zip")

# Unzip
with zipfile.ZipFile("DG_data/wikipedia/wikipedia.zip", 'r') as zip_ref:
    zip_ref.extractall("DG_data/wikipedia/")

# Clean up
os.remove("DG_data/wikipedia/wikipedia.zip")
```

Then preprocess:
```bash
cd preprocess_data
python preprocess_data.py --dataset_name wikipedia
cd ..
```

### Step 2: Run Training with Different Modes

**Full Training Set (70% of data)**
```bash
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode full \
    --batch_size 200 \
    --num_neighbors 10 \
    --num_runs 5 \
    --gpu 0
```

**Random 10% Sample**
```bash
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode random_10 \
    --batch_size 200 \
    --num_neighbors 10 \
    --num_runs 5 \
    --gpu 0
```

**Compressed 10% (Binary Search Window)**
```bash
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode compressed_10 \
    --batch_size 200 \
    --num_neighbors 10 \
    --num_runs 5 \
    --gpu 0
```

### Step 3: Evaluate Models

```bash
# Evaluate with random negative sampling (matches your PyG TGN setup)
python evaluate_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode compressed_10 \
    --negative_sample_strategy random \
    --batch_size 200 \
    --num_neighbors 10 \
    --num_runs 5 \
    --gpu 0
```

## 🧪 Quick Test (No Download Needed)

Test with the existing Myket dataset:
```bash
chmod +x quick_test_myket.sh
./quick_test_myket.sh
```

## 📊 Expected Output

You should see metrics for each run:
```
validate average_precision, 0.9524 ± 0.0032
validate roc_auc, 0.9456 ± 0.0041
validate mrr, 0.8234 ± 0.0067

test average_precision, 0.9480 ± 0.0045
test roc_auc, 0.9420 ± 0.0038
test mrr, 0.8156 ± 0.0052
```

## 📦 Available Datasets

From Zenodo (use download scripts):
- `wikipedia` - Wikipedia edit network
- `reddit` - Reddit post network
- `mooc` - MOOC user-course network
- `lastfm` - LastFM user-artist network
- `enron` - Enron email network
- `uci` - UCI message network
- `Flights` - Flight routes
- `CanParl` - Canadian Parliament
- `USLegis` - US Legislature
- `UNtrade` - UN Trade
- `UNvote` - UN Vote
- `Contacts` - Contact network
- `SocialEvo` - Social Evolution

Already included:
- `myket` - Myket Android app market

## 🔧 Training Modes Explained

1. **`--train_mode full`** (default)
   - Uses complete 70% training split
   - Standard baseline

2. **`--train_mode random_10`**
   - Random 10% sample of training data
   - Maintains temporal order
   - Fast baseline

3. **`--train_mode compressed_10`**
   - Binary search window compression
   - Per-(src,dst)-pair intelligent compression
   - Preserves temporal patterns

All modes evaluate on **full validation and test sets** (inductive setting).

## 📝 Run All Three Modes at Once

```bash
for mode in full random_10 compressed_10; do
    echo "Training with $mode mode..."
    python train_link_prediction.py \
        --dataset_name wikipedia \
        --model_name TGN \
        --train_mode $mode \
        --batch_size 200 \
        --num_neighbors 10 \
        --num_runs 5 \
        --gpu 0
done
```

## 🎯 Matching PyG TGN Setup

Your PyTorch Geometric TGN code matches these DyGLib settings:
- Split: `val_ratio=0.15, test_ratio=0.15` ✅ (DyGLib default)
- Batch size: `200` ✅
- Neighbors: `size=10` ✅ (`--num_neighbors 10`)
- Neg sampling: `neg_sampling_ratio=1.0` ✅ (`--negative_sample_strategy random`)
- Memory updates: ✅ (automatic in DyGLib for TGN)

## 💡 Tips

1. **Use best configs**: Add `--load_best_configs` to use paper-reported hyperparameters
2. **Reduce epochs for testing**: Add `--num_epochs 5` for quick tests
3. **Single run**: Use `--num_runs 1` for faster testing
4. **Different models**: Try `--model_name DyGFormer` or `--model_name TGAT`

## 📄 More Information

- Full documentation: `TRAIN_MODES_README.md`
- Dataset format: `DG_data/DATASETS_README.md`
- Original paper: https://arxiv.org/abs/2303.13047

## ✅ You're Ready!

Everything is set up. Just:
1. Download a dataset: `python download_dataset.py wikipedia --preprocess`
2. Run training: `python train_link_prediction.py --dataset_name wikipedia --model_name TGN --train_mode compressed_10`
3. Check results: Look for `average_precision`, `roc_auc`, and `mrr` metrics!
