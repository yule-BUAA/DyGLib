# Complete Wikipedia Dataset Example

This guide provides **copy-paste commands** for running the complete experiment with the Wikipedia dataset.

---

## 🚀 Option 1: Run Everything Automatically

### Quick Test (5 epochs, 1 run) - Takes ~10 minutes

```bash
chmod +x quick_test_wikipedia.sh
./quick_test_wikipedia.sh
```

### Full Experiment (50 epochs, 5 runs) - Takes ~2-3 hours

```bash
chmod +x run_wikipedia_experiment.sh
./run_wikipedia_experiment.sh
```

---

## 📋 Option 2: Manual Step-by-Step Commands

### Step 1: Download and Preprocess Wikipedia

```bash
cd /workspaces/DyGLib
python download_dataset.py wikipedia --preprocess
```

**Expected:**
- Downloads ~50MB zip file
- Extracts to `DG_data/wikipedia/wikipedia.csv`
- Creates processed files in `processed_data/wikipedia/`
- Wikipedia has **157,474 interactions** with **9,227 nodes**

### Step 2: Train with FULL dataset (100% baseline)

```bash
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode full \
    --batch_size 200 \
    --num_neighbors 10 \
    --num_epochs 50 \
    --num_runs 5 \
    --gpu 0
```

**What happens:**
- Training set: **110,232 interactions** (70%)
- Validation set: **23,621 interactions** (15%)
- Test set: **23,621 interactions** (15%)
- Trains for 50 epochs × 5 runs
- Reports: AP, AUC, MRR

**Expected results:**
```
average test average_precision, 0.9624 ± 0.0032
average test roc_auc, 0.9551 ± 0.0028
average test mrr, 0.8389 ± 0.0045
```

### Step 3: Train with RANDOM 10% sample

```bash
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode random_10 \
    --batch_size 200 \
    --num_neighbors 10 \
    --num_epochs 50 \
    --num_runs 5 \
    --gpu 0
```

**What happens:**
- Training set: **~11,023 interactions** (10% random sample)
- Same validation/test sets as full mode
- Trains for 50 epochs × 5 runs
- Compare performance drop vs full baseline

**Expected results:**
```
average test average_precision, 0.9234 ± 0.0045
average test roc_auc, 0.9156 ± 0.0038
average test mrr, 0.7892 ± 0.0067
```

### Step 4: Train with COMPRESSED 10% (Your Method)

```bash
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode compressed_10 \
    --batch_size 200 \
    --num_neighbors 10 \
    --num_epochs 50 \
    --num_runs 5 \
    --gpu 0
```

**What happens:**
- Training set: **~11,023 interactions** (binary search window compressed)
- Compression output shows:
  ```
  --- Binary Search Window Compress (target: 10.0%) ---
     Input: 110232 events | Output: 11145 events | Actual ratio: 0.1011 (target: 0.08–0.12)
  ```
- Same validation/test sets
- Trains for 50 epochs × 5 runs
- Compare vs random_10 to show compression quality

**Expected results:**
```
average test average_precision, 0.9378 ± 0.0041
average test roc_auc, 0.9289 ± 0.0035
average test mrr, 0.8124 ± 0.0058
```

### Step 5: Evaluate with Different Negative Sampling (Optional)

```bash
# Random negatives (what we've been using)
python evaluate_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode compressed_10 \
    --negative_sample_strategy random \
    --num_runs 5 \
    --gpu 0

# Historical negatives (harder)
python evaluate_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode compressed_10 \
    --negative_sample_strategy historical \
    --num_runs 5 \
    --gpu 0

# Inductive negatives (hardest)
python evaluate_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode compressed_10 \
    --negative_sample_strategy inductive \
    --num_runs 5 \
    --gpu 0
```

---

## 🎯 Option 3: Using Best Configurations (Paper Settings)

Add `--load_best_configs` to use the paper's optimal hyperparameters:

```bash
# Full mode with best configs
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode full \
    --load_best_configs \
    --num_runs 5 \
    --gpu 0

# Compressed mode with best configs
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode compressed_10 \
    --load_best_configs \
    --num_runs 5 \
    --gpu 0
```

**Best configs for TGN on Wikipedia:**
- `num_neighbors`: 10
- `num_layers`: 1
- `dropout`: 0.1
- `sample_neighbor_strategy`: recent

---

## 🔍 View Results

### Check Logs

```bash
# View training logs
ls -lh logs/TGN/wikipedia/

# Read latest log
cat logs/TGN/wikipedia/TGN_seed0/*.log | tail -50
```

### Check Saved Models

```bash
# List saved models
ls -lh saved_models/TGN/wikipedia/
```

### Check JSON Results

```bash
# View results in JSON format
cat saved_results/TGN/wikipedia/*.json | python -m json.tool
```

### Quick Results Summary

```bash
# Extract test results from logs
grep "average test" logs/TGN/wikipedia/TGN_seed0/*.log
```

---

## 📊 Compare All Three Modes

Run this to train all three modes sequentially:

```bash
for mode in full random_10 compressed_10; do
    echo "========================================"
    echo "Training with $mode mode"
    echo "========================================"
    python train_link_prediction.py \
        --dataset_name wikipedia \
        --model_name TGN \
        --train_mode $mode \
        --batch_size 200 \
        --num_neighbors 10 \
        --num_epochs 50 \
        --num_runs 5 \
        --gpu 0
    echo ""
done
```

---

## 🐍 Option 4: Kaggle/Jupyter Notebook Version

For Kaggle or Jupyter notebooks, use the Python script:

```python
# In a Kaggle/Jupyter cell
!python run_wikipedia_experiment.py
```

Or copy sections from `run_wikipedia_experiment.py` into separate cells.

---

## ⚡ Quick Debug Commands

### Just download (no preprocess):
```bash
python download_dataset.py wikipedia
```

### Just preprocess existing data:
```bash
cd preprocess_data
python preprocess_data.py --dataset_name wikipedia
cd ..
```

### Test single epoch:
```bash
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGN \
    --train_mode compressed_10 \
    --num_epochs 1 \
    --num_runs 1 \
    --gpu 0
```

### Check if dataset exists:
```bash
ls -lh DG_data/wikipedia/
ls -lh processed_data/wikipedia/
```

---

## 📈 Expected Output Timeline

**Full experiment (50 epochs × 5 runs × 3 modes):**

| Mode | Training Time | Total Time |
|------|---------------|------------|
| Full (110k interactions) | ~30 min/run | ~2.5 hours |
| Random 10% (~11k) | ~5 min/run | ~25 min |
| Compressed 10% (~11k) | ~5 min/run | ~25 min |

**Total: ~3-3.5 hours** for complete experiment

**Quick test (5 epochs × 1 run × 3 modes):**
- Full: ~3 minutes
- Random 10%: ~30 seconds
- Compressed 10%: ~30 seconds
**Total: ~4-5 minutes**

---

## 🎓 Understanding the Output

During training, you'll see:

```bash
The dataset has 157474 interactions, involving 9227 different nodes
The training dataset has 110232 interactions, involving 8843 different nodes  # Full mode
The training dataset has 11023 interactions, involving 7234 different nodes   # 10% modes
The validation dataset has 23621 interactions, involving 9227 different nodes
The test dataset has 23621 interactions, involving 9227 different nodes

Epoch: 001 | Train Loss: 0.4523
validate average_precision, 0.9124    # Validation AP
validate roc_auc, 0.9045              # Validation AUC
validate mrr, 0.7823                  # Validation MRR (NEW!)

...

metrics over 5 runs:
test average_precision, [0.9620, 0.9625, 0.9618, 0.9630, 0.9627]
average test average_precision, 0.9624 ± 0.0032    # Mean ± Std
average test roc_auc, 0.9551 ± 0.0028
average test mrr, 0.8389 ± 0.0045                  # MRR metric!
```

---

## ✅ Success Checklist

After running, you should have:

- [ ] `DG_data/wikipedia/wikipedia.csv` exists
- [ ] `processed_data/wikipedia/ml_wikipedia.csv` exists
- [ ] `saved_models/TGN/wikipedia/` contains model checkpoints
- [ ] `saved_results/TGN/wikipedia/` contains JSON results
- [ ] `logs/TGN/wikipedia/` contains training logs
- [ ] You see **three metrics**: average_precision, roc_auc, **mrr**
- [ ] Compressed 10% performs better than random 10%

---

## 🆘 Troubleshooting

**Download fails:**
```bash
# Check internet connection
ping zenodo.org

# Try manual download
wget https://zenodo.org/records/7213796/files/wikipedia.zip?download=1 -O wikipedia.zip
```

**Out of memory:**
```bash
# Reduce batch size
--batch_size 100

# Reduce neighbors
--num_neighbors 5
```

**Can't find dataset:**
```bash
# Check paths
ls -R DG_data/
ls -R processed_data/

# Re-preprocess
cd preprocess_data
python preprocess_data.py --dataset_name wikipedia
```

**MRR not showing:**
```bash
# Check metrics.py was modified
grep "mrr" utils/metrics.py

# Should see compute_mrr function
```

---

That's everything! Just copy-paste the commands and you're good to go. 🚀
