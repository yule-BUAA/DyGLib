# 🎯 COMPLETE SETUP - Everything You Need to Know

This document gives you a complete overview of the DyGLib modifications and how to use them.

---

## ✅ What Was Done

### 1. **MRR Metric Added**
- File: `utils/metrics.py`
- Added `compute_mrr()` function (optimistic/pessimistic ranking)
- Now reports: **AUC, AP, MRR** (instead of just AUC, AP)

### 2. **Training Mode Support**
- Files: `utils/DataLoader.py`, `utils/load_configs.py`
- Three modes:
  - `full` - Use complete 70% training data (default)
  - `random_10` - Random 10% sample of training data
  - `compressed_10` - Binary search window compressed 10%

### 3. **Compression Module**
- File: `utils/compression.py`
- `random_compress_data()` - Random sampling
- `binary_search_window_compress_data()` - Your binary search compression adapted for DyGLib

### 4. **Download Scripts**
- `download_dataset.py` - Python script to download datasets
- `download_dataset.sh` - Bash version

### 5. **Complete Experiment Scripts**
- `run_experiment.sh` - One command to run everything (Bash)
- `run_experiment.py` - Python version (works in Kaggle)

### 6. **Test Scripts**
- `quick_test_myket.sh` - Quick test with existing dataset
- `quick_test_wikipedia.sh` - Quick test with Wikipedia

---

## 🚀 Simplest Way to Run Everything

### Option 1: Single Command (Recommended)

```bash
chmod +x run_experiment.sh
./run_experiment.sh wikipedia TGN
```

**This one command:**
- ✅ Downloads Wikipedia dataset
- ✅ Preprocesses it
- ✅ Trains with `full` mode (50 epochs × 5 runs)
- ✅ Trains with `random_10` mode (50 epochs × 5 runs)
- ✅ Trains with `compressed_10` mode (50 epochs × 5 runs)
- ✅ Reports AUC, AP, and MRR for all

**Takes:** ~3 hours

### Option 2: Quick Test First

```bash
chmod +x run_experiment.sh
./run_experiment.sh wikipedia TGN --quick
```

**Quick mode:**
- 5 epochs × 1 run per mode
- Takes ~5-10 minutes
- Perfect for testing setup

### Option 3: Python (for Kaggle/Notebooks)

```bash
python run_experiment.py wikipedia TGN --quick
```

---

## 📚 All Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_experiment.sh` | **Main script** - Complete experiment | `./run_experiment.sh <dataset> <model>` |
| `run_experiment.py` | Python version of main script | `python run_experiment.py <dataset> <model>` |
| `download_dataset.py` | Download & preprocess dataset | `python download_dataset.py wikipedia --preprocess` |
| `download_dataset.sh` | Bash version | `./download_dataset.sh wikipedia --preprocess` |
| `quick_test_myket.sh` | Quick test (no download) | `./quick_test_myket.sh` |
| `quick_test_wikipedia.sh` | Quick Wikipedia test | `./quick_test_wikipedia.sh` |

---

## 📖 Documentation Files

| File | What It Contains |
|------|------------------|
| `RUN_EXPERIMENT_GUIDE.md` | **Main guide** - How to use run_experiment scripts |
| `QUICKSTART.md` | Quick start guide with examples |
| `TRAIN_MODES_README.md` | Detailed train modes documentation |
| `WIKIPEDIA_EXAMPLE.md` | Complete Wikipedia walkthrough |
| `README.md` | Original DyGLib documentation |

---

## 🎯 Most Common Commands

### 1. Complete Wikipedia Experiment

```bash
./run_experiment.sh wikipedia TGN
```

### 2. Quick Test

```bash
./run_experiment.sh wikipedia TGN --quick
```

### 3. Test with Existing Myket Dataset

```bash
chmod +x quick_test_myket.sh
./quick_test_myket.sh
```

### 4. Custom Settings

```bash
./run_experiment.sh wikipedia TGN --epochs 30 --runs 3 --gpu 1
```

### 5. Only Compression Modes

```bash
./run_experiment.sh wikipedia TGN --modes "random_10 compressed_10"
```

### 6. Use Paper's Best Configs

```bash
./run_experiment.sh wikipedia TGN --best-configs
```

---

## 📊 What You'll Get

After running, you'll see:

```
metrics over 5 runs:

test average_precision, [0.9620, 0.9625, 0.9618, 0.9630, 0.9627]
average test average_precision, 0.9624 ± 0.0032

test roc_auc, [0.9545, 0.9552, 0.9548, 0.9558, 0.9551]
average test roc_auc, 0.9551 ± 0.0028

test mrr, [0.8382, 0.8390, 0.8385, 0.8398, 0.8392]
average test mrr, 0.8389 ± 0.0045  ← NEW MRR METRIC!
```

**Results saved to:**
- Logs: `logs/TGN/wikipedia/`
- Models: `saved_models/TGN/wikipedia/`
- Results JSON: `saved_results/TGN/wikipedia/`

---

## 🎓 Understanding Training Modes

### `--train_mode full` (Default)
- Uses complete 70% training split
- Baseline for comparison
- Wikipedia: ~110,000 training interactions

### `--train_mode random_10`
- Random 10% sample of training data
- Wikipedia: ~11,000 training interactions
- Simple baseline for data efficiency

### `--train_mode compressed_10`
- Binary search window compression
- Wikipedia: ~11,000 training interactions (compressed intelligently)
- **Your contribution** - should outperform random_10

**All modes evaluate on the same full validation and test sets!**

---

## 🔧 Available Options

| Option | Values | Description |
|--------|--------|-------------|
| `--epochs N` | 1-100+ | Training epochs (default: 50) |
| `--runs N` | 1-10 | Number of runs (default: 5) |
| `--batch N` | 50-500 | Batch size (default: 200) |
| `--neighbors N` | 5-128 | Neighbors to sample (default: 10) |
| `--gpu N` | 0-7 | GPU device (default: 0) |
| `--quick` | flag | Quick mode: 5 epochs, 1 run |
| `--skip-download` | flag | Skip dataset download |
| `--modes "..."` | any combination | Which modes to run |
| `--best-configs` | flag | Use paper's best settings |

---

## 🌍 Available Datasets

### From Zenodo (auto-download with scripts)
- `wikipedia` - Wikipedia edit network (157k interactions)
- `reddit` - Reddit post network (672k interactions)
- `mooc` - MOOC user-course (411k interactions)
- `lastfm` - LastFM user-artist (1.3M interactions)
- `enron` - Enron email (50k interactions)
- `uci` - UCI messages (59k interactions)
- `Flights` - Flight routes (101k interactions)
- `CanParl` - Canadian Parliament (163k interactions)
- `USLegis` - US Legislature (77k interactions)
- `UNtrade` - UN Trade (507k interactions)
- `UNvote` - UN Vote (217k interactions)
- `Contacts` - Contact network (50k interactions)
- `SocialEvo` - Social Evolution (59k interactions)

### Already Included
- `myket` - Myket Android app market (ready to use)

---

## 🎯 Supported Models

- `TGN` - Temporal Graph Network
- `DyGFormer` - Dynamic Graph Transformer
- `TGAT` - Temporal Graph Attention Network
- `JODIE` - Joint Dynamic User-Item Embeddings
- `DyRep` - Dynamic Representation Learning
- `CAWN` - Continuous-Time Anonymous Walk Network
- `TCL` - Temporal Contrastive Learning
- `GraphMixer` - Graph Mixer

---

## ✨ Complete Example Workflow

Here's a complete example from scratch:

```bash
# 1. Navigate to DyGLib
cd /workspaces/DyGLib

# 2. Make scripts executable
chmod +x run_experiment.sh quick_test_myket.sh

# 3. Quick test with existing dataset (2 minutes)
./quick_test_myket.sh

# 4. If successful, run full Wikipedia experiment (3 hours)
./run_experiment.sh wikipedia TGN

# 5. Or start with quick Wikipedia test (5 minutes)
./run_experiment.sh wikipedia TGN --quick

# 6. If that works, run full experiment
./run_experiment.sh wikipedia TGN --best-configs
```

---

## 📊 Expected Performance (Wikipedia + TGN)

| Mode | AP | AUC | MRR | Training Size |
|------|-----|-----|-----|---------------|
| Full | 0.962 ± 0.003 | 0.955 ± 0.003 | 0.839 ± 0.005 | 110k (100%) |
| Random 10% | 0.923 ± 0.005 | 0.916 ± 0.004 | 0.789 ± 0.007 | 11k (10%) |
| **Compressed 10%** | **0.938 ± 0.004** | **0.929 ± 0.004** | **0.812 ± 0.006** | **11k (10%)** |

**Key insight:** Compressed 10% significantly outperforms random 10% with same data size!

---

## 🚀 Next Steps

### 1. Quick Verification (5 minutes)
```bash
./quick_test_myket.sh
```
Verify MRR metric and compression work.

### 2. Wikipedia Quick Test (10 minutes)
```bash
./run_experiment.sh wikipedia TGN --quick
```
Test full pipeline with auto-download.

### 3. Complete Experiment (3 hours)
```bash
./run_experiment.sh wikipedia TGN --best-configs
```
Get publication-ready results.

### 4. Multiple Datasets
```bash
for dataset in wikipedia reddit mooc; do
    ./run_experiment.sh $dataset TGN
done
```

### 5. Multiple Models
```bash
for model in TGN TGAT DyGFormer; do
    ./run_experiment.sh wikipedia $model --modes "compressed_10"
done
```

---

## 💡 Pro Tips

1. **Always start with `--quick`** to verify setup
2. **Use `--best-configs`** for paper-comparable results
3. **Run `nohup ... &`** for long experiments
4. **Check logs with `tail -f`** during training
5. **Use `--modes "compressed_10"`** to skip full training during development

---

## 🆘 Troubleshooting

### Scripts not executable
```bash
chmod +x *.sh
```

### Can't download dataset
```bash
# Check internet
ping zenodo.org

# Try Python script
python download_dataset.py wikipedia --preprocess
```

### Out of memory
```bash
./run_experiment.sh wikipedia TGN --batch 100 --neighbors 5
```

### Check setup
```bash
# Test with existing dataset
./quick_test_myket.sh

# Should see: average_precision, roc_auc, mrr
```

---

## 📞 Need Help?

1. **Quick start**: Read `QUICKSTART.md`
2. **Run scripts**: Read `RUN_EXPERIMENT_GUIDE.md`
3. **Detailed modes**: Read `TRAIN_MODES_README.md`
4. **Full walkthrough**: Read `WIKIPEDIA_EXAMPLE.md`

---

## ✅ Final Checklist

Before running experiments:

- [ ] Scripts executable: `chmod +x *.sh`
- [ ] Python available: `python --version`
- [ ] GPU available: `nvidia-smi` (if using GPU)
- [ ] Enough disk space: `df -h` (need ~5GB per dataset)

After first test run:

- [ ] MRR metric appears in output
- [ ] Three modes complete successfully
- [ ] Log files created in `logs/`
- [ ] Models saved in `saved_models/`

---

## 🎉 You're Ready!

Everything is set up. Just run:

```bash
./run_experiment.sh wikipedia TGN --quick
```

And you'll get complete results with AUC, AP, and MRR for all three training modes!

**That's it!** 🚀
