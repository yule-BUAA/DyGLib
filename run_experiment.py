#!/usr/bin/env python3
"""
Complete experiment script for any dataset and model
Can be used in Kaggle notebooks or run as a standalone script

Usage:
    python run_experiment.py wikipedia TGN
    python run_experiment.py reddit DyGFormer --quick
    python run_experiment.py mooc TGAT --epochs 30 --runs 3
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"\n{'='*60}")
        print(description)
        print('='*60)

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Command failed with exit code {e.returncode}")
        return False


def check_dataset(dataset_name):
    """Check if dataset is preprocessed"""
    processed_path = Path(f"processed_data/{dataset_name}/ml_{dataset_name}.csv")
    return processed_path.exists()


def download_dataset(dataset_name):
    """Download and preprocess dataset"""
    print(f"📥 Downloading and preprocessing {dataset_name}...")
    cmd = ["python", "download_dataset.py", dataset_name, "--preprocess"]
    return run_command(cmd, f"Downloading {dataset_name} dataset")


def train_model(dataset, model, mode, epochs, runs, batch_size, num_neighbors, gpu, extra_args):
    """Train model with specified parameters"""
    cmd = [
        "python", "train_link_prediction.py",
        "--dataset_name", dataset,
        "--model_name", model,
        "--train_mode", mode,
        "--batch_size", str(batch_size),
        "--num_neighbors", str(num_neighbors),
        "--num_epochs", str(epochs),
        "--num_runs", str(runs),
        "--gpu", str(gpu)
    ]

    # Add extra arguments
    cmd.extend(extra_args)

    return run_command(cmd, f"Training {model} on {dataset} with {mode} mode")


def print_results(model, dataset, modes):
    """Print results summary"""
    print("\n" + "="*60)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print(f"  Logs:    ./logs/{model}/{dataset}/")
    print(f"  Models:  ./saved_models/{model}/{dataset}/")
    print(f"  Results: ./saved_results/{model}/{dataset}/")

    # Try to show quick summary from logs
    log_dir = Path(f"logs/{model}/{dataset}")
    if log_dir.exists():
        print("\n" + "="*60)
        print("Test Results Summary:")
        print("="*60)

        for mode in modes:
            print(f"\n📊 {mode.upper()} mode:")
            # Find most recent log file
            log_files = sorted(log_dir.glob("*/*.log"), key=os.path.getmtime, reverse=True)
            if log_files:
                with open(log_files[0], 'r') as f:
                    lines = f.readlines()
                    # Find lines with "average test"
                    test_lines = [l.strip() for l in lines if "average test" in l.lower()]
                    if test_lines:
                        for line in test_lines[-3:]:  # Last 3 metrics
                            print(f"  {line}")
                    else:
                        print("  No results found in log")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete DyGLib experiment for any dataset and model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s wikipedia TGN
  %(prog)s reddit DyGFormer --quick
  %(prog)s mooc TGAT --epochs 30 --runs 3 --gpu 1
  %(prog)s wikipedia TGN --modes full compressed_10 --best-configs
        """
    )

    # Required arguments
    parser.add_argument("dataset", type=str,
                        help="Dataset name (wikipedia, reddit, mooc, etc.)")
    parser.add_argument("model", type=str,
                        help="Model name (TGN, DyGFormer, TGAT, etc.)")

    # Optional arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs (default: 5)")
    parser.add_argument("--batch", type=int, default=200,
                        help="Batch size (default: 200)")
    parser.add_argument("--neighbors", type=int, default=10,
                        help="Number of neighbors (default: 10)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device (default: 0)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 epochs, 1 run")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download")
    parser.add_argument("--modes", type=str, default="full random_10 compressed_10",
                        help="Train modes (default: 'full random_10 compressed_10')")
    parser.add_argument("--best-configs", action="store_true",
                        help="Use best configurations from paper")

    args = parser.parse_args()

    # Apply quick mode
    if args.quick:
        args.epochs = 5
        args.runs = 1

    # Parse modes
    modes = args.modes.split()

    # Extra arguments
    extra_args = []
    if args.best_configs:
        extra_args.append("--load_best_configs")

    # Print configuration
    print("="*60)
    print("DyGLib Complete Experiment")
    print("="*60)
    print(f"\n📋 Configuration:")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Model:        {args.model}")
    print(f"  Train modes:  {' '.join(modes)}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Runs:         {args.runs}")
    print(f"  Batch size:   {args.batch}")
    print(f"  Neighbors:    {args.neighbors}")
    print(f"  GPU:          {args.gpu}")
    if extra_args:
        print(f"  Extra args:   {' '.join(extra_args)}")

    # Step 1: Check/download dataset
    if not args.skip_download:
        if not check_dataset(args.dataset):
            print(f"\n⚠️  Dataset {args.dataset} not found. Downloading...")
            if not download_dataset(args.dataset):
                print("❌ Failed to download dataset")
                sys.exit(1)
        else:
            print(f"\n✅ Dataset {args.dataset} already preprocessed")
    else:
        print("\n⚠️  Skipping download (--skip-download)")

    # Step 2: Train with each mode
    for i, mode in enumerate(modes, 1):
        print(f"\n{'='*60}")
        print(f"Step {i+1}: Training with {mode} mode")
        print('='*60)

        success = train_model(
            dataset=args.dataset,
            model=args.model,
            mode=mode,
            epochs=args.epochs,
            runs=args.runs,
            batch_size=args.batch,
            num_neighbors=args.neighbors,
            gpu=args.gpu,
            extra_args=extra_args
        )

        if not success:
            print(f"❌ Training failed for {mode} mode")
            sys.exit(1)

    # Print results summary
    print_results(args.model, args.dataset, modes)
    print("\n🎉 Experiment complete!")


if __name__ == "__main__":
    main()
