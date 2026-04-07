"""
Download and Setup Script for DyGLib Datasets
Can be run in Kaggle notebooks or locally

Usage:
    python download_dataset.py wikipedia --preprocess
    python download_dataset.py reddit --preprocess
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path
import subprocess


# Dataset URLs from Zenodo
DATASET_URLS = {
    "wikipedia": "https://zenodo.org/records/7213796/files/wikipedia.zip?download=1",
    "reddit": "https://zenodo.org/records/7213796/files/reddit.zip?download=1",
    "mooc": "https://zenodo.org/records/7213796/files/mooc.zip?download=1",
    "lastfm": "https://zenodo.org/records/7213796/files/lastfm.zip?download=1",
    "enron": "https://zenodo.org/records/7213796/files/enron.zip?download=1",
    "uci": "https://zenodo.org/records/7213796/files/uci.zip?download=1",
    "Flights": "https://zenodo.org/records/7213796/files/Flights.zip?download=1",
    "CanParl": "https://zenodo.org/records/7213796/files/CanParl.zip?download=1",
    "USLegis": "https://zenodo.org/records/7213796/files/USLegis.zip?download=1",
    "UNtrade": "https://zenodo.org/records/7213796/files/UNtrade.zip?download=1",
    "UNvote": "https://zenodo.org/records/7213796/files/UNvote.zip?download=1",
    "Contacts": "https://zenodo.org/records/7213796/files/Contacts.zip?download=1",
    "SocialEvo": "https://zenodo.org/records/7213796/files/SocialEvo.zip?download=1",
}


def download_with_progress(url, output_path):
    """Download file with progress bar"""
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook)
    print()  # New line after progress


def download_dataset(dataset_name, base_path="DG_data"):
    """Download and extract dataset"""

    if dataset_name not in DATASET_URLS:
        print(f"❌ Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join(DATASET_URLS.keys())}")
        return False

    url = DATASET_URLS[dataset_name]

    # Create dataset folder
    dataset_folder = Path(base_path) / dataset_name
    dataset_folder.mkdir(parents=True, exist_ok=True)

    zip_path = dataset_folder / f"{dataset_name}.zip"
    csv_path = dataset_folder / f"{dataset_name}.csv"

    # Check if already exists
    if csv_path.exists():
        print(f"ℹ️  Dataset {dataset_name}.csv already exists")
        response = input("Re-download? (y/n): ").lower()
        if response != 'y':
            print("Skipping download.")
            return True

    print(f"📥 Downloading {dataset_name} from Zenodo...")

    try:
        download_with_progress(url, str(zip_path))
        print(f"✓ Downloaded to {zip_path}")

        # Unzip
        print(f"📦 Extracting {dataset_name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_folder)
        except Exception as e:
            # If Python's zipfile fails, try system unzip command
            print(f"⚠️  Python unzip failed ({e}), trying system unzip...")
            import subprocess
            result = subprocess.run(
                ['unzip', '-q', '-o', str(zip_path), '-d', str(dataset_folder)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"System unzip failed: {result.stderr}")

        # Remove zip file
        zip_path.unlink()

        # Handle nested folder structure (zip extracts to dataset_name/dataset_name/)
        nested_folder = dataset_folder / dataset_name
        if nested_folder.exists() and nested_folder.is_dir():
            print(f"📁 Moving files from nested folder...")
            for item in nested_folder.iterdir():
                item.rename(dataset_folder / item.name)
            nested_folder.rmdir()

        # Check if preprocessed files already exist in the download
        ml_csv = dataset_folder / f"ml_{dataset_name}.csv"
        if ml_csv.exists():
            print(f"✅ Found preprocessed files in download!")
            # Copy to processed_data folder
            processed_folder = Path("processed_data") / dataset_name
            processed_folder.mkdir(parents=True, exist_ok=True)

            import shutil
            for ml_file in dataset_folder.glob(f"ml_{dataset_name}*"):
                shutil.copy2(ml_file, processed_folder / ml_file.name)

            print(f"✅ Preprocessed files copied to processed_data/{dataset_name}/")

        # Verify CSV exists
        if csv_path.exists():
            print(f"✅ {dataset_name}.csv ready at {csv_path}")
            return True
        else:
            print(f"⚠️  Warning: {dataset_name}.csv not found after extraction")
            return False

    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return False


def preprocess_dataset(dataset_name, node_feat_dim=172):
    """Preprocess the downloaded dataset"""

    # Check if already preprocessed
    processed_csv = Path(f"processed_data/{dataset_name}/ml_{dataset_name}.csv")
    if processed_csv.exists():
        print(f"\n✅ Dataset {dataset_name} already preprocessed!")
        print(f"📁 Preprocessed data at: processed_data/{dataset_name}/")
        return True

    print(f"\n🔧 Preprocessing {dataset_name}...")

    try:
        # Run preprocessing script
        result = subprocess.run(
            ["python", "preprocess_data/preprocess_data.py",
             "--dataset_name", dataset_name,
             "--node_feat_dim", str(node_feat_dim)],
            check=True,
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(f"✅ {dataset_name} preprocessed successfully")
        print(f"📁 Processed data at: processed_data/{dataset_name}/")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error preprocessing {dataset_name}:")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and setup DyGLib datasets")
    parser.add_argument("dataset_name", type=str,
                        choices=list(DATASET_URLS.keys()),
                        help="Name of the dataset to download")
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess the dataset after downloading")
    parser.add_argument("--node_feat_dim", type=int, default=172,
                        help="Dimension of node features (default: 172)")
    parser.add_argument("--base_path", type=str, default="DG_data",
                        help="Base path for datasets (default: DG_data)")

    args = parser.parse_args()

    print("=" * 60)
    print("DyGLib Dataset Download and Setup")
    print("=" * 60)
    print()

    # Download
    success = download_dataset(args.dataset_name, args.base_path)

    if not success:
        sys.exit(1)

    # Preprocess if requested
    if args.preprocess:
        success = preprocess_dataset(args.dataset_name, args.node_feat_dim)
        if not success:
            sys.exit(1)
        print("\n✅ Setup Complete!")
    else:
        print("\n✅ Download Complete!")
        print(f"💡 To preprocess, run:")
        print(f"   python preprocess_data/preprocess_data.py --dataset_name {args.dataset_name}")


if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("Usage: python download_dataset.py <dataset_name> [--preprocess]")
        print()
        print("Available datasets:")
        for dataset in DATASET_URLS.keys():
            print(f"  - {dataset}")
        print()
        print("Examples:")
        print("  python download_dataset.py wikipedia --preprocess")
        print("  python download_dataset.py reddit --preprocess")
        print()
        sys.exit(0)

    main()
