import os
import subprocess
import sys

def install_kaggle():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

def download_dataset(dataset_slug, output_path):
    print(f"\n📥 Downloading {dataset_slug}...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", output_path, "--unzip"],
            check=True
        )
        print(f"✅ {dataset_slug} downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {dataset_slug}: {e}")
        print("   Make sure your Kaggle credentials are configured (kaggle.json)")
        return False
    return True

def main():
    print("=" * 60)
    print("DEEPFAKE DATASET DOWNLOADER (KAGGLE)")
    print("=" * 60)
    print("\nThis script downloads deepfake datasets from Kaggle.\n")

    try:
        import kaggle
    except ImportError:
        print("📦 Installing kaggle...")
        install_kaggle()

    datasets = [
        ("shahrik123/faceforensics-original", "dataset/raw/faceforensics", "FaceForensics++"),
        ("sdESDLH17AI/FaceForensics", "dataset/raw/celeb_df", "Celeb-DF"),
        ("pranay22077/dfdc-10", "dataset/raw/dfdc", "DFDC"),
    ]

    for dataset_slug, output_path, name in datasets:
        os.makedirs(output_path, exist_ok=True)
        download_dataset(dataset_slug, output_path)

    print("\n" + "=" * 60)
    print("Download complete! Run `python preprocess_dataset.py` to process the data.")

if __name__ == "__main__":
    main()
