import os

def main():
    print("=========================================================")
    print("🌍 DEEPFAKE MULTI-DATASET DOWNLOADER (KAGGLE)")
    print("=========================================================")
    print("To make your model extremely powerful, you should train it on")
    print("multiple datasets so it doesn't overfit to just one type of fake.\n")
    
    print("⚠️  REQUIREMENT: You must have the Kaggle API installed and configured.")
    print("   Run: pip install kaggle\n")

    print("Below are the terminal commands to download the Top 3 Datasets")
    print("directly into your raw data folder:\n")

    print("-------- 1. FaceForensics++ (Standard Benchmark) --------")
    print("kaggle datasets download -d greatgamedota/faceforensics -p dataset/raw/faceforensics --unzip\n")

    print("-------- 2. Celeb-DF (Extremely High Quality Fakes) --------")
    print("kaggle datasets download -d reubensuju/celeb-df-v2 -p dataset/raw/celeb_df --unzip\n")

    print("-------- 3. DFDC (Facebook/Meta AI Challenge - 10% Subset) --------")
    print("kaggle datasets download -d pranay22077/dfdc-10 -p dataset/raw/dfdc --unzip\n")
    
    print("Once downloaded, run `python preprocess_dataset.py` to ")
    print("automatically crop and merge all of them into the master folder!")

if __name__ == "__main__":
    main()
