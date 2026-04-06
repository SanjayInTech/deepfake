import os
import shutil
import json
import glob

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'dataset', 'raw')
VIDEO_DIR = os.path.join(BASE_DIR, 'dataset', 'videos')
VIDEO_REAL = os.path.join(VIDEO_DIR, 'real')
VIDEO_FAKE = os.path.join(VIDEO_DIR, 'fake')

def setup_dirs():
    os.makedirs(VIDEO_REAL, exist_ok=True)
    os.makedirs(VIDEO_FAKE, exist_ok=True)

def process_celeb_df():
    print("Sorting Celeb-DF videos...")
    celeb_raw = os.path.join(RAW_DIR, 'celeb_df')
    if not os.path.exists(celeb_raw):
        print("⚠️ Celeb-DF raw folder not found.")
        return

    # Real folders
    for folder in ['Celeb-real', 'YouTube-real']:
        src = os.path.join(celeb_raw, folder)
        if os.path.exists(src):
            for f in glob.glob(os.path.join(src, "*.mp4")):
                shutil.copy(f, VIDEO_REAL)
    
    # Fake folders
    src = os.path.join(celeb_raw, 'Celeb-synthesis')
    if os.path.exists(src):
        for f in glob.glob(os.path.join(src, "*.mp4")):
            shutil.copy(f, VIDEO_FAKE)
    print("✅ Celeb-DF sorted.")

def process_dfdc():
    print("Sorting DFDC videos...")
    dfdc_raw = os.path.join(RAW_DIR, 'dfdc')
    if not os.path.exists(dfdc_raw):
        print("⚠️ DFDC raw folder not found.")
        return

    # Find all part folders
    part_folders = glob.glob(os.path.join(dfdc_raw, "dfdc_train_part_*"))
    for part in part_folders:
        meta_path = os.path.join(part, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        for vid_name, info in metadata.items():
            src_path = os.path.join(part, vid_name)
            if not os.path.exists(src_path):
                continue
            
            target = VIDEO_REAL if info['label'] == 'REAL' else VIDEO_FAKE
            shutil.copy(src_path, target)
    print("✅ DFDC sorted.")

if __name__ == "__main__":
    setup_dirs()
    process_celeb_df()
    process_dfdc()
    print(f"\nAll videos sorted into:\n - {VIDEO_REAL}\n - {VIDEO_FAKE}")
