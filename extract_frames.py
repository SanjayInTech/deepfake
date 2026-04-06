import cv2
import os
import glob
import json

def extract_frames_from_list(video_paths, output_img_folder, frame_skip=15):
    """
    Extracts frames from a list of video paths.
    """
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)

    for video_in in video_paths:
        if not os.path.exists(video_in):
            continue
        
        vid_name = os.path.basename(video_in).split('.')[0]
        # Use a subfolder for each video to keep things organized if needed, 
        # but the project structure expects them all in one 'real' or 'fake' folder.
        
        vidcap = cv2.VideoCapture(video_in)
        success, image = vidcap.read()
        count = 0
        saved_count = 0
        
        while success:
            if count % frame_skip == 0:
                frame_filename = os.path.join(output_img_folder, f"{vid_name}_frame_{count}.jpg")
                cv2.imwrite(frame_filename, image)
                saved_count += 1
                
            success, image = vidcap.read()
            count += 1
        
        vidcap.release()
        print(f"✅ Extracted {saved_count} frames from {vid_name}")

def get_celeb_df_paths(raw_dir):
    real = []
    fake = []
    celeb_raw = os.path.join(raw_dir, 'celeb_df')
    if os.path.exists(celeb_raw):
        # Real
        for folder in ['Celeb-real', 'YouTube-real']:
            real.extend(glob.glob(os.path.join(celeb_raw, folder, "*.mp4")))
        # Fake
        fake.extend(glob.glob(os.path.join(celeb_raw, 'Celeb-synthesis', "*.mp4")))
    return real, fake

def get_dfdc_paths(raw_dir):
    real = []
    fake = []
    dfdc_raw = os.path.join(raw_dir, 'dfdc')
    if os.path.exists(dfdc_raw):
        part_folders = glob.glob(os.path.join(dfdc_raw, "dfdc_train_part_*"))
        for part in part_folders:
            meta_path = os.path.join(part, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                for vid_name, info in metadata.items():
                    full_path = os.path.join(part, vid_name)
                    if info['label'] == 'REAL':
                        real.append(full_path)
                    else:
                        fake.append(full_path)
    return real, fake

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'dataset', 'raw')
    
    img_real = os.path.join(base_dir, 'dataset', 'images', 'real')
    img_fake = os.path.join(base_dir, 'dataset', 'images', 'fake')
    
    print("🔍 Searching for videos in raw folders...")
    
    c_real, c_fake = get_celeb_df_paths(raw_dir)
    d_real, d_fake = get_dfdc_paths(raw_dir)
    
    all_real = c_real + d_real
    all_fake = c_fake + d_fake
    
    print(f"Found {len(all_real)} Real videos and {len(all_fake)} Fake videos.")
    
    # Extract a limited number of frames to avoid disk issues
    # Frame skip 15 means roughly 2 frames per second
    extract_frames_from_list(all_real, img_real, frame_skip=15)
    extract_frames_from_list(all_fake, img_fake, frame_skip=15)
    
    print("\n🚀 Frame extraction complete! Next step: python preprocess_dataset.py")
