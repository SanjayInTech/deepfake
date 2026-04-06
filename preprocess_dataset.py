import os
import cv2
import glob
from mtcnn import MTCNN

# Multi-Dataset Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Here is where you drop FaceForensics, Celeb-DF, DFDC folders
RAW_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'images') 
# Here is where the ultimate uniform dataset goes
MASTER_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'cropped_images')

detector = MTCNN()

def process_and_merge(class_name):
    """
    class_name should be 'real' or 'fake'.
    This will search recursively through ALL dataset folders inside dataset/raw/
    and pull out any image located in a 'real' or 'fake' subfolder, crop it,
    and consolidate it into dataset/cropped_images/real (or fake).
    """
    output_folder = os.path.join(MASTER_DATA_DIR, class_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Recursively find all JPG and PNG files inside ANY folder named 'real' or 'fake'
    search_pattern_jpg = os.path.join(RAW_DATA_DIR, '**', class_name, '*.jpg')
    search_pattern_png = os.path.join(RAW_DATA_DIR, '**', class_name, '*.png')
    
    image_paths = glob.glob(search_pattern_jpg, recursive=True) + \
                  glob.glob(search_pattern_png, recursive=True)
                  
    if not image_paths:
        print(f"⚠️ No '{class_name}' images found across your datasets.")
        return

    print(f"🔍 Aggregating {len(image_paths)} '{class_name}' images from all datasets...")
    success_count = 0
    max_images = 10000 # Increased for 20k total (10k real + 10k fake)

    for i, img_path in enumerate(image_paths):
        if success_count >= max_images:
            break
        try:
            # We rename the image logically to avoid name collisions between different datasets
            parent_dataset = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            clean_img_name = f"{parent_dataset}_{class_name}_{i}.jpg"
            
            img = cv2.imread(img_path)
            if img is None: continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            
            if faces:
                # Crop the largest face
                face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                x, y, w, h = face['box']
                
                # 10% safety margin
                margin_x, margin_y = int(w * 0.1), int(h * 0.1)
                x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
                x2, y2 = min(img.shape[1], x + w + margin_x), min(img.shape[0], y + h + margin_y)
                
                cropped_face = img[y1:y2, x1:x2]
                
                # Save to the master folder
                out_path = os.path.join(output_folder, clean_img_name)
                cv2.imwrite(out_path, cropped_face)
                success_count += 1
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            continue
            
        # Logging progress
        if (i+1) % 500 == 0:
            print(f"   Processed {i+1}/{len(image_paths)}...")

    print(f"✅ Successfully fused {success_count} '{class_name}' faces into the Master Dataset.\n")

if __name__ == "__main__":
    print("=========================================================")
    print("🧬 ULTIMATE MULTI-DATASET ALIGNMENT & EXTRACTION PIPELINE")
    print("=========================================================")
    print(f"Scanning {RAW_DATA_DIR} for subsets (FaceForensics, Celeb-DF, DFDC)...\n")
    
    process_and_merge('real')
    process_and_merge('fake')
    
    print("🎉 Master Dataset is ready for high-accuracy Training at: dataset/cropped_images/")
