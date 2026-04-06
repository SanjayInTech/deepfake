import os
import cv2
import glob
from mtcnn import MTCNN

# Multi-Dataset Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'images') 
MASTER_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'cropped_images')

detector = MTCNN()

def process_and_merge(class_name):
    output_folder = os.path.join(MASTER_DATA_DIR, class_name)
    os.makedirs(output_folder, exist_ok=True)
    
    search_pattern_jpg = os.path.join(RAW_DATA_DIR, '**', class_name, '*.jpg')
    search_pattern_png = os.path.join(RAW_DATA_DIR, '**', class_name, '*.png')
    
    image_paths = glob.glob(search_pattern_jpg, recursive=True) + \
                  glob.glob(search_pattern_png, recursive=True)
                  
    if not image_paths:
        print(f"⚠️ No '{class_name}' images found.")
        return

    print(f"🔍 Processing {len(image_paths)} '{class_name}' images...")
    success_count = 0
    max_images = 10000 

    for i, img_path in enumerate(image_paths):
        if success_count >= max_images:
            break
        try:
            parent_dataset = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            clean_img_name = f"{parent_dataset}_{class_name}_{i}.jpg"
            
            # Skip if already exists
            out_path = os.path.join(output_folder, clean_img_name)
            if os.path.exists(out_path):
                success_count += 1
                continue

            img = cv2.imread(img_path)
            if img is None: continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            
            if faces:
                face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                x, y, w, h = face['box']
                margin_x, margin_y = int(w * 0.1), int(h * 0.1)
                x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
                x2, y2 = min(img.shape[1], x + w + margin_x), min(img.shape[0], y + h + margin_y)
                cropped_face = img[y1:y2, x1:x2]
                cv2.imwrite(out_path, cropped_face)
                success_count += 1
        except Exception as e:
            continue
            
        if (i+1) % 100 == 0:
            print(f"   {class_name}: Processed {i+1}, Saved {success_count}...")

    print(f"✅ Finished {class_name}.\n")

if __name__ == "__main__":
    process_and_merge('fake')
