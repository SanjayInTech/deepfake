import os
import cv2
import glob
from mtcnn import MTCNN

def debug_preprocess():
    detector = MTCNN()
    class_name = 'fake'
    img_dir = os.path.join('dataset', 'images', class_name)
    out_dir = os.path.join('dataset', 'cropped_images', class_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Simple non-recursive glob first
    image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    print(f"Found {len(image_paths)} images in {img_dir}")
    
    count = 0
    for i, img_path in enumerate(image_paths):
        print(f"Processing {i}: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print("  Image read failed")
            continue
        
        try:
            faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if faces:
                face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                x, y, w, h = face['box']
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
                cropped = img[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(out_dir, f"debug_fake_{count}.jpg"), cropped)
                count += 1
                print(f"  ✅ Saved face {count}")
            else:
                print("  ❌ No face detected")
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            
        if count >= 10:
            break

if __name__ == "__main__":
    debug_preprocess()
