import os
import cv2
import glob
from mtcnn import MTCNN

def prepare_subset(limit=500):
    detector = MTCNN()
    base_out = os.path.join('dataset', 'cropped_images')
    if os.path.exists(base_out):
        import shutil
        shutil.rmtree(base_out)
    os.makedirs(base_out, exist_ok=True)
    
    classes = ['real', 'fake']
    for cls in classes:
        print(f"Processing {cls}...")
        out_dir = os.path.join(base_out, cls)
        os.makedirs(out_dir, exist_ok=True)
        img_dir = os.path.join('dataset', 'images', cls)
        
        image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        if not image_paths:
            print(f"  ⚠️ No images found in {img_dir}")
            continue
            
        count = 0
        for i, p in enumerate(image_paths):
            try:
                img = cv2.imread(p)
                if img is None: continue
                faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if faces:
                    f = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = f['box']
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
                    crop = img[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(out_dir, f"crop_{count}.jpg"), crop)
                    count += 1
                    if count % 50 == 0:
                        print(f"  Done {count}/{limit}")
                    if count >= limit:
                        break
            except Exception as e:
                continue
        print(f"✅ Finished {cls}: {count} crops.")

if __name__ == "__main__":
    prepare_subset(10000)
