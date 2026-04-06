import os
import cv2
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from mtcnn import MTCNN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Top-Level Xception Model
MODEL_PATH = 'deepfake_xception_model.h5'
model = None

# Initialize Face Detector
print("[INFO] Initializing MTCNN Face Detector...")
face_detector = MTCNN()

if os.path.exists(MODEL_PATH):
    print("[INFO] Loading Xception model into memory...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
else:
    print("⚠️ WARNING: deepfake_xception_model.h5 not found! Please run train_model.py first.")

def extract_and_prepare_face(image_path):
    """Uses MTCNN to crop the face, then prepares it for Xception (299x299)."""
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError("Invalid image file.")
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Detect face
    faces = face_detector.detect_faces(img_rgb)
    
    if faces:
        # Get largest face
        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = face['box']
        
        # 10% margin
        margin_x, margin_y = int(w * 0.1), int(h * 0.1)
        x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
        x2, y2 = min(img_rgb.shape[1], x + w + margin_x), min(img_rgb.shape[0], y + h + margin_y)
        
        cropped_face = img_rgb[y1:y2, x1:x2]
        img_pil = Image.fromarray(cropped_face).resize((299, 299)) # Xception size
    else:
        # Fallback if no face detected (resize full image)
        print("[WARNING] No face detected by MTCNN, using full image as fallback.")
        img_pil = Image.fromarray(img_rgb).resize((299, 299))
        
    img_array = np.array(img_pil) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0) # Batch dim
    return img_array

def process_video(video_path, num_frames=10):
    """Extracts frames from a video, analyzes them, and averages the predictions."""
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If the video is very short, adjust num_frames
    if total_frames < num_frames:
        num_frames = max(1, total_frames)
        
    step = max(1, total_frames // num_frames)
    
    predictions = []
    
    for i in range(num_frames):
        frame_id = i * step
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, image = vidcap.read()
        
        if success:
            # Save frame temporarily
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_frame_{i}.jpg")
            cv2.imwrite(temp_path, image)
            
            try:
                # Process frame
                img_array = extract_and_prepare_face(temp_path)
                pred = model.predict(img_array, verbose=0)[0][0]
                predictions.append(pred)
            except Exception as e:
                print(f"Skipping frame {i} due to error: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
    vidcap.release()
    
    if not predictions:
        raise ValueError("Could not extract any valid faces from the video.")
        
    # Fake=0, Real=1
    average_prediction = np.mean(predictions)
    return average_prediction

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Top-level Model not trained yet!'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            is_video = filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            
            if is_video:
                print(f"[INFO] Processing video: {filename}")
                prediction = process_video(filepath)
            else:
                print(f"[INFO] Processing image: {filename}")
                img_array = extract_and_prepare_face(filepath)
                prediction = model.predict(img_array)[0][0]
            
            # 3. Output Translation (Fake=0, Real=1)
            result = "Real Media" if prediction > 0.5 else "Fake Media"
            confidence = float(prediction if result == "Real Media" else 1 - prediction)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return jsonify({
                'result': result,
                'confidence': f"{confidence * 100:.2f}%"
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("[INFO] Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
