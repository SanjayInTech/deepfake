# DEEPFAKE DETECTION SYSTEM

## PROJECT TITLE
**Deepfake Detection System using Xception CNN with MTCNN Face Detection**

---

## PROJECT DESCRIPTION
This project is a web-based deepfake detection application that can analyze images and videos to determine whether the content is real or fake (AI-generated). The system uses a deep learning model based on Xception architecture pretrained on ImageNet and fine-tuned on a custom deepfake dataset. The frontend provides an intuitive glassmorphism UI where users can upload images or videos for instant analysis.

**Key Features:**
- Image deepfake detection (JPG, PNG)
- Video deepfake detection (MP4, AVI, MOV, MKV)
- Multi-frame analysis for videos with averaged predictions
- MTCNN-based automatic face detection and cropping
- Real-time prediction with confidence score
- User-friendly web interface with drag-and-drop support

---

## TOOLS & TECHNOLOGIES

### Programming Languages
- **Python** - Backend development, ML model training
- **HTML/CSS/JavaScript** - Frontend UI development

### Frameworks & Libraries
| Category | Tools |
|----------|-------|
| Web Framework | Flask 3.0.3 |
| Deep Learning | TensorFlow 2.21.0 |
| Computer Vision | OpenCV 4.10.0.84, MTCNN 1.0.0 |
| Image Processing | Pillow 10.4.0 |
| Frontend | Vanilla JavaScript, CSS3 |

### ML Model
- **Base Model**: Xception (pretrained on ImageNet)
- **Training Approach**: Transfer Learning with two-phase training
  - Phase 1: Train top layers only (frozen base)
  - Phase 2: Fine-tune top layers of base model

### Dataset
- Real Videos: 890 videos from Face2Face dataset
- Fake Videos: 500+ face-swap videos
- Total Cropped Images: 20,190 (11,687 real + 8,503 fake)

---

## SYSTEM ARCHITECTURE

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│  Frontend   │ ───► │  Flask API   │ ───► │ Xception Model │
│  (HTML/CSS) │      │  (app.py)   │      │ (.h5 file)     │
└─────────────┘      └──────────────┘      └────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  MTCNN Face  │
                    │  Detector    │
                    └──────────────┘
```

---

## HOW IT WORKS

1. **User Upload**: User uploads an image or video via web interface
2. **Face Detection**: MTCNN detects and crops the face from the input
3. **Preprocessing**: Face is resized to 299x299 (Xception input size)
4. **Prediction**: Model analyzes the face and outputs probability
5. **Result Display**: Shows "Real Media" or "Fake Media" with confidence %

---

## PROJECT FOLDER STRUCTURE

```
deepfake/
├── app.py                    # Flask backend application
├── train_model.py            # Model training script
├── deepfake_xception_model.h5 # Trained model (166MB)
├── requirements.txt          # Python dependencies
├── PROJECT_DIARY.md         # Project development diary
├── templates/
│   └── index.html           # Frontend UI
├── test_samples/            # 40 test files for testing
├── dataset/
│   ├── videos/              # Original videos
│   └── cropped_images/      # Preprocessed face images
└── uploads/                 # Temporary upload folder
```

---

## SETUP & RUN

### Installation
```bash
pip install -r requirements.txt
```

### Run Application
```bash
python app.py
```

### Access
Open browser: http://127.0.0.1:5000

---

## ACCURACY & PERFORMANCE
- Model achieves high accuracy on test dataset
- Supports both images and videos
- Average prediction time: <2 seconds for images, ~10-30 seconds for videos