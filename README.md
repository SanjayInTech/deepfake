# Deepfake Detection System

A web-based deepfake detection application using Xception CNN with MTCNN face detection. Analyze images and videos to detect AI-generated content.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)

## Features

- **Image Detection** - Upload JPG, PNG images for instant analysis
- **Video Detection** - Support for MP4, AVI, MOV, MKV with multi-frame analysis
- **Face Detection** - Automatic MTCNN-based face detection and cropping
- **Real-time Prediction** - Confidence scores for each prediction
- **Glassmorphism UI** - Modern, user-friendly interface with drag-and-drop

## Tech Stack

| Category | Technology |
|----------|------------|
| Backend | Flask 3.0.3 |
| Deep Learning | TensorFlow 2.21 |
| Computer Vision | OpenCV 4.10, MTCNN |
| Frontend | HTML/CSS/JavaScript |

## Getting Started

### Prerequisites

- Python 3.8+
- Kaggle account (for dataset download)

### Installation

```bash
# Clone the repository
git clone https://github.com/SanjayInTech/deepfake.git
cd deepfake

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Place your kaggle.json in ~/.kaggle/
# Then run:
python download_datasets.py
```

### Preprocess Data

```bash
python preprocess_dataset.py
```

### Train Model

```bash
python train_model.py
```

### Run Application

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## How It Works

1. **Upload** - User uploads image or video via web interface
2. **Detect** - MTCNN detects and crops the face
3. **Preprocess** - Face resized to 299x299 (Xception input size)
4. **Predict** - Model analyzes and outputs probability
5. **Result** - Displays "Real" or "Fake" with confidence %

## Project Structure

```
deepfake/
├── app.py                    # Flask backend
├── train_model.py            # Model training script
├── download_datasets.py       # Dataset downloader
├── preprocess_dataset.py     # Data preprocessing
├── templates/
│   └── index.html           # Frontend UI
├── dataset/                 # Dataset storage (download separately)
└── uploads/                 # Temporary upload folder
```

## Model Details

- **Architecture**: Xception (pretrained on ImageNet)
- **Training**: Transfer Learning with two-phase training
- **Dataset**: FaceForensics++, Celeb-DF, DFDC datasets

## License

MIT License
