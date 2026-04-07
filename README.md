# Deepfake Detection System

A web-based deepfake detection application using Xception CNN with MTCNN face detection. Analyze images and videos to detect AI-generated content with high accuracy.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Download](#dataset-download)
  - [Using the Script](#using-the-script)
  - [Manual Download (If Script Fails)](#manual-download-if-script-fails)
- [Preprocessing Data](#preprocessing-data)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Using the Web Interface](#using-the-web-interface)
- [Testing](#testing)
- [Model Details](#model-details)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a deepfake detection system that can analyze images and videos to determine whether the content is real or fake (AI-generated/manipulated). The system uses:

- **Xception CNN** pretrained on ImageNet, fine-tuned on deepfake datasets
- **MTCNN** for automatic face detection and cropping
- **Flask** backend with a modern glassmorphism web interface

The model achieves high accuracy through transfer learning with two-phase training:
1. **Phase 1**: Train only the top layers (frozen base)
2. **Phase 2**: Fine-tune the top layers of the base model

---

## Features

- **Image Detection** - Upload JPG, PNG images for instant analysis
- **Video Detection** - Support for MP4, AVI, MOV, MKV with multi-frame analysis
- **Face Detection** - Automatic MTCNN-based face detection and cropping
- **Real-time Prediction** - Confidence scores for each prediction
- **Glassmorphism UI** - Modern, user-friendly interface with drag-and-drop support
- **Multi-frame Analysis** - Videos are analyzed across multiple frames for better accuracy

---

## Tech Stack

| Category | Technology | Version |
|----------|------------|---------|
| Backend | Flask | 3.0.3 |
| Deep Learning | TensorFlow | 2.21.0 |
| Computer Vision | OpenCV | 4.10.0.84 |
| Face Detection | MTCNN | 1.0.0 |
| Image Processing | Pillow | 10.4.0 |
| Array Processing | NumPy | 2.1.1 |
| Frontend | HTML/CSS/JavaScript | - |

---

## Project Structure

```
deepfake/
├── app.py                        # Flask backend application
├── train_model.py                # Model training script
├── download_datasets.py          # Automated dataset downloader
├── preprocess_dataset.py         # Face extraction & data preprocessing
├── create_mock_model.py          # Creates a placeholder model for testing
├── extract_frames.py             # Video frame extraction utility
├── organize_datasets.py          # Dataset organization utility
├── deepfake_xception_model.h5    # Trained model (generated after training)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── templates/
│   └── index.html              # Frontend UI
├── dataset/
│   ├── raw/                     # Raw downloaded videos
│   └── cropped_images/          # Preprocessed face images (after preprocessing)
│       ├── real/                # Real face images
│       └── fake/                # Fake face images
├── uploads/                     # Temporary uploaded files
└── test_samples/               # Test files for verification
```

---

## Installation

### Option 1: Download as ZIP (Recommended if Git is not installed)

If Git is not installed on your system, you can download the project as a ZIP file:

1. **Go to the GitHub repository**
   ```
   https://github.com/SanjayInTech/deepfake
   ```

2. **Click the green "Code" button** (top right of the repository)

3. **Click "Download ZIP"**
   ```
   deepfake-main.zip
   ```

4. **Extract the ZIP file**
   - Right-click the downloaded ZIP
   - Select "Extract All" or "Extract Here"
   - Copy the extracted folder to your desired location

5. **Open the extracted folder**
   ```
   deepfake-main/
   ```

6. **Proceed to Installation step below**

---

### Option 2: Clone with Git (If Git is installed)

```bash
git clone https://github.com/SanjayInTech/deepfake.git
cd deepfake
```

---

```bash
git clone https://github.com/SanjayInTech/deepfake.git
cd deepfake
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** TensorFlow installation may take 5-10 minutes. Ensure you have ~5GB free space.

---

## Dataset Download

The project uses three deepfake datasets from Kaggle:
- **FaceForensics++** - Standard benchmark dataset
- **Celeb-DF** - High quality fake videos
- **DFDC** - Facebook/Meta AI Challenge dataset

### Using the Script

**Step 1:** Get your Kaggle API credentials

1. **Create a Kaggle account** (if you don't have one)
   - Go to [Kaggle.com](https://www.kaggle.com)
   - Click "Sign Up" and register with email or Google account

2. **Enable Kaggle API for your account**
   - Go to your account settings: Click your profile picture → "Settings"
   - Scroll down to "API" section
   - Click "Create New Token" button

3. **Download kaggle.json**
   - This will download a file named `kaggle.json`
   - This file contains your username and API token
   - Keep this file secure - it gives access to your Kaggle downloads

**Step 2:** Configure Kaggle credentials

```bash
# Windows
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\

# macOS/Linux
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Step 3:** Run the download script

```bash
python download_datasets.py
```

This will download all three datasets into `dataset/raw/` folder.

---

### Manual Download (If Script Fails)

If the automated script fails, download manually:

1. **FaceForensics++**
   - Go to: https://www.kaggle.com/datasets/shahrik123/faceforensics-original
   - Click "Download"
   - Extract to `dataset/raw/faceforensics/`

2. **Celeb-DF**
   - Go to: https://www.kaggle.com/datasets/sdESDLH17AI/FaceForensics
   - Click "Download"
   - Extract to `dataset/raw/celeb_df/`

3. **DFDC**
   - Go to: https://www.kaggle.com/datasets/pranay22077/dfdc-10
   - Click "Download"
   - Extract to `dataset/raw/dfdc/`

**Expected folder structure after download:**
```
dataset/raw/
├── faceforensics/
│   └── original_sequences_youtube/
│       └── c23/
│           └── videos/  (contains .mp4 files)
├── celeb_df/
│   └── videos/          (contains .mp4 files)
└── dfdc/
    └── videos/          (contains .mp4 files)
```

---

## Preprocessing Data

After downloading datasets, you need to extract faces and prepare them for training.

```bash
python preprocess_dataset.py
```

**What this script does:**
1. Reads all videos from `dataset/raw/`
2. Extracts frames from each video
3. Uses MTCNN to detect and crop faces
4. Saves cropped faces to `dataset/cropped_images/real/` and `dataset/cropped_images/fake/`
5. Organizes images for binary classification

**Output:**
```
dataset/cropped_images/
├── real/     # ~10,000+ face images from real videos
└── fake/     # ~10,000+ face images from fake videos
```

**Time:** This process takes 1-4 hours depending on dataset size.

---

## Training the Model

Once preprocessing is complete, train the model:

```bash
python train_model.py
```

**What this script does:**

1. **Phase 1 (15 epochs)** - Train top layers only
   - Loads Xception pretrained on ImageNet
   - Freezes base model layers
   - Trains custom classification head
   - Learning rate: 0.001

2. **Phase 2 (15 epochs)** - Fine-tune the model
   - Unfreezes top 100 layers of Xception
   - Fine-tunes with very low learning rate (1e-5)
   - Improves accuracy significantly

**Output:**
```
deepfake_xception_model.h5  # ~166MB trained model file
```

**Training Time:**
- GPU: 30-60 minutes
- CPU: 4-8 hours

**Hardware Requirements:**
- Minimum: 8GB RAM, 4GB VRAM
- Recommended: 16GB RAM, 8GB+ VRAM

---

## Running the Application

### Start the Flask Server

```bash
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5000
* Press CTRL+C to quit
```

### Access the Web Interface

Open your browser and go to:
```
http://127.0.0.1:5000
```

---

## How It Works

### Detection Pipeline

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│   Upload    │ ───► │  MTCNN Face  │ ───► │  Preprocessing │
│   File      │      │   Detector   │      │  (299x299)     │
└─────────────┘      └──────────────┘      └────────────────┘
                                                  │
                                                  ▼
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│   Display   │ ◄─── │   Prediction │ ◄─── │ Xception Model │
│   Result    │      │   & Score    │      │   Analysis     │
└─────────────┘      └──────────────┘      └────────────────┘
```

### Step-by-Step Process

1. **Upload** - User uploads image or video via web interface
2. **Face Detection** - MTCNN detects and locates faces in the media
3. **Cropping** - Face region is extracted and cropped
4. **Preprocessing** - Face resized to 299x299 pixels (Xception input size)
5. **Normalization** - Pixel values normalized to [0, 1]
6. **Prediction** - Model outputs probability (0 = Fake, 1 = Real)
7. **Result Display** - Shows prediction with confidence percentage

### For Videos

1. Video is loaded and frames are extracted
2. Multiple frames (5-10) are sampled evenly throughout the video
3. Each frame goes through the detection pipeline
4. Predictions are averaged for final result
5. Result displays average confidence across all frames

---

## Using the Web Interface

### Upload an Image

1. Click the upload area or drag and drop an image
2. Supported formats: JPG, PNG, JPEG
3. Wait for processing (1-3 seconds)
4. View result with confidence score

### Upload a Video

1. Click the upload area or drag and drop a video
2. Supported formats: MP4, AVI, MOV, MKV
3. Wait for processing (10-30 seconds depending on length)
4. View result with average confidence score

### Understanding Results

- **Confidence 0-40%** - Likely FAKE (high confidence it's manipulated)
- **Confidence 40-60%** - Uncertain (borderline case)
- **Confidence 60-100%** - Likely REAL (high confidence it's authentic)

---

## Testing

The project includes test samples in `test_samples/` folder:

```
test_samples/
├── real_images/    # 10 real images
├── fake_images/    # 10 fake images
├── real_videos/    # 10 real videos
└── fake_videos/    # 10 fake videos
```

To test the application:
1. Start the app: `python app.py`
2. Upload files from `test_samples/`
3. Verify predictions match expected labels

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Xception |
| Base Model | ImageNet (pretrained) |
| Input Size | 299 x 299 x 3 |
| Output | Sigmoid (0-1 probability) |
| Training Method | Transfer Learning (2-phase) |
| Data Augmentation | Rotation, shift, zoom, flip |
| Validation Split | 20% |
| Batch Size | 16 |
| Total Epochs | 30 (15 + 15) |

### Class Labels

| Label | Meaning |
|-------|---------|
| 0 | Fake (AI-generated/manipulated) |
| 1 | Real (authentic/original) |

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Kaggle credentials not found"

**Solution:**
1. Download `kaggle.json` from Kaggle account settings
2. Place in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows)
3. Ensure file permissions are 600

### Issue: "CUDA/GPU not available"

The model will run on CPU. Training will be slower but still works. For GPU support:
1. Install NVIDIA drivers
2. Install CUDA Toolkit 11.8+
3. Install cuDNN 8.6+
4. Reinstall TensorFlow: `pip install tensorflow[and-cuda]`

### Issue: "Out of memory" during training

**Solution:**
Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 8  # instead of 16
```

### Issue: "No face detected" in uploaded media

**Solution:**
- Ensure the face is clearly visible
- Use high-quality images/videos
- Try a different image/video

### Issue: "Model file not found" when running app

**Solution:**
Train the model first:
```bash
python train_model.py
```

Or create a placeholder model for testing:
```bash
python create_mock_model.py
```

---

## Development Timeline

| Date | Activity |
|------|----------|
| 26/03 | Researched deepfake detection techniques |
| 27/03 | Set up development environment |
| 28/03 | Downloaded deepfake dataset |
| 29/03 | Created preprocessing pipeline |
| 30/03 | Implemented MTCNN face detection |
| 31/03 | Organized dataset structure |
| 01/04 | Created Xception training model |
| 02/04 | Trained model with data augmentation |
| 03/04 | Created Flask backend API |
| 04/04 | Designed glassmorphism frontend |
| 05/04 | Integrated frontend with backend |
| 06/04 | End-to-end testing completed |

---

## License

MIT License - Feel free to use, modify, and distribute.

---

## Author

**SanjayInTech** - [GitHub Profile](https://github.com/SanjayInTech)

---

## Acknowledgments

- Xception model architecture from Keras/TensorFlow
- MTCNN face detection by FaceNet
- Deepfake datasets from FaceForensics++, Celeb-DF, and DFDC
