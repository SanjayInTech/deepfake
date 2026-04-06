import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Setup Paths (Now using the HIGH-ACCURACY CROPPED DATASET)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'cropped_images')

# Advanced Hyperparameters
IMG_SIZE = (299, 299) # Xception standard
BATCH_SIZE = 16
PHASE_1_EPOCHS = 15
PHASE_2_EPOCHS = 15

def train_model():
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"⚠️ Error: Cropped dataset directory {DATASET_DIR} does not exist.")
        print("Please run 'python preprocess_dataset.py' first!")
        return

    print("🚀 Preparing Advanced Data Generators...")
    
    # Aggressive Data Augmentation & Normalization
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', subset='validation'
    )

    print("🧠 Building the Top-Level Model (Xception Network)...")
    base_model = Xception(input_shape=(299, 299, 3), include_top=False, weights='imagenet')
    
    # Freeze base model for Phase 1
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    # 0 for Fake, 1 for Real
    predictions = Dense(1, activation='sigmoid')(x) 

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    print("🔥 PHASE 1: Training Top Layers Only...")
    checkpoint = ModelCheckpoint('deepfake_xception_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    
    model.fit(
        train_generator,
        epochs=PHASE_1_EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint]
    )

    print("🔬 PHASE 2: Fine-Tuning the Base Model...")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Re-freeze the bottom 100 layers, leaving the top layers to fine-tune to deepfakes
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    # Re-compile with an extremely low learning rate!
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=PHASE_2_EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint]
    )

    # Save the State-of-the-Art Model
    model.save('deepfake_xception_model.h5')
    print("🎯 Top-Level Training Complete! Model saved as 'deepfake_xception_model.h5'")

if __name__ == "__main__":
    train_model()
