import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def create_dummy_model():
    print("🛠️ Creating Mock Xception Model for UI testing...")
    base_model = Xception(input_shape=(299, 299, 3), include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x) 
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.save('deepfake_xception_model.h5')
    print("✅ Mock model saved as 'deepfake_xception_model.h5'")

if __name__ == "__main__":
    create_dummy_model()
