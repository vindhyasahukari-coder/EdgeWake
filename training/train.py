import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import librosa

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.feature_extraction import extract_features
from training.model import build_kws_model

def load_dataset(list_file):
    X = []
    y = []
    
    with open(list_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        filepath, label = line.strip().split('\t')
        
        # Load audio
        audio, _ = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
        # Feature extraction
        features = extract_features(audio)
        
        # Expand dims for Conv2D (channels=1)
        features = np.expand_dims(features, axis=-1)
        
        X.append(features)
        y.append(config.CLASSES.index(label))
        
    return np.array(X), np.array(y)

def train_model():
    print("Loading training data...")
    X_train, y_train = load_dataset(os.path.join(config.DATA_DIR, "train_list.txt"))
    print("Loading validation data...")
    X_val, y_val = load_dataset(os.path.join(config.DATA_DIR, "val_list.txt"))
    
    # y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(config.CLASSES))
    # y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(config.CLASSES))
    
    input_shape = X_train[0].shape
    print(f"Input shape: {input_shape}")
    
    model = build_kws_model(input_shape, len(config.CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=8, 
        min_lr=1e-5, 
        verbose=1
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config.MODELS_DIR, "kws_model.h5"),
        save_best_only=True,
        monitor='val_accuracy'
    )
    
    model.summary()
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, "kws_model.h5")
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )
    
    print(f"Model saved to {model_path}")
    return history

if __name__ == "__main__":
    train_model()
