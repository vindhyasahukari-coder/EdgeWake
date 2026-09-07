import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.train import load_dataset

def representative_dataset_generator():
    """Generator function for INT8 quantization."""
    # Load a small subset of training data for calibration
    print("Loading representative dataset for quantization...")
    X_train, _ = load_dataset(os.path.join(config.DATA_DIR, "train_list.txt"))
    
    # Use max 100 samples
    num_samples = min(100, len(X_train))
    indices = np.random.choice(len(X_train), num_samples, replace=False)
    
    for i in indices:
        # Expand dims to include batch size
        sample = np.expand_dims(X_train[i], axis=0).astype(np.float32)
        yield [sample]

def export_models():
    model_path = os.path.join(config.MODELS_DIR, "kws_model.h5")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    print("Loading Keras model...")
    model = tf.keras.models.load_model(model_path)
    
    # 1. Float32 TFLite Model
    print("\nConverting to Float32 TFLite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float32 = converter.convert()
    float32_path = os.path.join(config.MODELS_DIR, "kws_model_float32.tflite")
    with open(float32_path, "wb") as f:
        f.write(tflite_float32)
    print(f"Saved: {float32_path} (Size: {len(tflite_float32) / 1024:.2f} KB)")
    
    # 2. Dynamic Range Quantization
    print("\nConverting to Dynamic Range TFLite model...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter.convert()
    dynamic_path = os.path.join(config.MODELS_DIR, "kws_model_dynamic.tflite")
    with open(dynamic_path, "wb") as f:
        f.write(tflite_dynamic)
    print(f"Saved: {dynamic_path} (Size: {len(tflite_dynamic) / 1024:.2f} KB)")
    
    # 3. Full INT8 Quantization
    print("\nConverting to Full INT8 TFLite model...")
    converter.representative_dataset = representative_dataset_generator
    # Ensure all ops are quantized
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_int8 = converter.convert()
        int8_path = os.path.join(config.MODELS_DIR, "kws_model_int8.tflite")
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        print(f"Saved: {int8_path} (Size: {len(tflite_int8) / 1024:.2f} KB)")
    except Exception as e:
        print(f"Failed to generate INT8 model: {e}")

if __name__ == "__main__":
    export_models()
