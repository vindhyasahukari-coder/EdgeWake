import os
import sys
import numpy as np
import tensorflow as tf
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.feature_extraction import extract_features

class KWSModel:
    def __init__(self, model_path=None):
        if model_path is None:
            # We use float32 for PC testing, INT8 for ESP32
            model_path = os.path.join(config.MODELS_DIR, "kws_model_float32.tflite")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Did you run Phase 1?")
            
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Check if the model is quantized
        self.is_quantized = self.input_details[0]['dtype'] == np.int8

    def predict(self, audio_data):
        """
        Runs inference on a 1-second audio clip.
        Returns the index of the predicted class and the confidence score.
        """
        start_time = time.time()
        
        # 1. Extract MFCC features
        features = extract_features(audio_data)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=-1) # Add channel dimension
        
        # 2. Quantize input if necessary
        if self.is_quantized:
            scale, zero_point = self.input_details[0]['quantization']
            features = features / scale + zero_point
            features = features.astype(np.int8)
        else:
            features = features.astype(np.float32)
            
        # 3. Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        
        # 4. Dequantize output if necessary
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        if self.is_quantized:
            scale, zero_point = self.output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale
            
        inference_time = (time.time() - start_time) * 1000 # ms
            
        # 5. Return max prediction
        pred_idx = np.argmax(output)
        confidence = output[pred_idx]
        
        return pred_idx, confidence, inference_time
