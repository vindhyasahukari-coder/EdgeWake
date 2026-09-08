import os

# Project configuration

# Target Wake Words
WAKE_WORDS = ["SixSync", "Lights", "Fan"]
CLASSES = ["SixSync", "Lights", "Fan", "negative"]

# Audio config
SAMPLE_RATE = 16000
CLIP_DURATION = 1.0 # seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# Feature extraction config
N_MFCC = 20
FRAME_LENGTH = 640  # 40ms at 16kHz
FRAME_STEP = 320    # 20ms at 16kHz

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TEST_SAMPLES_DIR = os.path.join(DATA_DIR, "test_samples")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODEL_DIR = os.path.join(MODELS_DIR, "saved_model")

# Augmentation config
AUGMENTATION = {
    "noise_factor": 0.005,
    "time_shift_max": 0.1, # Max shift 10%
    "pitch_shift_steps": 2, # Number of half-steps
    "speed_variation": [0.9, 1.1]
}

# Training config
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Real-time config
CHUNK_SIZE = 1600
WINDOW_DURATION = 1.0
INFERENCE_INTERVAL = 0.2
WAKE_WORD_THRESHOLD = 0.70
CONSECUTIVE_DETECTIONS = 2
COOLDOWN_SECONDS = 3

MAX_COMMAND_DURATION = 10
SILENCE_TIMEOUT = 0.8
