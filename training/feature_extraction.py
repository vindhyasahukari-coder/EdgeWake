import numpy as np
import librosa
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def extract_features(audio_data, sr=None):
    """
    Extracts MFCC features from raw audio data.
    This exact function must be used for training, evaluation, and real-time inference.
    """
    if sr is None:
        sr = config.SAMPLE_RATE
        
    # Ensure float32
    audio_float = audio_data.astype(np.float32)
    
    # Extract MFCCs
    # frame_length 640 = 40ms, hop_length 320 = 20ms
    mfccs = librosa.feature.mfcc(
        y=audio_float,
        sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.FRAME_LENGTH,
        hop_length=config.FRAME_STEP
    )
    
    # Output shape from librosa is (n_mfcc, n_frames). We transpose it to (n_frames, n_mfcc)
    mfccs = mfccs.T
    
    # Normalize features per sample to prevent gradient explosion
    mean = np.mean(mfccs)
    std = np.std(mfccs)
    if std > 0:
        mfccs = (mfccs - mean) / (std + 1e-6)
        
    return mfccs
