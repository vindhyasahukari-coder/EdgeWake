import numpy as np
import librosa
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def add_noise(audio_data, noise_factor=None):
    if noise_factor is None:
        noise_factor = config.AUGMENTATION["noise_factor"]
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + noise_factor * noise
    # Cast back to same type
    return augmented_data.astype(type(audio_data[0]))

def time_shift(audio_data, max_shift=None):
    if max_shift is None:
        max_shift = config.AUGMENTATION["time_shift_max"]
    
    shift_amt = int(np.random.uniform(-max_shift, max_shift) * len(audio_data))
    if shift_amt == 0:
        return audio_data
        
    augmented_data = np.roll(audio_data, shift_amt)
    # Zero pad the wrapped around part
    if shift_amt > 0:
        augmented_data[:shift_amt] = 0
    else:
        augmented_data[shift_amt:] = 0
        
    return augmented_data

def change_speed(audio_data, speed_range=None, sr=None):
    if speed_range is None:
        speed_range = config.AUGMENTATION["speed_variation"]
    if sr is None:
        sr = config.SAMPLE_RATE
        
    speed_rate = np.random.uniform(speed_range[0], speed_range[1])
    
    # librosa.effects.time_stretch requires float
    audio_float = audio_data.astype(np.float32)
    augmented_data = librosa.effects.time_stretch(audio_float, rate=speed_rate)
    
    # Needs to be same length
    if len(augmented_data) > len(audio_data):
        augmented_data = augmented_data[:len(audio_data)]
    else:
        augmented_data = np.pad(augmented_data, (0, len(audio_data) - len(augmented_data)), 'constant')
        
    return augmented_data

def change_pitch(audio_data, pitch_steps=None, sr=None):
    if pitch_steps is None:
        pitch_steps = config.AUGMENTATION["pitch_shift_steps"]
    if sr is None:
        sr = config.SAMPLE_RATE
        
    n_steps = np.random.uniform(-pitch_steps, pitch_steps)
    
    audio_float = audio_data.astype(np.float32)
    augmented_data = librosa.effects.pitch_shift(audio_float, sr=sr, n_steps=n_steps)
    return augmented_data
