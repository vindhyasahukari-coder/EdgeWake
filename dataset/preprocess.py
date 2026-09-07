import os
import sys
import glob
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset.augment import add_noise, time_shift, change_speed, change_pitch

def load_audio(filepath):
    audio, _ = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
    
    # Ensure exactly CLIP_DURATION (pad or truncate)
    if len(audio) < config.CLIP_SAMPLES:
        audio = np.pad(audio, (0, config.CLIP_SAMPLES - len(audio)), 'constant')
    elif len(audio) > config.CLIP_SAMPLES:
        audio = audio[:config.CLIP_SAMPLES]
        
    return audio

def save_processed(audio, label, prefix, idx):
    out_dir = os.path.join(config.PROCESSED_DATA_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_{idx}.wav")
    sf.write(out_path, audio, config.SAMPLE_RATE)

def main():
    print(f"Preprocessing data from: {config.RAW_DATA_DIR}")
    print(f"Saving processed data to: {config.PROCESSED_DATA_DIR}")
    
    for label in config.CLASSES:
        label_dir = os.path.join(config.RAW_DATA_DIR, label)
        if not os.path.exists(label_dir):
            print(f"Warning: Directory {label_dir} does not exist.")
            continue
            
        wav_files = glob.glob(os.path.join(label_dir, "*.wav"))
        print(f"Found {len(wav_files)} files for label '{label}'.")
        
        for idx, filepath in enumerate(tqdm(wav_files, desc=f"Processing {label}")):
            audio = load_audio(filepath)
            
            # Save original
            save_processed(audio, label, "orig", idx)
            
            # Save augmented versions
            # 1. Noise
            audio_noise = add_noise(audio)
            save_processed(audio_noise, label, "noise", idx)
            
            # 2. Shift
            audio_shift = time_shift(audio)
            save_processed(audio_shift, label, "shift", idx)
            
            # 3. Speed
            audio_speed = change_speed(audio)
            save_processed(audio_speed, label, "speed", idx)
            
            # 4. Pitch
            audio_pitch = change_pitch(audio)
            save_processed(audio_pitch, label, "pitch", idx)

if __name__ == "__main__":
    main()
