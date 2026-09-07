import argparse
import os
import sys
import wave
import time
import numpy as np

# Try to import sounddevice; handle errors if no mic is available
try:
    import sounddevice as sd
except Exception as e:
    print(f"Error importing sounddevice (audio backend might be missing): {e}")
    sd = None

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def record_sample(filename, duration, sample_rate):
    if sd is None:
        print("sounddevice is not available. Simulating recording by creating empty/random audio.")
        audio_data = np.random.randn(int(duration * sample_rate)).astype(np.float32)
        save_wav(filename, audio_data, sample_rate)
        return True

    print(f"Recording for {duration} seconds... Please speak now.")
    try:
        # Record audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        
        # Display simple VU meter while recording
        for _ in range(int(duration * 10)):
            time.sleep(0.1)
            # Estimate volume from current recording slice (if we were streaming, this would be more accurate)
            # For simplicity in offline mode, just show a progress bar
            sys.stdout.write(".")
            sys.stdout.flush()
            
        sd.wait() # Wait until recording is finished
        print(" Done!")
        
        save_wav(filename, recording.flatten(), sample_rate)
        return True
    except Exception as e:
        print(f"Error during recording: {e}")
        return False

def save_wav(filename, audio_data, sample_rate):
    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    audio_data_int16 = np.int16(audio_data * 32767)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 2 bytes = 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data_int16.tobytes())

def main():
    parser = argparse.ArgumentParser(description="Record audio samples for KWS dataset.")
    parser.add_argument("--label", type=str, required=True, choices=config.CLASSES, help="Label for the recorded data")
    parser.add_argument("--speaker", type=str, required=True, help="Speaker name or ID")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to record")
    
    args = parser.parse_args()
    
    label_dir = os.path.join(config.RAW_DATA_DIR, args.label)
    os.makedirs(label_dir, exist_ok=True)
    
    print(f"Recording {args.samples} samples for label '{args.label}' from speaker '{args.speaker}'.")
    print("Press Ctrl+C to stop early.\n")
    
    successful_recordings = 0
    for i in range(args.samples):
        timestamp = int(time.time())
        filename = os.path.join(label_dir, f"{args.speaker}_{args.label}_{timestamp}_{i+1}.wav")
        
        print(f"\n--- Sample {i+1}/{args.samples} ---")
        time.sleep(1) # Short pause before recording
        
        if record_sample(filename, config.CLIP_DURATION, config.SAMPLE_RATE):
            print(f"Saved: {filename}")
            successful_recordings += 1
        else:
            print("Skipping sample due to error.")
            
    print(f"\nFinished! Successfully recorded {successful_recordings}/{args.samples} samples.")

if __name__ == "__main__":
    main()
