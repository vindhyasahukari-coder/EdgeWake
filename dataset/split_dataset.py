import os
import sys
import glob
import random
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def split_data():
    print("Splitting dataset into train/val/test...")
    
    # We'll just manage lists of files for the tf.data pipeline to load,
    # or actually move them into split directories. Let's create a list file.
    
    train_files = []
    val_files = []
    test_files = []
    
    for label in config.CLASSES:
        label_dir = os.path.join(config.PROCESSED_DATA_DIR, label)
        if not os.path.exists(label_dir):
            continue
            
        wav_files = glob.glob(os.path.join(label_dir, "*.wav"))
        random.shuffle(wav_files)
        
        n_total = len(wav_files)
        n_val = int(n_total * config.VALIDATION_SPLIT)
        n_test = int(n_total * 0.1) # 10% for testing
        n_train = n_total - n_val - n_test
        
        train_files.extend([(f, label) for f in wav_files[:n_train]])
        val_files.extend([(f, label) for f in wav_files[n_train:n_train+n_val]])
        test_files.extend([(f, label) for f in wav_files[n_train+n_val:]])
        
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Write paths to text files
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(os.path.join(config.DATA_DIR, "train_list.txt"), "w") as f:
        for p, l in train_files: f.write(f"{p}\t{l}\n")
    with open(os.path.join(config.DATA_DIR, "val_list.txt"), "w") as f:
        for p, l in val_files: f.write(f"{p}\t{l}\n")
    with open(os.path.join(config.DATA_DIR, "test_list.txt"), "w") as f:
        for p, l in test_files: f.write(f"{p}\t{l}\n")

if __name__ == "__main__":
    split_data()
