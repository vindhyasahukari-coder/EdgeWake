import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class WakeWordDetector:
    def __init__(self):
        self.negative_idx = config.CLASSES.index("negative")
        self.wake_word_indices = [i for i, label in enumerate(config.CLASSES) if label != "negative"]
        self.consecutive_count = 0
        self.last_trigger_time = 0
        self.last_detected_word = None

    def process_prediction(self, pred_idx, confidence):
        """
        Takes the raw prediction and applies smoothing logic.
        Returns the wake word if triggered, False otherwise.
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_trigger_time < config.COOLDOWN_SECONDS:
            self.consecutive_count = 0
            return False
            
        is_keyword = pred_idx in self.wake_word_indices
        
        if is_keyword and confidence >= config.WAKE_WORD_THRESHOLD:
            self.consecutive_count += 1
            self.last_detected_word = config.CLASSES[pred_idx]
        else:
            # Reset on any frame that drops below threshold or is not keyword
            self.consecutive_count = 0
            self.last_detected_word = None
            
        if self.consecutive_count >= config.CONSECUTIVE_DETECTIONS:
            # Trigger!
            self.last_trigger_time = current_time
            self.consecutive_count = 0
            return self.last_detected_word
            
        return False
