import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.train import load_dataset

def evaluate_model():
    model_path = os.path.join(config.MODELS_DIR, "kws_model.h5")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading test data...")
    X_test, y_test = load_dataset(os.path.join(config.DATA_DIR, "test_list.txt"))
    
    print("Evaluating model...")
    start_time = time.time()
    y_pred_prob = model.predict(X_test)
    end_time = time.time()
    
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    avg_inference_time = (end_time - start_time) / len(X_test)
    print(f"Average Inference Time per sample: {avg_inference_time*1000:.2f} ms")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=config.CLASSES))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(config.CLASSES))
    plt.xticks(tick_marks, config.CLASSES, rotation=45)
    plt.yticks(tick_marks, config.CLASSES)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODELS_DIR, 'confusion_matrix.png'))
    print(f"Saved confusion matrix plot to {os.path.join(config.MODELS_DIR, 'confusion_matrix.png')}")
    
    # Plot Evaluation Metrics Bar Chart
    report_dict = classification_report(y_test, y_pred, target_names=config.CLASSES, output_dict=True)
    
    classes = config.CLASSES
    precision = [report_dict[c]['precision'] * 100 for c in classes]
    recall = [report_dict[c]['recall'] * 100 for c in classes]
    f1 = [report_dict[c]['f1-score'] * 100 for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision', color='#3498db')
    plt.bar(x, recall, width, label='Recall', color='#2ecc71')
    plt.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    plt.ylabel('Percentage (%)')
    plt.title('EdgeWake Model Performance Metrics (Test Set)')
    plt.xticks(x, classes)
    plt.ylim(0, 110)
    plt.legend(loc='lower right')
    
    # Add value labels on top of bars
    for i in range(len(classes)):
        plt.text(i - width, precision[i] + 1, f'{precision[i]:.1f}', ha='center', va='bottom', fontsize=9)
        plt.text(i, recall[i] + 1, f'{recall[i]:.1f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width, f1[i] + 1, f'{f1[i]:.1f}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODELS_DIR, 'metrics_bar_chart.png'))
    print(f"Saved metrics bar chart to {os.path.join(config.MODELS_DIR, 'metrics_bar_chart.png')}")

if __name__ == "__main__":
    evaluate_model()
