"""
Analyze model performance and find misclassified samples.
Usage: python -m src.analyze --data ./dataset --weights ./models/improved_cnn_best.h5
"""
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import joblib

from src.data_laoder import load_data, get_hog_features


def plot_confusion_matrix(cm, title, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def find_misclassifications(X, y, y_pred, y_probs, num_samples=5):
    """Find and analyze misclassified samples."""
    # Find misclassified indices
    misclassified_idx = np.where(y != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassifications found!")
        return []
    
    # Get confidence scores for misclassified samples
    confidences = y_probs[misclassified_idx].max(axis=1)
    
    # Sort by confidence (high confidence mistakes are more interesting)
    sorted_idx = np.argsort(confidences)[::-1]
    top_idx = misclassified_idx[sorted_idx[:num_samples]]
    
    misclassifications = []
    for idx in top_idx:
        misclassifications.append({
            'index': int(idx),
            'true_label': int(y[idx]),
            'predicted_label': int(y_pred[idx]),
            'confidence': float(y_probs[idx].max()),
            'image': X[idx]
        })
    
    return misclassifications


def plot_misclassifications(misclassifications, save_path):
    """Plot misclassified samples."""
    n = len(misclassifications)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    
    if n == 1:
        axes = [axes]
    
    for i, mis in enumerate(misclassifications):
        img = mis['image'].squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(
            f"True: {mis['true_label']}\n"
            f"Pred: {mis['predicted_label']}\n"
            f"Conf: {mis['confidence']:.3f}",
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Misclassifications plot saved to {save_path}")


def analyze_model(model_path, X_val, y_val, X_test, y_test, output_dir, model_name):
    """Analyze model performance."""
    print(f"\n=== Analyzing {model_name} ===")
    
    # Load model and predict
    if model_path.endswith('.pkl'):
        # HOG + Logistic Regression
        clf = joblib.load(model_path)
        X_val_hog = get_hog_features(X_val)
        X_test_hog = get_hog_features(X_test)
        
        val_pred = clf.predict(X_val_hog)
        test_pred = clf.predict(X_test_hog)
        val_probs = clf.predict_proba(X_val_hog)
        test_probs = clf.predict_proba(X_test_hog)
    else:
        # CNN model
        model = keras.models.load_model(model_path)
        X_val_norm = X_val.astype('float32') / 255.0
        X_test_norm = X_test.astype('float32') / 255.0
        
        val_probs = model.predict(X_val_norm, verbose=0)
        test_probs = model.predict(X_test_norm, verbose=0)
        val_pred = np.argmax(val_probs, axis=1)
        test_pred = np.argmax(test_probs, axis=1)
    
    # Calculate metrics
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    
    val_cm = confusion_matrix(y_val, val_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    
    print(f"Validation - Accuracy: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")
    print(f"Test       - Accuracy: {test_acc:.4f}, Macro-F1: {test_f1:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrix(val_cm, f"{model_name} - Validation", 
                         os.path.join(output_dir, f'{model_name}_val_cm.png'))
    plot_confusion_matrix(test_cm, f"{model_name} - Test", 
                         os.path.join(output_dir, f'{model_name}_test_cm.png'))
    
    # Find and plot misclassifications
    print("\nFinding misclassifications...")
    test_misclass = find_misclassifications(X_test, y_test, test_pred, test_probs, num_samples=5)
    
    if test_misclass:
        plot_misclassifications(test_misclass, 
                               os.path.join(output_dir, f'{model_name}_misclass.png'))
        
        # Save misclassifications to JSON (without images)
        misclass_report = []
        for mis in test_misclass:
            misclass_report.append({
                'true_label': mis['true_label'],
                'predicted_label': mis['predicted_label'],
                'confidence': mis['confidence']
            })
        
        with open(os.path.join(output_dir, f'{model_name}_misclass.json'), 'w') as f:
            json.dump(misclass_report, f, indent=2)
    
    return {
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'misclassifications': test_misclass
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze model performance')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = './analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        args.data, 
        img_size=args.img_size,
        seed=42
    )
    
    # Determine model name from path
    model_name = os.path.splitext(os.path.basename(args.weights))[0]
    
    # Analyze model
    results = analyze_model(args.weights, X_val, y_val, X_test, y_test, 
                           output_dir, model_name)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to {output_dir}/")


if __name__ == '__main__':
    main()