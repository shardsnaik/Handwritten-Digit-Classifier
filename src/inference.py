"""
Inference script for digit classification.
Usage: python -m src.infer --images ./sample --weights ./models/best.pt --out preds.csv
"""
import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from pathlib import Path
from PIL import Image

from src.data_laoder import get_hog_features


def load_images(image_dir, img_size=64):
    """Load all images from directory."""
    image_paths = []
    images = []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for file_path in sorted(Path(image_dir).rglob('*')):
        if file_path.suffix.lower() in valid_extensions:
            try:
                img = Image.open(file_path).convert('L')  # Convert to grayscale
                img = img.resize((img_size, img_size))
                img_array = np.array(img)
                
                images.append(img_array)
                image_paths.append(str(file_path.relative_to(image_dir)))
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
    
    if not images:
        raise ValueError(f"No valid images found in {image_dir}")
    
    images = np.array(images)
    if len(images.shape) == 3:  # Add channel dimension if needed
        images = np.expand_dims(images, axis=-1)
    
    return images, image_paths


def predict_hog_model(model_path, images):
    """Predict using HOG + Logistic Regression model."""
    # Load model
    clf = joblib.load(model_path)
    
    # Extract HOG features
    X_hog = get_hog_features(images)
    
    # Predict
    predictions = clf.predict(X_hog)
    confidences = clf.predict_proba(X_hog).max(axis=1)
    
    return predictions, confidences


def predict_cnn_model(model_path, images):
    """Predict using CNN model."""
    # Load model
    model = keras.models.load_model(model_path)
    
    # Preprocess
    X = images.astype('float32') / 255.0
    
    # Predict
    probs = model.predict(X, verbose=0)
    predictions = np.argmax(probs, axis=1)
    confidences = probs.max(axis=1)
    
    return predictions, confidences


def main():
    parser = argparse.ArgumentParser(description='Run inference on digit images')
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--out', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    args = parser.parse_args()
    
    # Load images
    print(f"Loading images from {args.images}...")
    images, image_paths = load_images(args.images, img_size=args.img_size)
    print(f"Loaded {len(images)} images")
    
    # Detect model type and predict
    print(f"Running inference with {args.weights}...")
    
    if args.weights.endswith('.pkl'):
        predictions, confidences = predict_hog_model(args.weights, images)
        model_type = 'HOG + Logistic Regression'
    elif args.weights.endswith('.h5') or args.weights.endswith('.keras'):
        predictions, confidences = predict_cnn_model(args.weights, images)
        model_type = 'CNN'
    else:
        raise ValueError(f"Unsupported model format: {args.weights}")
    
    print(f"Model type: {model_type}")
    
    # Create output DataFrame
    results = pd.DataFrame({
        'image_path': image_paths,
        'predicted_digit': predictions,
        'confidence': confidences
    })
    
    # Save to CSV
    results.to_csv(args.out, index=False)
    print(f"\nPredictions saved to {args.out}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(results['predicted_digit'].value_counts().sort_index())
    print(f"\nMean confidence: {confidences.mean():.4f}")
    print(f"Min confidence: {confidences.min():.4f}")
    print(f"Max confidence: {confidences.max():.4f}")


if __name__ == '__main__':
    main()