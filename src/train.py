"""
Training script for digit classification models.
Usage: python -m src.train --data ./dataset --epochs 15 --img_size 64
"""
import argparse
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
from pathlib import Path

from src.data_laoder import load_data, get_hog_features
from src.models import create_baseline_cnn, create_improved_cnn


def train_hog_baseline(X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    """ Logistic Regression baseline."""
    print("\n=== Training HOG + Logistic Regression Baseline ===")
    
    # Extract HOG features
    print("Extracting HOG features...")
    X_train_hog = get_hog_features(X_train)
    X_val_hog = get_hog_features(X_val)
    X_test_hog = get_hog_features(X_test)
    
    # Train logistic regression
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
    clf.fit(X_train_hog, y_train)
    
    # Evaluate
    val_pred = clf.predict(X_val_hog)
    test_pred = clf.predict(X_test_hog)
    
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    
    print(f"Validation - Accuracy: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")
    print(f"Test - Accuracy: {test_acc:.4f}, Macro-F1: {test_f1:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, 'hog_logreg.pkl')
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics = {
        'model': 'HOG + Logistic Regression',
        'val_accuracy': float(val_acc),
        'val_f1': float(val_f1),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'val_confusion_matrix': confusion_matrix(y_val, val_pred).tolist(),
        'test_confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
    }
    
    with open(os.path.join(output_dir, 'hog_logreg_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def train_cnn_model(model, model_name, X_train, y_train, X_val, y_val, 
                    X_test, y_test, epochs, output_dir):
    """Train CNN model."""
    print(f"\n=== Training {model_name} ===")
    
    # Prepare data
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Data augmentation for training only
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # brightness_range=[0.8, 1.2],
        # fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_best.h5')
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    val_f1 = f1_score(y_val, val_pred, average='macro')
    test_f1 = f1_score(y_test, test_pred, average='macro')
    
    print(f"Validation - Accuracy: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")
    print(f"Test - Accuracy: {test_acc:.4f}, Macro-F1: {test_f1:.4f}")
    
    # Save metrics
    metrics = {
        'model': model_name,
        'val_accuracy': float(val_acc),
        'val_f1': float(val_f1),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'val_confusion_matrix': confusion_matrix(y_val, val_pred).tolist(),
        'test_confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
    }
    
    with open(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def run_main():
    parser = argparse.ArgumentParser(description='Train digit classification models')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--models', type=str, default='all', 
                       help='Models to train: all, hog, baseline_cnn, improved_cnn')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = './models'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        args.data, 
        img_size=args.img_size,
        seed=42
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(np.unique(y_train))
    
    all_metrics = {}
    
    # Train HOG baseline
    if args.models in ['all', 'hog']:
        metrics = train_hog_baseline(X_train, y_train, X_val, y_val, X_test, y_test, output_dir)
        all_metrics['hog_logreg'] = metrics

    
    # Train baseline CNN
    if args.models in ['all', 'baseline_cnn']:
        model = create_baseline_cnn(input_shape=(args.img_size, args.img_size, 1))
        metrics = train_cnn_model(
            model, 'Baseline CNN', 
            X_train, y_train, X_val, y_val, X_test, y_test,
            args.epochs, output_dir
        )
        all_metrics['baseline_cnn'] = metrics
    
    # Train improved CNN
    if args.models in ['all', 'improved_cnn']:
        model = create_improved_cnn(input_shape=(args.img_size, args.img_size, 1))
        metrics = train_cnn_model(
            model, 'Improved CNN',
            X_train, y_train, X_val, y_val, X_test, y_test,
            args.epochs, output_dir
        )
        all_metrics['improved_cnn'] = metrics
    
    # Save all metrics
    # if os.path.exists('output_dir', )
    with open(os.path.join(output_dir, 'all_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for model_name, metrics in all_metrics.items():
        print(f"\n{metrics['model']}:")
        print(f"  Val  - Acc: {metrics['val_accuracy']:.4f}, F1: {metrics['val_f1']:.4f}")
        print(f"  Test - Acc: {metrics['test_accuracy']:.4f}, F1: {metrics['test_f1']:.4f}")
    
    print("\nTraining complete! Models saved to ./models/")


if __name__ == '__main__':
    run_main()