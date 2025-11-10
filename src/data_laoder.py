"""
Data loading and preprocessing utilities.
"""
import os, argparse
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from tqdm import tqdm


def load_data(data_dir, img_size=64, seed=42):
    """
    Load digit images from directory structure and create stratified splits.
    
    Args:
        data_dir: Path to dataset directory with structure dataset/0/, dataset/1/, ...
        img_size: Target image size (default: 64)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    images = []
    labels = []
    
    # Load images from each digit folder
    for digit in range(10):
        digit_dir = os.path.join(data_dir, str(digit))
        if not os.path.exists(digit_dir):
            raise ValueError(f"Directory not found: {digit_dir}")
        
        image_files = list(Path(digit_dir).glob('*'))
        print(f"Loading digit {digit}: {len(image_files)} images")
        
        for img in tqdm(image_files, desc=f"Digit {digit}"):
            try:
                # Load and preprocess image
                img = Image.open(img)
                if img.mode != 'L':
                    # Convert to grayscale
                    img= img.convert('L')
                img = img.resize((img_size, img_size))
                img_array = np.array(img)
                
                images.append(img_array)
                labels.append(digit)
            except Exception as e:
                print(f"Warning: Could not load {img}: {e}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Add channel dimension for CNN compatibility
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
    
    print(f"\nTotal images loaded: {len(X)}")
    print(f"Image shape: {X.shape}")
    
    # Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    
    # Debugging for low accur
    for digit in range(10):
        print(digit, np.count_nonzero(y_train == digit))

    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Verify stratification
    print(f"\nClass distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)

    print(y_train[:10])

    for digit, count in zip(unique, counts):
        print(f"  {digit}: {count}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_hog_features(images):
    """
    Extract HOG (Histogram of Oriented Gradients) features from images.
    
    Args:
        images: Numpy array of images (N, H, W, 1) or (N, H, W)
    
    Returns:
        HOG feature vectors (N, feature_dim)
    """
    # Remove channel dimension if present
    if len(images.shape) == 4:
        images = images.squeeze(-1)
    
    hog_features = []
    
    for img in tqdm(images, desc="Extracting HOG features"):
        # Extract HOG features
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        hog_features.append(features)
    
    return np.array(hog_features)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Loading the images from dataset')
#     parser.add_argument('--data_folder', type=str,required=True, help='Path to dataset directory')
    
#     args = parser.parse_args()
#     load_data(data_dir=args.data_folder)
