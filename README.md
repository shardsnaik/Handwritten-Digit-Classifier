# Digit Classification Project


A complete digit classification system (0-9) with two baseline models: HOG + Logistic Regression and CNNs.

### The deployed front-end using `streamlit`


## https://handwritten-digit-classifier-bcaesvazfgrmghljhbpess.streamlit.app/

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   ├── analyze.py         # Analysis and visualization
│   ├── data_loader.py     # Data loading utilities
│   └── models.py          # Model architectures
├── dataset/               # Your dataset (not included)
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
├── models/                # Saved models (generated)
├── analysis/              # Analysis outputs (generated)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd digit-classification

# Create virtual environment (recommended)
source myenv_env/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Format

Place your images in the following structure:
```
dataset/
  0/
    img001.jpg
    img002.jpg
    ...
  1/
    img001.jpg
    ...
  ...
  9/
    img001.jpg
    ...
```

## Training

### Train all models (recommended)

```bash
python -m src.train --data ./dataset --epochs 15 --img_size 64
```

### Train specific models

```bash
# HOG + Logistic Regression only
python -m src.train --data ./dataset --epochs 15 --img_size 64 --models hog

# Baseline CNN only
python -m src.train --data ./dataset --epochs 15 --img_size 64 --models baseline_cnn

# Improved CNN only
python -m src.train --data ./dataset --epochs 15 --img_size 64 --models improved_cnn

```

**Outputs:**
- `models/hog_logreg.pkl` - HOG + Logistic Regression model
- `models/baseline_cnn_best.h5` - Baseline CNN model
- `models/improved_cnn_best.h5` - Improved CNN model
- `models/all_metrics.json` - Training metrics for all models

## Inference

### Using the best model (Improved CNN)

```bash
python -m src.inference --images ./test_images --weights ./models/improved_cnn_best.h5 --out preds.csv
```

### Using other models

```bash
# Baseline CNN
python -m src.inference --images ./test_images --weights ./models/baseline_cnn_best.h5 --out preds.csv

# HOG + Logistic Regression
python -m src.inference --images ./test_images --weights ./models/hog_logreg.pkl --out preds.csv
```

**Output CSV format:**
```csv
image_path,predicted_digit,confidence
img1.jpg,5,0.9876
img2.jpg,3,0.8543
...
```

## Analysis & Visualization

Generate confusion matrices and find misclassifications:

```bash
python -m src.analyze --data ./dataset --weights ./models/improved_cnn_best.h5
```

**Outputs in `analysis/` directory:**
- Confusion matrices (validation and test)
- Misclassified samples visualization
- Detailed misclassification report (JSON)

## Model Descriptions

### 1. Baseline: HOG + Logistic Regression

A classical computer vision approach:
- **Feature extraction**: Histogram of Oriented Gradients (HOG)
  - 9 orientations
  - 8x8 pixels per cell
  - 2x2 cells per block
- **Classifier**: Logistic Regression with L2 regularization
- **Pros**: Fast, interpretable, no GPU required
- **Cons**: Limited capacity for complex patterns

### 2. Baseline CNN (3-layer)

A simple convolutional neural network:
- **Architecture**:
  - Conv2D(32) → MaxPool
  - Conv2D(64) → MaxPool
  - Conv2D(64) → MaxPool
  - Dense(64) → Dropout(0.5) → Output
- **Parameters**: ~150K
- **Training**: Data augmentation (rotation ±15°, shift ±10%, brightness)

### 3. Improved CNN (Deeper)

An enhanced architecture with modern techniques:
- **Architecture**:
  - 4 convolutional blocks with batch normalization
  - Progressive filters: 32 → 64 → 128 → 256
  - Global Average Pooling
  - Dense layers with dropout
- **Parameters**: ~1M
- **Techniques**:
  - Batch normalization for faster convergence
  - Dropout for regularization (0.25 and 0.5)
  - Data augmentation
  - Early stopping

## Data Split Strategy

- **Method**: Stratified split with fixed seed (42)
- **Ratios**: Train/Val/Test = 70/15/15
- **Stratification**: Ensures balanced class distribution across all splits
- **Augmentation**: Applied ONLY to training data
  - Rotation: ±15 degrees
  - Width/Height shift: ±10%
  - Brightness: 80-120%

## Metrics

All models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Macro-F1**: Average F1-score across all classes (handles class imbalance)
- **Confusion Matrix**: Per-class performance visualization

## Expected Performance

Based on typical digit datasets:

| Model | Val Accuracy | Test Accuracy | Val Macro-F1 | Test Macro-F1 |
|-------|-------------|---------------|--------------|---------------|
| HOG + LogReg | ~88-92% | ~88-92% | ~0.88-0.92 | ~0.88-0.92 |
| Baseline CNN | ~92-95% | ~92-95% | ~0.92-0.95 | ~0.92-0.95 |
| Improved CNN | **~95-98%** | **~95-98%** | **~0.95-0.98** | **~0.95-0.98** |

*Note: Actual performance depends on your dataset quality and size.*
