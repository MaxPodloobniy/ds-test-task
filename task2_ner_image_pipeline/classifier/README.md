# Image Classification Model

This directory contains the implementation of a custom image classification model for animal recognition using transfer learning with VGG16.

## Overview

The model is designed to classify images of animals into multiple categories. It uses a fine-tuned VGG16 architecture with custom dense layers to achieve high accuracy on a custom-collected dataset.

## Dataset
The dataset was collected from multiple sources:
- **Pinterest scraping** - automated collection using the web scraper
- **Manual collection** - additional images gathered manually
- **Manual filtering** - all images were manually reviewed and filtered for quality

The web scraper code is available in `web-scraper.py`. 

> **Note:** If you want to test the scraper functionality, you need to install Playwright browsers:
> ```bash
> playwright install
> ```
> Simply importing Playwright is not sufficient - the browsers must be installed separately.

### Dataset Structure
```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

## Model Architecture

### Base Model
- **Architecture**: VGG16 (pre-trained on ImageNet)
- **Input shape**: (224, 224, 3)
- **Modification**: Top layers removed (`include_top=False`)

### Fine-tuning Strategy
- **Trainable layers**: First 15 layers of VGG16
- **Frozen layers**: Layers 15 onwards remain frozen

### Custom Head
The model includes a custom classification head:

1. **Global Average Pooling 2D**
2. **Dense Block 1**: Dense layer (1024 neurons), Batch Normalization, ReLU activation, Dropout (0.3)
3. **Dense Block 2**: Dense layer (512 neurons), Batch Normalization, ReLU activation, Dropout (0.2)
4. **Output layer**: Dense layer (num_classes neurons), Softmax activation

### Training Configuration
- **Loss function**: `categorical_crossentropy`
- **Optimizer**: Adam
- **Default learning rate**: 1e-4
- **Image size**: 224x224
- **Batch size**: 24 (training), 2 (validation)
- **Test split**: 20%
- **Epochs**: 100 (with early stopping)

### Callbacks
- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training after 4 epochs without improvement
- **ReduceLROnPlateau**: Reduces learning rate by 0.5 after 2 epochs without improvement

## Training Results

The model was trained on **Kaggle** resources using GPU acceleration.

**Validation Accuracy**: 94.97% (0.94972)

The complete training process, including all results, visualizations, and logs, can be found in the Kaggle notebook:  
**File**: `Training image classifier model.ipynb` This notebook contains the same code as `train_classifier.py` but adapted for Kaggle environment

## Files Description

### Training
- **`train_classifier.py`** - Main training script for local/server training
- **`Training image classifier model.ipynb`** - Kaggle notebook with training process and results
- **`infer_classifier.py`** - Script for making predictions on single images
- **`model/classifier_model.keras`** - Trained model weights
- **`model/classifier_model_classes.json`** - Class indices mapping
- **`eval_dataset/`** - Sample images for testing the model

## Usage

### Training the Model

Run from the terminal:

```bash
python train_classifier.py \
    --dataset_path path/to/dataset \
    --img_size 224 \
    --test_split 0.2 \
    --batch_size 24 \
    --val_batch_size 2 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --model_save_path model/classifier_model.keras
```

#### Arguments:
- `--dataset_path` (required) - Path to the dataset directory
- `--img_size` (default: 224) - Input image size
- `--test_split` (default: 0.2) - Validation split ratio
- `--batch_size` (default: 24) - Training batch size
- `--val_batch_size` (default: 2) - Validation batch size
- `--learning_rate` (default: 1e-4) - Learning rate for optimizer
- `--epochs` (default: 100) - Maximum number of training epochs
- `--model_save_path` (default: model/classifier_model.keras) - Path to save the model

### Making Predictions

Run from the terminal:

```bash
python infer_classifier.py \
    --image_path path/to/image.jpg \
    --model_path model/classifier_model.keras
```

#### Arguments:
- `--image_path` (required) - Path to the image for classification
- `--model_path` (default: model/classifier_model.keras) - Path to the trained model

## Model Output

The inference script returns:
- **Predicted class**: The animal category
- **Confidence score**: Probability of the prediction (0-100%)

Example output:
```
2024-10-07 12:34:56 [INFO] Predicted class: chimpanzee (confidence: 98.75%)
```

