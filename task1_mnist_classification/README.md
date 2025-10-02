# MNIST Classifier

## Overview
This project implements an image classification system for the MNIST dataset using three different machine learning models:

1. **Random Forest** (RF)
2. **Feed-Forward Neural Network** (NN)
3. **Convolutional Neural Network** (CNN)

The implementation follows an object-oriented approach, encapsulating each model within a dedicated class that adheres to a common interface.

## Project Structure
```
MNIST_Classifier/
│── models/
│   ├── RandomForestClassifierMnist.py
│   ├── NeuralNetworkClassifierMnist.py
│   ├── CNNClassifierMnist.py
│── MnistClassifier.py
│── MnistClassifierInterface.py
│── requirements.txt
│── MnistTest.ipynb
│── README.md
```

- `MnistClassifier.py`: Central class that initializes and manages different classification models based on user input.
- `MnistClassifierInterface.py`: Defines a common interface for all classifiers.
- `models/`: Contains implementations for the three classifiers.
- `demo.ipynb`: Jupyter Notebook with test cases and example usage.
- `requirements.txt`: Dependencies required for the project.

## Implementation Details
### MnistClassifierInterface
An abstract base class that enforces the implementation of two methods:
- `train(dataset)`: Trains the classifier on the provided dataset.
- `predict(X)`: Predicts labels for the given input data.

### MnistClassifier
A wrapper class that allows selecting a model type (`'rf'`, `'nn'`, or `'cnn'`) and provides a unified way to train and predict.

### Model Implementations
- **RandomForestClassifierMnist**
  - Uses `sklearn.ensemble.RandomForestClassifier` with 100 trees.
  - Requires dataset reshaping to a 1D vector of size 784.
  - Evaluates accuracy after training.

- **NeuralNetworkClassifierMnist**
  - Implements a feed-forward neural network with two hidden layers.
  - Uses `ReLU` activation and dropout for regularization.
  - Trained with `adam` optimizer and `sparse_categorical_crossentropy` loss.

- **CNNClassifierMnist**
  - Uses two convolutional layers with max-pooling.
  - Flatten layer followed by dense layers.
  - Optimized with `adam` and `sparse_categorical_crossentropy`.

## Setup & Usage
### Installation
Ensure Python 3 is installed, then install dependencies:
```sh
pip install -r requirements.txt
```

### Running the Classifier
```python
from MnistClassifier import MnistClassifier

# Initialize with desired model type
classifier = MnistClassifier(alg_type='cnn')
classifier.train()
predictions = classifier.predict()
```

### Jupyter Notebook Demo
Run `demo.ipynb` to see model training, evaluation, and sample predictions.

## Dependencies
- TensorFlow
- Scikit-learn
- NumPy
- Matplotlib
