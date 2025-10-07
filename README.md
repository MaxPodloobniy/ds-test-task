# ML Engineering Internship Test Assignment

This repository contains solutions for a two-part machine learning test assignment covering various aspects of ML engineering, including object-oriented programming, computer vision, and natural language processing.

## Repository Structure

The assignment consists of two independent tasks, each located in its own directory with complete documentation, source code, and demonstrations:

```
.
├── task1_mnist_classification/     # Image classification with OOP design
│   ├── README.md                   # Complete task documentation
│   ├── requirements.txt            # Task-specific dependencies
│   ├── demo.ipynb                  # Jupyter notebook with demonstrations
│   └── ...
│
├── task2_animal_verification/      # NER + Image classification pipeline
│   ├── README.md                   # Complete task documentation
│   ├── requirements.txt            # Task-specific dependencies
│   ├── demo.ipynb                  # Jupyter notebook with demonstrations
│   └── ...
│
└── README.md                       # This file
```

## Task Descriptions

### Task 1: Image Classification with OOP

The first task demonstrates object-oriented programming principles applied to machine learning. It implements three different classification algorithms for the MNIST dataset, all following a unified interface pattern. The solution includes a Random Forest classifier, a Feed-Forward Neural Network, and a Convolutional Neural Network, each encapsulated in separate classes that implement a common interface. A unified `MnistClassifier` wrapper class provides a consistent API regardless of which algorithm is selected.

**Key Features:**
- Interface-based design with `MnistClassifierInterface`
- Three classification models: Random Forest, Feed-Forward NN, and CNN
- Unified `MnistClassifier` wrapper for algorithm selection
- Consistent input/output structure across all models

**Full Documentation:** See `task1_mnist_classification/README.md` for detailed information about the architecture, usage examples, and implementation details.

### Task 2: Animal Verification Pipeline

The second task builds an end-to-end machine learning pipeline that combines computer vision and natural language processing to verify statements about animals. The system accepts a text statement and an image, then determines whether the animal mentioned in the text matches the animal shown in the image. This involves training a custom image classification model on a self-collected dataset and a Named Entity Recognition model for extracting animal names from text.

**Key Features:**
- Custom image classification model (VGG16-based, 94.97% validation accuracy)
- Custom NER model (spaCy-based, 95.12% F1-score)
- Integrated verification pipeline
- Handles various edge cases and text variations

**Full Documentation:** See `task2_animal_verification/README.md` for complete information about dataset collection, model training, pipeline usage, and performance metrics.

## Quick Start

Each task can be set up and run independently. Navigate to the respective task directory and follow the installation instructions in its README file.

### Task 1 Setup

```bash
cd task1_mnist_classification
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

### Task 2 Setup

```bash
cd task2_animal_verification
pip install -r requirements.txt
python -m spacy download en_core_web_md
jupyter notebook demo.ipynb
```

Detailed setup instructions, including optional dependencies and configuration steps, are available in each task's README file.

## Project Requirements

Both tasks follow the general requirements specified in the assignment:

- **Language:** All code is written in Python 3.9+
- **Code Quality:** Clear, well-commented, and following best practices
- **Documentation:** Complete README files with setup instructions and solution explanations
- **Dependencies:** Each task includes its own `requirements.txt` file
- **Demonstrations:** Jupyter notebooks showcasing functionality and edge cases
- **Language:** All documentation and comments are in English

## Demonstrations

Each task includes a comprehensive Jupyter notebook (`demo.ipynb`) that demonstrates the solution's functionality. These notebooks include standard use cases, edge case handling, visual examples, and detailed explanations of how the models work. The demonstrations are designed to provide clear evidence of the solution's robustness and capabilities.

## Technical Stack

The project utilizes modern machine learning and data science libraries:

**Task 1:**
- scikit-learn for traditional ML algorithms
- TensorFlow/Keras for deep learning models
- NumPy and pandas for data manipulation

**Task 2:**
- TensorFlow/Keras for image classification
- spaCy for NER model training
- Custom data collection and preprocessing pipelines

## Testing

Both tasks can be tested using their respective demo notebooks or by running the provided inference scripts with custom inputs. Each component has been tested with various inputs including edge cases to ensure robustness and reliability.
