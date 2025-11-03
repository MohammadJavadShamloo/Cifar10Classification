# CIFAR10 and CIFAR100 Image Classification with CNN and Transfer Learning

This repository contains a Jupyter notebook (Classification.ipynb) that demonstrates image classification on the CIFAR10 dataset using a Convolutional Neural Network (CNN). It includes data loading, preprocessing, model design, training, evaluation, and in-depth analysis of the feature space. The second part extends the model via transfer learning to the CIFAR100 dataset, with modifications to the final layer, retraining, and further evaluation of generalization and features.

The notebook is designed for educational purposes, focusing on practical implementation and analysis of CNNs in PyTorch.

## Table of Contents

*   Project Overview
*   Dataset
*   Features
*   Requirements
*   Installation
*   Usage
*   Results and Analysis
*   Contributing
*   License

## Project Overview

*   **Part 1: CIFAR10 Classification**
    *   Load and preprocess the CIFAR10 dataset with normalization and data augmentation.
    *   Visualize sample images from each class.
    *   Design and train a custom CNN model.
    *   Evaluate model performance on test data.
    *   Analyze the feature space using KNN for nearest neighbors, clustering, and visualization of intermediate layer outputs.
*   **Part 2: Transfer Learning to CIFAR100**
    *   Adapt the pre-trained CIFAR10 model by modifying the final layer.
    *   Retrain on CIFAR100 dataset.
    *   Evaluate accuracy and analyze feature extraction, generalization, and potential issues like class imbalances.

The notebook also includes a discussion on accuracy differences between classes and relates them to the model's feature space.

## Dataset

*   **CIFAR10**: 60,000 32x32 color images in 10 classes (e.g., plane, car, bird), with 50,000 training and 10,000 test images.
*   **CIFAR100**: Extension with 100 classes, used for transfer learning to test model adaptability.
*   Datasets are automatically downloaded via torchvision.datasets.CIFAR10 and CIFAR100.

## Features

*   Data augmentation (random horizontal flip, crop) for improved generalization.
*   Custom CNN architecture (explained in the notebook).
*   Training with validation split and evaluation metrics.
*   Feature space analysis: KNN, clustering (e.g., t-SNE), and layer visualizations.
*   Transfer learning demonstration with layer modifications.
*   Visualization of best/worst performing classes and accuracy analysis.

## Requirements

*   Python 3.10+
*   PyTorch
*   Torchvision
*   Matplotlib
*   NumPy
*   Scikit-learn (for t-SNE and KNN)
*   Seaborn (for visualizations)
*   tqdm (for progress bars)

See the notebook's import section for the full list:

```python
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from time import time
import random
from sklearn.manifold import TSNE
import numpy as np
from random import sample
import math
import torch.optim as optim
import seaborn as sns
```

## Installation

1.  Clone the repository:
    ```
    git clone https://github.com/yourusername/cifar-cnn-classification.git
    cd cifar-cnn-classification
    ```
    
2.  Install dependencies:    
    ```
    pip install torch torchvision matplotlib numpy scikit-learn seaborn tqdm
    ```
    
3.  (Optional) Use a virtual environment:
    ```
    python -m venv env
    source env/bin/activate  # On Linux/Mac
    .\env\Scripts\activate   # On Windows
    pip install -r requirements.txt  # Create this file with the above libraries
    ```
    

## Usage

1.  Open the Jupyter notebook:
    ```
    jupyter notebook Classification.ipynb
    ```
    
2.  Run the cells sequentially:
    *   Data loading and preprocessing.
    *   Model definition and training.
    *   Evaluation and visualizations.
    *   Transfer learning section.

Note: Training requires a GPU for faster computation (set device = 'cuda' if available). The notebook includes code to check for CUDA availability.

## Results and Analysis

*   **CIFAR10 Accuracy**: The model achieves competitive accuracy (detailed in the notebook's evaluation section).
*   **Class Performance**: Visualizations show top-5 best/worst performing classes (e.g., high accuracy for distinct classes like "ship" vs. lower for similar ones like "cat/dog").
*   **Feature Space Insights**:
    *   KNN reveals closest samples in embeddings.
    *   Clustering (t-SNE) highlights separability.
    *   Intermediate layer outputs visualize learned features.
*   **Transfer Learning**: Accuracy on CIFAR100 is lower due to increased classes, but demonstrates effective fine-tuning.
*   **Discussion**: Accuracy differences stem from class similarities (e.g., animals vs. vehicles), dataset biases, and feature space limitations. The model excels at broad distinctions but struggles with fine-grained subclasses.

Refer to the notebook for plots, confusion matrices, and detailed analysis.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or additional features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
