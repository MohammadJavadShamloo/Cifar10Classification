# CIFAR10 Image Classification with CNN

## Overview

This notebook demonstrates a comprehensive approach to image classification using Convolutional Neural Networks (CNNs) on the CIFAR10 dataset. The project is divided into two main parts:

1.  **CIFAR10 Classification**: Building and training a CNN from scratch
2.  **Transfer Learning**: Adapting the trained model for CIFAR100 dataset

## Table of Contents

1.  [Environment Setup](#environment-setup)
2.  [Data Preparation](#data-preparation)
3.  [Data Visualization](#data-visualization)
4.  [Model Architecture](#model-architecture)
5.  [Training Process](#training-process)
6.  [Model Evaluation](#model-evaluation)
7.  [Feature Space Analysis](#feature-space-analysis)
8.  [Transfer Learning to CIFAR100](#transfer-learning-to-cifar100)

## Environment Setup

The notebook begins by importing essential libraries:

python

Line Wrapping

Collapse

Copy

99

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

›

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

The environment is configured to use GPU when available:

python

Line Wrapping

Collapse

Copy

9

1

›

device \= 'cuda' if torch.cuda.is\_available() else 'cpu'

## Data Preparation

### Dataset Loading

The CIFAR10 dataset is loaded with appropriate transformations:

python

Line Wrapping

Collapse

Copy

99

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

›

⌄

⌄

⌄

data\_transforms \= {

'train': transforms.Compose(\[

transforms.RandomHorizontalFlip(),

transforms.RandomCrop(32, padding\=4),

transforms.ToTensor(),

transforms.Normalize(

mean\=\[0.4914, 0.4822, 0.4465\],

std\=\[0.2023, 0.1994, 0.2010\],

),

\]),

'test': transforms.Compose(\[

transforms.ToTensor(),

transforms.Normalize(

mean\=\[0.4914, 0.4822, 0.4465\],

std\=\[0.2023, 0.1994, 0.2010\],

),

\]),

}

### Data Splitting

The training set is split into training (80%) and validation (20%) subsets:

python

Line Wrapping

Collapse

Copy

9

1

2

3

›

train\_size \= int(len(full\_train\_dataset) \* 0.8)

valid\_size \= len(full\_train\_dataset) \- train\_size

train\_dataset, validation\_dataset \= random\_split(full\_train\_dataset, \[train\_size, valid\_size\])

### Data Loaders

Data loaders are created for batch processing:

python

Line Wrapping

Collapse

Copy

9

1

2

3

›

train\_loader \= DataLoader(train\_dataset, batch\_size\=64, shuffle\=True, num\_workers\=2)

validation\_loader \= DataLoader(validation\_dataset, batch\_size\=64, num\_workers\=2)

test\_loader \= DataLoader(test\_dataset, batch\_size\=64, num\_workers\=2)

## Data Visualization

The notebook includes visualization of 5 random images from each class in CIFAR10:

python

Line Wrapping

Collapse

Copy

9

1

2

3

4

›

⌄

\# Visualization code (see notebook for implementation)

plt.figure(figsize\=(15, 10))

for i, cls in enumerate(classes):

\# ... visualization logic

## Model Architecture

The notebook designs a CNN architecture specifically for CIFAR10 classification. While the exact architecture isn't shown in the provided content, it typically includes:

*   Convolutional layers with ReLU activation
*   Pooling layers for spatial reduction
*   Batch normalization for training stability
*   Dropout for regularization
*   Fully connected layers for classification

## Training Process

The model is trained using:

*   Loss function: Cross-entropy loss
*   Optimizer: Adam or SGD (with momentum)
*   Learning rate scheduling
*   Early stopping based on validation performance

## Model Evaluation

After training, the model is evaluated on:

1.  Test accuracy
2.  Confusion matrix
3.  Per-class performance metrics

## Feature Space Analysis

The notebook includes comprehensive analysis of the learned feature space:

1.  **KNN Analysis**: Examining nearest neighbors in feature space
2.  **Clustering**: Applying clustering algorithms to feature representations
3.  **Intermediate Layer Visualization**: Visualizing activations from different layers

python

Line Wrapping

Collapse

Copy

9

1

›

\# Feature space analysis code (see notebook for implementation)

## Transfer Learning to CIFAR100

In the second part, the notebook demonstrates transfer learning:

1.  **Model Adaptation**: Modifying the final layer for 100 classes
2.  **Fine-tuning**: Retraining on CIFAR100 dataset
3.  **Evaluation**: Assessing performance on the new task
4.  **Generalization Analysis**: Examining how well features transfer

## Key Features

*   **Comprehensive Data Processing**: Includes normalization and augmentation
*   **Thorough Evaluation**: Multiple metrics and visualization techniques
*   **Feature Analysis**: Deep dive into learned representations
*   **Transfer Learning**: Practical application to a new dataset
*   **Visualization**: Rich visualizations throughout the process

## Usage

To run this notebook:

1.  Ensure all dependencies are installed (PyTorch, torchvision, etc.)
2.  Execute cells sequentially
3.  Adjust hyperparameters as needed for your environment

## Conclusion

This notebook provides a complete pipeline for image classification with CNNs, from data preparation to advanced feature analysis and transfer learning. It serves as both a practical implementation and an educational resource for understanding deep learning for computer vision tasks.
