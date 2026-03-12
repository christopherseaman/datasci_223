# Assignment 6: Neural Networks and Health Data Analysis

## Overview

This assignment introduces neural networks and their applications in healthcare data analysis. You'll work through three parts:

1. **Basic Neural Networks**: Implement a simple neural network for EMNIST character recognition
2. **Convolutional Neural Networks**: Build a CNN for more complex image classification
3. **Time Series Analysis**: Apply neural networks to ECG signal classification

## Learning Objectives

- Implement and train neural networks using TensorFlow/Keras
- Apply CNNs for image classification
- Work with time series data using RNNs
- Evaluate model performance using appropriate metrics
- Interpret results in a healthcare context

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Directory Structure**:
   ```
   datasci223_assignment6/
   ├── models/              # Saved models
   ├── results/            # Evaluation metrics
   │   ├── part_1/        # Part 1 results
   │   ├── part_2/        # Part 2 results
   │   └── part_3/        # Part 3 results
   ├── logs/              # Training logs
   ├── data/              # Downloaded datasets
   ├── part1_neural_networks_basics.md
   ├── part2_cnn_classification.md
   ├── part3_ecg_analysis.md
   └── requirements.txt
   ```

## Part 1: Neural Networks Basics

- Implement a simple neural network for EMNIST character recognition
- Use dense layers, activation functions, and dropout
- Save model as `models/emnist_classifier.keras`
- Save metrics in `results/part_1/emnist_classifier_metrics.txt`

## Part 2: CNN Classification

- Implement a CNN for EMNIST classification
- Choose between TensorFlow/Keras or PyTorch
- Save model as:
  - TensorFlow: `models/cnn_keras.keras`
  - PyTorch: `models/cnn_pytorch.pt` and `models/cnn_pytorch_arch.txt`
- Save metrics in `results/part_2/cnn_{framework}_metrics.txt`

## Part 3: ECG Analysis

- Work with MIT-BIH Arrhythmia Database
- Choose between simple neural network or RNN/LSTM
- Save model as `models/ecg_classifier_{model_type}.keras`
- Save metrics in `results/part_3/ecg_classifier_{model_type}_metrics.txt`

## Submission Requirements

1. **Code Organization**:
   - Complete all three parts
   - Follow the provided directory structure
   - Include all required files

2. **Model Files**:
   - Part 1: `models/emnist_classifier.keras`
   - Part 2: Either `models/cnn_keras.keras` or `models/cnn_pytorch.pt` + `models/cnn_pytorch_arch.txt`
   - Part 3: `models/ecg_classifier_{model_type}.keras`

3. **Metrics Files**:
   - Part 1: `results/part_1/emnist_classifier_metrics.txt`
   - Part 2: `results/part_2/cnn_{framework}_metrics.txt`
   - Part 3: `results/part_3/ecg_classifier_{model_type}_metrics.txt`

4. **Metrics Format**:
   ```
   model: model_name
   accuracy: float
   precision: float
   recall: float
   f1_score: float
   confusion_matrix: [[TN, FP], [FN, TP]]
   ----
   ```

## Intended Endpoint

1. **Part 1 (Basic NN)**:
   - Accuracy > 80% on test set
   - Must use at least 2 dense layers
   - Must include dropout layers
   - Must use categorical crossentropy loss

2. **Part 2 (CNN)**:
   - Accuracy > 85% on test set
   - Must use at least 2 convolutional layers
   - Must include pooling and batch normalization
   - Must use categorical crossentropy loss

3. **Part 3 (ECG)**:
   - Accuracy > 75% on test set
   - AUC > 0.80
   - F1-score > 0.70
   - Must use appropriate architecture (simple NN or RNN)

## Framework Options

1. **TensorFlow/Keras**:
   - Recommended for beginners
   - Simpler syntax and more examples
   - Save models as `.keras` files

2. **PyTorch**:
   - More flexible and powerful
   - Better for research and customization
   - Save models as `.pt` files with architecture in separate `.txt` file

## Common Issues and Solutions

1. **Data Loading**:
   - Problem: Dataset not found
   - Solution: Check directory structure and download scripts

2. **Model Training**:
   - Problem: Training instability
   - Solution: Use batch normalization, reduce learning rate
   - Problem: Overfitting
   - Solution: Increase dropout, use data augmentation

3. **Evaluation**:
   - Problem: Metrics format incorrect
   - Solution: Follow the exact format specified
   - Problem: Model performance below threshold
   - Solution: Adjust architecture, hyperparameters

## Resources

1. **Documentation**:
   - [TensorFlow Guide](https://www.tensorflow.org/guide)
   - [PyTorch Tutorials](https://pytorch.org/tutorials/)
   - [MIT-BIH Database](https://www.physionet.org/content/mitdb/1.0.0/)
   - [TensorFlow Neural Networks](https://www.tensorflow.org/tutorials)
   - [PyTorch CNNs](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
   - [RNN/LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
