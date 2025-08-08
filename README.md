# Handwritten Digit Recognition from Scratch

A Python project to build and understand two classic machine learning models for recognizing handwritten digits from the MNIST dataset, implemented without high-level ML frameworks.

## 🌟 Project Overview

This project was a journey to understand the inner workings of machine learning by building classifiers from the ground up. The goal was to rely only on fundamental libraries like NumPy for numerical operations and to implement the learning algorithms manually.

The project includes two main classifiers:
1.  **K-Nearest Neighbors (KNN):** A simple, instance-based learning algorithm.
2.  **Neural Network:** A basic multi-layer perceptron with backpropagation for training.

## ✨ Features

- **K-Nearest Neighbors (KNN) Classifier:**
  - Built from scratch using NumPy.
  - Calculates Euclidean distance to find the nearest neighbors.
  - Predicts digits based on a majority vote.
- **Neural Network Classifier:**
  - Implemented with a flexible architecture (input, hidden, output layers).
  - Uses the Sigmoid activation function.
  - Learns through backpropagation and gradient descent.
- **Model Evaluation:**
  - Scripts to calculate model accuracy on a test set.
  - Functionality to plot the learning curve (accuracy vs. epochs) for the Neural Network.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/jules-gd-dev/from-scratch-digits-recognition
    cd from-scratch-digits-recognition
    ```
2.  Install the required libraries. The goal was to keep dependencies minimal. You should use a virtualenv (e.g., pew)
    ```bash
    pip install numpy matplotlib
    ```

## 🚀 Usage

To run the models, you can execute the Python scripts directly. Make sure you have the MNIST dataset files (e.g., `mnist_train.csv`) in the same directory.

- **To run the KNN classifier:**
  ```bash
  python knn_digits.py
  ```

- **To run the Neural Network:** (n.b: you won't be able to "use" the neural network for the moment, I have only make the training part, homewer, you will be able to see the accuracy of the model, in training_accr.png)
  ```bash
  python neural_network.py
  ```