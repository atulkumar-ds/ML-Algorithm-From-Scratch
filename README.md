# Machine Learning Algorithms from Scratch

## Overview

This repository contains implementations of core machine learning algorithms built entirely from first principles using Python and NumPy.

The primary objective of this project is to develop a deep understanding of optimization, gradient-based learning, and distance-based methods by manually implementing model training and evaluation pipelines without relying on high-level machine learning libraries such as scikit-learn.

The emphasis is on mathematical clarity, algorithmic correctness, and clean implementation.

---

## Motivation

Modern machine learning frameworks abstract away much of the underlying mathematics. While this improves productivity, it can limit conceptual depth.

This project focuses on:

- Understanding gradient-based optimization
- Implementing loss functions explicitly
- Studying convergence behavior
- Strengthening intuition behind classification algorithms
- Preparing for technical interviews requiring algorithm-level clarity

---

## Implemented Algorithms

### Logistic Regression
- Binary classification
- Sigmoid activation function
- Cross-entropy loss
- Manual gradient computation
- Batch gradient descent optimization
- Custom train-test split implementation

### Gradient Descent (Standalone)
- Parameter initialization
- Iterative update rule
- Learning rate experimentation
- Convergence tracking

### K-Nearest Neighbors (KNN)
- Euclidean distance computation
- Majority voting
- Manual prediction pipeline
- Accuracy evaluation without external ML libraries

---

## Technical Approach

Each algorithm follows a structured implementation pipeline:

1. Data loading using Pandas  
2. Conversion to NumPy arrays for numerical computation  
3. Parameter initialization  
4. Forward propagation  
5. Loss computation  
6. Gradient calculation  
7. Parameter updates  
8. Prediction and evaluation  

All mathematical steps are implemented explicitly to ensure transparency and full control over the training process.

---

## Project Structure

ML-Algorithm-From-Scratch/
│
├── Logistic_Regression.ipynb
├── Gradient_Descent.ipynb
├── Knn.ipynb
├── datasets (.csv files)
└── README.md

---

## Design Principles

- No high-level ML frameworks used for training
- Clear separation between training and evaluation logic
- Emphasis on mathematical transparency
- Readable and structured code

---

## Future Improvements

- Linear Regression with L1/L2 regularization
- Decision Tree (CART) implementation
- Naive Bayes classifier
- Performance benchmarking against scikit-learn
- Refactoring notebooks into modular Python scripts
- Adding unit tests and benchmarking support

---

## Author

Atul Kumar  
B.Tech – Artificial Intelligence & Data Science  
IIITDM Kurnool
