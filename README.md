Overview

This repository contains the implementation of a k-Nearest Neighbors (k-NN) classifier from scratch and using scikit-learn for digit classification on the MNIST dataset. The assignment is divided into two parts:

Part 1: Implementing the k-NN classifier from scratch without using any machine learning toolkits (except NumPy, Pandas, Matplotlib, etc.).

Part 2: Replicating the implementation using the scikit-learn library.

Dataset
The MNIST dataset consists of 70,000 labeled images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is provided in a CSV file with 70,001 rows and 785 columns (1 label column + 784 pixel columns).

Part 1: Implement from Scratch
Tasks
Load and Preprocess Data:

Load the MNIST dataset and split it into training (60,000 images) and test (10,000 images) sets.

Visualize images from the dataset.

Implement Distance Metrics:

Euclidean Distance.

Manhattan Distance.

k-NN Classifier:

Implement the k-NN classifier function.

Handle ties by backing off to smaller values of k.

Evaluation:

Implement functions to compute the confusion matrix, classification accuracy, and macro-average F1 score.

Perform 5-fold cross-validation for k values from 1 to 10 using both distance metrics.

Report evaluation metrics and visualize confusion matrices as heatmaps.

Part 2: Replicate with scikit-learn
Use scikit-learn's k-NN classifier to replicate the results obtained in Part 1.

Compare the performance of the custom implementation with scikit-learn's implementation.



How to Run
Clone the repository:

bash
git clone https://github.com/yourusername/knn-classification.git
cd knn-classification

Install the required libraries:
bash
pip install numpy pandas matplotlib seaborn pillow scikit-learn

Run the Jupyter notebook:
bash
jupyter notebook AyeshaFazalLashkarwala_23100136_Assignment1.ipynb

Results
The notebook includes detailed results for both the custom and scikit-learn implementations, including accuracy, F1 scores, and confusion matrices.
Visualizations of sample images and performance metrics are provided.

