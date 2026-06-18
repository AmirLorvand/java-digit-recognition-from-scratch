# Java Digit Recognition from Scratch

This project is a Java-based **digit recognition machine learning coursework project**. It implements multiple machine learning algorithms **from scratch**, without using machine learning libraries such as scikit-learn, TensorFlow, PyTorch, or Weka.

The goal of the project is to classify handwritten digit data using different algorithmic approaches and compare their behaviour on the same digit recognition task.

---

## Project Timeline

* **Originally completed:** 2024
* **Refactored and published on GitHub:** 2026
* **Context:** Artificial Intelligence coursework project

This repository has been cleaned and documented for portfolio purposes.

---

## Project Status

This is a completed coursework-style machine learning project. The focus was not on using ready-made ML libraries, but on implementing the core logic of different algorithms manually in Java.

The project demonstrates understanding of machine learning fundamentals, including distance-based classification, neural networks, support vector machines, feature transformation, cross-validation, and ensemble voting.

---

## Algorithms Implemented

### 1. Nearest Neighbour using Euclidean Distance

Implemented in:

```text
digit_reco_NN/
```

This classifier predicts the digit label by comparing test samples to training samples using Euclidean distance.

---

### 2. Multi-Layer Perceptron

Implemented in:

```text
digit_reco_MLP/
```

This implementation uses a neural network with one hidden layer and includes:

* forward propagation
* ReLU activation
* Softmax output
* backpropagation
* hyperparameter testing

---

### 3. Support Vector Machine

Implemented in:

```text
digit_reco_SVM/
```

This implementation uses a multi-class SVM approach with:

* one-vs-all classification
* RBF kernel
* gradient-based optimisation
* hyperparameter search

---

### 4. SVM with Centroid-Based Kernel Features

Implemented in:

```text
SVM_centroid_kernel/
```

This version extends the SVM approach by using centroid-based feature transformation before classification.

---

### 5. Voting System

Implemented in:

```text
voting_system/
```

This combines predictions from multiple models using a voting/ensemble approach.

---

## Dataset

The project uses two CSV datasets:

```text
dataSet1.csv
dataSet2.csv
```

Each dataset represents handwritten digit samples. The input features represent pixel values, and the final column represents the digit label.

The project uses a 2-fold cross-validation style approach:

* train on `dataSet1.csv`, test on `dataSet2.csv`
* train on `dataSet2.csv`, test on `dataSet1.csv`

---

## Technologies Used

* Java
* Machine Learning from scratch
* Artificial Intelligence
* Nearest Neighbour
* Multi-Layer Perceptron
* Support Vector Machine
* Ensemble voting
* CSV file handling
* Cross-validation
* Makefile

---

## Project Structure

```text
.
├── algorithms/
│   ├── nearest-neighbor/
│   │   └── NN_euclidean_dsitance.java
│   ├── mlp/
│   │   └── MLP.java
│   ├── svm/
│   │   └── SVM.java
│   ├── svm-centroid-kernel/
│   │   └── SVM_centroid_kernel.java
│   └── voting-system/
│       └── voting_system.java
├── data/
│   ├── dataSet1.csv
│   └── dataSet2.csv
├── Makefile
├── .gitignore
└── README.md
---

## How to Compile

Compile all algorithms:

```bash
make
```

This creates compiled class files inside the `out/` directory.

---

## How to Run Each Algorithm

Run Nearest Neighbour:

```bash
make run-nn
```

Run Multi-Layer Perceptron:

```bash
make run-mlp
```

Run Support Vector Machine:

```bash
make run-svm
```

Run SVM with centroid-based features:

```bash
make run-svm-centroid
```

Run the voting system:

```bash
make run-voting
```

---

## Clean Build Files

```bash
make clean
```

This removes the generated `out/` directory.

---

## Important Note on Runtime

Some algorithms, especially the SVM and MLP versions, may take longer to run because they include hyperparameter testing and repeated training loops.

For faster experimentation, the hyperparameter arrays inside each Java file can be reduced.

---

## What I Learned

Through this coursework project, I practised:

* Implementing machine learning algorithms manually in Java
* Understanding how digit recognition works from pixel-based features
* Applying Euclidean distance for classification
* Building a neural network from scratch
* Implementing forward propagation and backpropagation
* Implementing SVM-style classification
* Using one-vs-all classification for multi-class problems
* Applying cross-validation
* Comparing different machine learning approaches
* Creating an ensemble voting system
* Working with CSV datasets
* Preparing an AI coursework project for GitHub presentation

---

## Future Improvements

Possible improvements include:

* Refactoring each algorithm into cleaner reusable classes
* Moving duplicate dataset files into a shared `data/` folder
* Adding a results table comparing all models
* Adding visualisations of digit samples
* Adding confusion matrices
* Improving naming consistency across files and packages
* Adding unit tests
* Adding command-line arguments for dataset paths and hyperparameters
* Creating a single runner program to compare all algorithms automatically

---

## Author

**Amir Lorvand**

