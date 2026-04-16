## Source Code (src)

This folder contains the Python scripts used to train and evaluate machine learning models for ransomware detection using static file features.

### Scripts

`train_models.py`
Main training and evaluation pipeline.

It trains and evaluates four supervised machine learning models:

* Logistic Regression (LR) — baseline linear model
* Random Forest (RF) — ensemble model for non-linear patterns
* Support Vector Machine (SVM, RBF kernel) — best overall performance
* K-Nearest Neighbors (KNN) — distance-based classifier

### Features

The script performs:

* Dataset loading from CSV (or demo dataset generation)

* Train/test split

* Model training and evaluation

* Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC

* Confusion matrix generation (saved as PNG)

### How to Run

From the repository root:

**1) Demo mode (no dataset required)**
Generates a synthetic dataset and runs the full pipeline.

```bash
python src/train_models.py
```

**2) Run with your dataset (CSV)**
Provide a CSV path and the label column name:

```bash
python src/train_models.py --data path/to/your_dataset.csv --label-col label
```
