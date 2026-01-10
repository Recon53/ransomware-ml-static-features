# ransomware-ml-static-features

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v1.0-blueviolet.svg)](https://github.com/Recon53/ransomware-ml-static-features/releases/tag/v1.0)

## Quick Start
~~~bash
pip install -r requirements.txt
python src/train_models.py
~~~

Machine learning-based ransomware detection using **static file features**, evaluating **Logistic Regression** and **Random Forest** classifiers.

## Preview
![Confusion Matrix - Random Forest](results/confusion_matrix_random_forest.png)
---

## Project Overview

Ransomware remains one of the most damaging categories of malware, capable of encrypting victim files and disrupting individuals and organizations.  
This project explores a **machine learning (ML) approach** to ransomware detection using **static features**, meaning the model does **not** need to execute the file to make a prediction.

The goal is to evaluate whether static indicators can effectively distinguish ransomware from benign software using supervised ML classifiers.

---

## Repository Structure

~~~
ransomware-ml-static-features/
├── report/                      # Final report document (DOCX)
├── src/                         # Source code (training + evaluation scripts)
├── results/                     # Output figures (confusion matrix, plots)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
~~~

---

## Installation

### 1) Clone repository
~~~bash
git clone https://github.com/Recon53/ransomware-ml-static-features.git
cd ransomware-ml-static-features
~~~

### 2) Install dependencies
~~~bash
pip install -r requirements.txt
~~~

Dependencies include:
- numpy
- pandas
- scikit-learn
- matplotlib

---

## How to Run

From the repository root:

### 1) Demo mode (no dataset required)
Generates a synthetic dataset and runs the full pipeline.

~~~bash
python src/train_models.py
~~~

### 2) Run with your dataset (CSV)
Provide a path to your dataset and the label column name.

~~~bash
python src/train_models.py --data path/to/your_dataset.csv --label-col label
~~~

---

## Expected Output

The script prints evaluation metrics such as:
- Accuracy
- Precision
- Recall
- F1-score

It also saves result images into the `results/` folder, including:

- `results/confusion_matrix_random_forest.png`
- `results/model_accuracy_random_forest.png`
- `results/feature_importance_random_forest.png`

---

## Results (Screenshots)

### Confusion Matrix (Random Forest)
![Confusion Matrix - Random Forest](results/confusion_matrix_random_forest.png)

### Model Accuracy (Random Forest)
![Model Accuracy - Random Forest](results/model_accuracy_random_forest.png)

### Top Features (Random Forest)
![Feature Importance - Random Forest](results/feature_importance_random_forest.png)

---

## Citation / Acknowledgements

This project was developed for academic coursework and experimentation using publicly available ML libraries such as scikit-learn.

---
