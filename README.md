# ransomware-ml-static-features

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v1.0-blueviolet.svg)](https://github.com/Recon53/ransomware-ml-static-features/releases/tag/v1.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18209938.svg)](https://doi.org/10.5281/zenodo.18209938)


## Quick Start
~~~bash
pip install -r requirements.txt
python src/train_models.py
~~~


Machine learning-based ransomware detection using **static file features**, evaluating **Logistic Regression** and **Random Forest** classifiers.

---
### Run with your own dataset (CSV)

python src/train_models.py --data data/your_dataset.csv --label-col label

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Results (Demo Run)

- Logistic Regression Accuracy: ~0.83
- Random Forest Accuracy: ~0.94  
Random Forest performed better overall in this baseline experiment.

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

## Presentation (CAP 5610)

This repository includes the final course presentation and written report for the Machine Learning project **“Detection of Ransomware Using Static Features.”** The presentation summarizes the motivation, dataset, feature engineering approach, model training pipeline, evaluation metrics, and final results.

### Files Included
- **Slide Deck (PowerPoint):** `presentation/Ransomware_ML_Presentation_Miguel.pptx`
- **Final Report (Word/PDF):** `report/Ransomware_Static_Features_Report.docx`

### What the Presentation Covers
- Problem motivation: ransomware detection using machine learning
- Dataset and static PE feature extraction
- Models compared:
  - Logistic Regression (baseline)
  - Random Forest (ensemble approach)
- Performance evaluation using:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
- Key takeaway:
  - **Random Forest consistently outperformed Logistic Regression**, demonstrating that ensemble methods provide stronger performance and robustness for ransomware classification using static features.

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

### Confusion Matrix (Logistic Regression)
<img src="results/confusion_matrix_logistic_regression.png" width="650">

---

### Confusion Matrix (Random Forest)
<img src="results/confusion_matrix_random_forest.png" width="650">
### Model Accuracy (Random Forest)
![Model Accuracy - Random Forest](results/model_accuracy_random_forest.png)

### Top Features (Random Forest)
![Feature Importance - Random Forest](results/feature_importance_random_forest.png)

---

## Citation / Acknowledgements

This project was developed for academic coursework and experimentation using publicly available ML libraries such as scikit-learn.

---

## Citation

If you use this repository, please cite the Zenodo record:

```bibtex
@software{guadalupe_ransomware_ml_static_features_2026,
  author       = {Guadalupe, Miguel},
  title        = {ransomware-ml-static-features},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18209938},
  url          = {https://doi.org/10.5281/zenodo.18209938}
}
