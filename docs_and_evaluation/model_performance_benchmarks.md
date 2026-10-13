# CardioVision3D — Model Performance & Evaluation Benchmarks

This document records the exact held-out test evaluation benchmarks, cross-validation metrics, and classification reports across all four trained machine learning pipelines.

## 1. Held-Out Stratified Test Set Evaluation (N=61)

Evaluated on an untouched 20% stratified test split from the Z-Alizadeh Sani cohort:

| Target Key | Clinical Target | Selected Model | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test Cases (Pos / Neg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`cad`** | Overall CAD (`Cath`) | Random Forest | 0.918 ± 0.040 | **85.2%** | **88.6%** | **90.7%** | **0.897** | **0.870** | 61 (43 / 18) |
| **`lad`** | LAD Stenosis | Random Forest | 0.836 ± 0.038 | **77.0%** | **78.9%** | **83.3%** | **0.811** | **0.836** | 61 (36 / 25) |
| **`lcx`** | LCX Stenosis | Random Forest | 0.709 ± 0.046 | **62.3%** | **52.6%** | **50.0%** | **0.511** | **0.700** | 61 (20 / 41) |
| **`rca`** | RCA Stenosis | Random Forest | 0.781 ± 0.065 | **68.9%** | **60.0%** | **52.2%** | **0.558** | **0.730** | 61 (23 / 38) |

## 2. Confusion Matrices (Held-Out Test Set N=61)

### Cath (Overall CAD)
```
                Predicted Normal    Predicted CAD
Actual Normal         13                  5
Actual CAD             4                 39
```
- True Negatives: 13 | False Positives: 5 | False Negatives: 4 | True Positives: 39

### LAD (Left Anterior Descending)
```
                Predicted Normal    Predicted Stenotic
Actual Normal         17                  8
Actual Stenotic        6                 30
```
- True Negatives: 17 | False Positives: 8 | False Negatives: 6 | True Positives: 30

### LCX (Left Circumflex)
```
                Predicted Normal    Predicted Stenotic
Actual Normal         28                 13
Actual Stenotic       10                 10
```
- True Negatives: 28 | False Positives: 13 | False Negatives: 10 | True Positives: 10

### RCA (Right Coronary Artery)
```
                Predicted Normal    Predicted Stenotic
Actual Normal         30                  8
Actual Stenotic       11                 12
```
- True Negatives: 30 | False Positives: 8 | False Negatives: 11 | True Positives: 12

## 3. Algorithm Selection (5-Fold Cross-Validation on Training Folds)

Benchmarking candidate algorithms using stratified 5-fold cross-validation:

| Target | Logistic Regression CV ROC-AUC | Random Forest Ensemble CV ROC-AUC | Selected Model |
| :--- | :---: | :---: | :---: |
| **Overall CAD** | 0.916 ± 0.025 | **0.918 ± 0.040** | **Random Forest** |
| **LAD Stenosis** | 0.793 ± 0.042 | **0.836 ± 0.038** | **Random Forest** |
| **LCX Stenosis** | 0.655 ± 0.052 | **0.709 ± 0.046** | **Random Forest** |
| **RCA Stenosis** | 0.741 ± 0.057 | **0.781 ± 0.065** | **Random Forest** |

## 4. Reproducible Code
Full plots, ROC curves, and confusion matrix visualizations are available in:
[`docs_and_evaluation/model_performance_evaluation.ipynb`](model_performance_evaluation.ipynb)
