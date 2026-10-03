# Breast Cancer Diagnostic Classification

A machine learning project for classifying breast tumours as **malignant** or **benign** using numerical measurements of cell nuclei from digitised fine needle aspirate images.

> **Note:** This is an educational machine learning project and is not a clinical diagnostic tool.

## Project Overview

This project treats breast cancer diagnosis as a binary classification problem.

Three classical machine learning models are compared:

* Logistic Regression
* Decision Tree
* Random Forest

The workflow includes exploratory data analysis, model training, evaluation, cross-validation, confusion matrices, ROC curves, and Random Forest feature importance.

## Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset available through `scikit-learn`.

* 569 samples
* 30 numerical features
* 212 malignant samples
* 357 benign samples
* No missing values

The features describe characteristics of cell nuclei, including:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

The target labels are transformed as:

```text
1 = Malignant
0 = Benign
```

## Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Exploratory Data Analysis
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Confusion Matrices + ROC Curves
   ↓
Random Forest Feature Importance
```

## Machine Learning Models

### Logistic Regression

Logistic Regression is combined with `StandardScaler` using a scikit-learn pipeline.

```python
LogisticRegression(max_iter=1000, random_state=42)
```

### Decision Tree

The Decision Tree uses a maximum depth of 4.

```python
DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
```

### Random Forest

The Random Forest uses 300 decision trees.

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
```

## Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix

The notebook also performs **5-fold Stratified Cross-Validation**.

## Results

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC | 5-fold CV Accuracy |
| ------------------- | -------: | --------: | -----: | -------: | ------: | -----------------: |
| Logistic Regression |   96.49% |    97.50% | 92.86% |   95.12% |  0.9960 |             97.37% |
| Decision Tree       |   91.23% |    94.44% | 80.95% |   87.18% |  0.8770 |             92.27% |
| Random Forest       |   97.37% |   100.00% | 92.86% |   96.30% |  0.9944 |             95.26% |

These results correspond to the notebook's 80/20 test split. Cross-validation accuracy is reported separately.

### Confusion Matrices

The confusion matrices use the order:

```text
[[TN, FP],
 [FN, TP]]
```

**Logistic Regression**

```text
[[71, 1],
 [ 3, 39]]
```

**Decision Tree**

```text
[[70, 2],
 [ 8, 34]]
```

**Random Forest**

```text
[[72, 0],
 [ 3, 39]]
```

## Random Forest Feature Importance

The 10 highest Random Forest feature importances reported in the notebook are:

| Feature              | Importance |
| -------------------- | ---------: |
| worst perimeter      |     0.1431 |
| worst area           |     0.1424 |
| worst concave points |     0.1130 |
| mean concave points  |     0.0887 |
| worst radius         |     0.0826 |
| mean radius          |     0.0610 |
| mean perimeter       |     0.0529 |
| mean area            |     0.0438 |
| mean concavity       |     0.0398 |
| area error           |     0.0322 |

These values represent model feature importance and should not be interpreted as evidence that these features independently cause cancer.

## Project Structure

```text
Breast_Cancer_Classification/
│
├── Task2_Breast_Cancer_Classification.ipynb
├── README.md
└── requirements.txt
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Reproducibility

The project uses:

```python
RANDOM_STATE = 42
```

The train/test split and machine learning models therefore use a fixed random state for reproducibility.

To install the required libraries:

```bash
pip install -r requirements.txt
```

Then open:

```text
Task2_Breast_Cancer_Classification.ipynb
```

and run the notebook cells.

## Disclaimer

This project is intended for educational and machine learning practice purposes. The dataset is a benchmark dataset and the resulting models should not be used for medical diagnosis or clinical decision-making.
