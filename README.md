# Dry Bean Classification using Machine Learning

## Problem Statement

The project objective is to perform multi-class classification on Dry Bean Dataset using supervised machine learning algorithms. The application compares the performance of multiple classification models on the Dry Bean Dataset using various evaluation metrics and provides an interactive Streamlit web UI to upload test dataset for model evaluation.

---

## Dataset Description

**Dataset Name:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

**Dataset Type:** Multi-Class Classification

### Dataset Information

- Number of Instances: 13,611
- Number of Features: 16
- Number of Classes: 7

### Bean Classes

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

The dataset consists of geometric characteristics extracted from images of dry beans. These features are used to classify each bean into one of the seven bean varieties.

---

## GitHub Repository

**Repository Link**

https://github.com/2025ac05309-rohit/2025ac06309-ML-Assignment2

---

## Live Streamlit Application

**Streamlit URL**

https://2025ac05309.streamlit.app/

---

## Models Used

The following machine learning models were implemented:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Random Forest Classifier

---

## Evaluation Metrics

The following metrics were used for evaluating every model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison

| ML Model Name            | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------------ | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression      |   0.9195 | 0.9935 |    0.9201 | 0.9195 |   0.9197 | 0.9028 |
| Decision Tree            |   0.8955 | 0.9357 |    0.8954 | 0.8955 |   0.8953 | 0.8737 |
| kNN                      |   0.9155 | 0.9811 |    0.9163 | 0.9155 |   0.9157 | 0.8978 |
| Naive Bayes              |   0.7630 | 0.9644 |    0.7647 | 0.7630 |   0.7607 | 0.7143 |
| Random Forest (Ensemble) |   0.9169 | 0.9905 |    0.9170 | 0.9169 |   0.9169 | 0.8995 |

---

## Model Observations

| Model                    | Observation about model performance                                                                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved the highest overall performance with **91.95% accuracy**, **0.9935 AUC**, and the highest MCC (0.9028), indicating excellent classification performance on the Dry Bean dataset. |
| Decision Tree            | Produced good classification results with **89.55% accuracy**. It effectively captured non-linear relationships but performed slightly lower than the ensemble and linear models.         |
| kNN                      | Achieved **91.55% accuracy** and competitive precision and F1-score, demonstrating strong performance based on similarity between bean samples.                                           |
| Naive Bayes              | Recorded the lowest performance (**76.30% accuracy**). The assumption of feature independence likely reduced its effectiveness for this dataset.                                          |
| Random Forest (Ensemble) | Achieved **91.69% accuracy** with excellent AUC (0.9905). It produced stable and reliable predictions, performing almost as well as Logistic Regression.                                  |
| **Overall Winner**       | **Logistic Regression** achieved the best overall performance across Accuracy, AUC, Precision, F1 Score, and MCC, making it the best-performing model for this dataset.                   |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/2025ac05309-rohit/2025ac06309-ML-Assignment2
```

Move into the project directory:

```bash
cd 2025ac06309-ML-Assignment2
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train and Evaluate All Models

```bash
python run_all_models.py
```

### Launch the Streamlit Application

```bash
streamlit run app.py
```

---

## Streamlit Features

The deployed Streamlit application includes:

- Upload test dataset (CSV)
- Machine learning model selection
- Model evaluation
- Performance metrics display
- Confusion Matrix
- Classification Report

---

## Author

**ROHIT BANIK**
**BITS ID:** 2025AC06309
Machine Learning Assignment – 2
M.Tech (Artificial Intelligence & Machine Learning)
BITS Pilani WILP
