# Assignment 2 Submission - Credit Risk ML Classification

**Student Name:** A. JEEVITHA
**Student ID:** 2025AA05523
**Date:** February 15, 2026

---

## 1. GitHub Repository Link

**Repository URL:** https://github.com/jeevz18/CreditRisk_ML

### Contents:
- Complete source code (credit_risk_notebook.py, credit_risk_notebook.ipynb, app.py)
- requirements.txt
- A clear README.md

---

## 2. Live Streamlit App Link

**Streamlit App URL:** https://creditriskml-3frd9ppjnjkgiygpeo9pur.streamlit.app/

- Deployed using Streamlit Community Cloud
- Must open an interactive frontend when clicked

---

## 3. Screenshot

Screenshot of assignment execution on BITS Virtual Lab:

[Screenshot will be inserted here after taking it]

---

## 4. GitHub README Content (Section 3 - Step 5)

### a. Problem Statement

Predict credit risk (good/bad) for loan applicants using machine learning to help financial institutions make informed lending decisions.

### b. Dataset Description

German Credit Data - 1000 customer records with 20 features for binary credit risk classification.

### c. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.6800 | 0.7287 | 0.8455 | 0.6643 | 0.7440 | 0.3509 |
| Decision Tree | 0.6050 | 0.6510 | 0.7798 | 0.6071 | 0.6827 | 0.1906 |
| KNN | 0.6850 | 0.6565 | 0.7421 | 0.8429 | 0.7893 | 0.1811 |
| Naive Bayes | 0.6800 | 0.7261 | 0.8016 | 0.7214 | 0.7594 | 0.2893 |
| Random Forest (Ensemble) | 0.7200 | 0.7875 | 0.8333 | 0.7500 | 0.7895 | 0.3797 |
| XGBoost (Ensemble) | 0.7100 | 0.7385 | 0.8015 | 0.7786 | 0.7899 | 0.3228 |

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Good baseline with balanced precision-recall. Lower recall indicates misses some bad credit cases. |
| Decision Tree | Lowest accuracy (60.5%). Prone to overfitting. Not recommended. |
| KNN | Best recall (84.3%) - catches most bad credit cases. Good for risk-averse lending. |
| Naive Bayes | Fast and simple. Decent performance with probabilistic predictions. |
| Random Forest (Ensemble) | Best overall accuracy (72%) and AUC (78.75%). Best balance. Recommended model. |
| XGBoost (Ensemble) | Second best (71%). Strong F1 score. Good alternative to Random Forest. |

---

**End of Submission**
