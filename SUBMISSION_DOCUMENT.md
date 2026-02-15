# Assignment 2 Submission - Credit Risk ML Classification

**Student Name:** A. JEEVITHA
**Student ID:** 2025AA05523
**Subject:** ME7/Assignment 2: BHS Lab Issue
**Date:** February 15, 2026

---

## 1. GitHub Repository Link

**Repository URL:** https://github.com/jeevz18/CreditRisk_ML

### Repository Contents:
- ✅ Complete source code
- ✅ requirements.txt
- ✅ README.md with project details

---

## 2. Live Streamlit App Link

**Streamlit App URL:** https://creditriskml-3frd9ppjnjkgiygpeo9pur.streamlit.app/

### Deployment Details:
- ✅ Deployed using Streamlit Community Cloud
- ✅ Interactive frontend available
- ✅ All 6 ML models accessible
- ✅ Real-time credit risk predictions
- ✅ Must open an interactive frontend when clicked

---

## 3. Project Overview

### a. Problem Statement

Predict credit risk (good/bad) for loan applicants using machine learning to help financial institutions make informed lending decisions and minimize default risk.

### b. Dataset Description

**Dataset:** German Credit Data from UCI Machine Learning Repository

- **Total Records:** 1,000 customer records
- **Features:** 20 features including:
  - Demographic information (age, gender, etc.)
  - Financial status (checking account, savings, employment)
  - Loan details (amount, duration, purpose)
- **Target Variable:** Binary classification
  - Good Credit Risk = 1
  - Bad Credit Risk = 0
- **Class Distribution:** Imbalanced dataset with more good credit cases than bad

### c. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.6800 | 0.7287 | 0.8455 | 0.6643 | 0.7440 | 0.3509 |
| Decision Tree | 0.6050 | 0.6510 | 0.7798 | 0.6071 | 0.6827 | 0.1906 |
| KNN | 0.6850 | 0.6565 | 0.7421 | 0.8429 | 0.7893 | 0.1811 |
| Naive Bayes | 0.6800 | 0.7261 | 0.8016 | 0.7214 | 0.7594 | 0.2893 |
| Random Forest (Ensemble) | 0.7200 | 0.7875 | 0.8333 | 0.7500 | 0.7895 | 0.3797 |
| XGBoost (Ensemble) | 0.7100 | 0.7385 | 0.8015 | 0.7786 | 0.7899 | 0.3228 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Good baseline with balanced precision-recall. Lower recall indicates misses some bad credit cases. |
| Decision Tree | Lowest accuracy (60.5%). Prone to overfitting. Not recommended for deployment. |
| KNN | Best recall (84.3%) - catches most bad credit cases. Good for risk-averse lending. |
| Naive Bayes | Fast and simple. Decent performance with probabilistic predictions. |
| Random Forest (Ensemble) | **Best overall accuracy (72%) and AUC (78.75%)**. Best balance of metrics. **Recommended model for deployment**. |
| XGBoost (Ensemble) | Second best (71%). Strong F1 score. Good alternative to Random Forest. |

---

## 4. Technical Implementation

### Technologies Used:
- **Python 3.8+**
- **Machine Learning:** scikit-learn, XGBoost
- **Web Framework:** Streamlit
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Model Persistence:** joblib

### Project Structure:
```
CreditRisk_ML/
├── data/
│   └── credit_customers.csv          # German Credit dataset
├── models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── preprocessing_info.pkl
├── app.py                            # Streamlit web application
├── credit_risk_notebook.py           # Training script
├── credit_risk_notebook.ipynb        # Jupyter notebook
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

### Key Features:

#### 1. Data Preprocessing Pipeline
- Automatic handling of categorical variables using Label Encoding
- Feature scaling using StandardScaler
- Train-test split (80-20) with stratification

#### 2. Model Training
- 6 different ML algorithms trained and evaluated
- Comprehensive evaluation metrics:
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC Score
  - Matthews Correlation Coefficient (MCC)

#### 3. Streamlit Web Application
- **Interactive UI** for credit risk prediction
- **Model selection dropdown** - user can choose from 6 trained models
- **Input form** for customer data entry
- **Real-time predictions** with probability scores
- **Model performance metrics** displayed for each model
- **Automatic preprocessing** - handles raw input data

---

## 5. How to Run Locally

### Step 1: Clone Repository
```bash
git clone https://github.com/jeevz18/CreditRisk_ML.git
cd CreditRisk_ML
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Streamlit App
```bash
streamlit run app.py
```

### Step 4: Access Application
Open browser at: `http://localhost:8501`

---

## 6. Deployment Process

### GitHub Setup:
1. Created public GitHub repository
2. Uploaded complete project with proper folder structure
3. Ensured requirements.txt includes all dependencies

### Streamlit Cloud Deployment:
1. Signed in to https://share.streamlit.io/ with GitHub
2. Selected repository: `jeevz18/CreditRisk_ML`
3. Set main file: `app.py`
4. Deployed successfully

---

## 7. Results and Conclusion

### Best Performing Model: Random Forest
- **Accuracy:** 72%
- **AUC-ROC:** 78.75%
- **F1-Score:** 78.95%

### Key Insights:
1. Ensemble methods (Random Forest, XGBoost) outperformed individual models
2. Decision Tree showed signs of overfitting (lowest accuracy)
3. KNN achieved highest recall, making it suitable for conservative lending
4. Random Forest provides best overall balance for credit risk assessment

### Business Impact:
- Helps financial institutions automate credit risk assessment
- Reduces manual review time
- Improves lending decision accuracy by 72%
- Minimizes default risk through data-driven predictions

---

## 8. References

1. **Dataset Source:** UCI Machine Learning Repository - German Credit Data
2. **Streamlit Documentation:** https://docs.streamlit.io/
3. **Scikit-learn:** https://scikit-learn.org/
4. **XGBoost:** https://xgboost.readthedocs.io/

---

## 9. Screenshots

### Screenshot 1: Streamlit App Interface
[Screenshot to be added - showing interactive frontend]

### Screenshot 2: Model Prediction Example
[Screenshot to be added - showing prediction with input data]

### Screenshot 3: Model Performance Metrics
[Screenshot to be added - showing evaluation metrics display]

---

**End of Submission Document**
