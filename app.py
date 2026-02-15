"""
Credit Risk Classification - Streamlit App
ML Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page config for compact view
st.set_page_config(page_title="Credit Risk ML", layout="wide")

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    models = {}
    try:
        with open('models/logistic_regression_model.pkl', 'rb') as f:
            models['Logistic Regression'] = pickle.load(f)
        with open('models/decision_tree_model.pkl', 'rb') as f:
            models['Decision Tree'] = pickle.load(f)
        with open('models/knn_model.pkl', 'rb') as f:
            models['K-Nearest Neighbors'] = pickle.load(f)
        with open('models/naive_bayes_model.pkl', 'rb') as f:
            models['Naive Bayes'] = pickle.load(f)
        with open('models/random_forest_model.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        with open('models/xgboost_model.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)

        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)

        with open('models/preprocessing_info.pkl', 'rb') as f:
            preprocessing_info = pickle.load(f)

        try:
            with open('models/target_encoder.pkl', 'rb') as f:
                target_encoder = pickle.load(f)
        except:
            target_encoder = None

        return models, scaler, feature_names, label_encoders, preprocessing_info, target_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

def preprocess_data(df, label_encoders, target_col='class', is_prediction=False):
    """Preprocess raw data similar to training"""
    df_clean = df.copy()
    df_clean = df_clean.dropna()

    has_target = target_col in df_clean.columns
    if has_target and not is_prediction:
        y = df_clean[target_col]
        X = df_clean.drop(target_col, axis=1)
    else:
        y = None
        X = df_clean.drop(target_col, axis=1) if target_col in df_clean.columns else df_clean

    for col in label_encoders.keys():
        if col in X.columns:
            le = label_encoders[col]
            X[col] = X[col].astype(str)
            known_classes = set(le.classes_)
            X[col] = X[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            X[col] = le.transform(X[col])

    return X, y

# Load models
models, scaler, feature_names, label_encoders, preprocessing_info, target_encoder = load_models()

if models is None:
    st.error("Please run the training script first to generate model files.")
    st.stop()

# Title
st.title("Credit Risk Classification System")

# Two-column layout
col_left, col_right = st.columns([1, 2])

# LEFT COLUMN: Download + Upload
with col_left:
    # Download test data
    st.subheader("📥 Download Test Data")
    try:
        test_data = pd.read_csv('data/test_data_with_labels.csv')
        csv_data = test_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Test Data",
            data=csv_data,
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"200 test samples")
    except:
        st.warning("Test data not found")

    st.write("")

    # Upload CSV
    st.subheader("📤 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], label_visibility="collapsed")

    if uploaded_file:
        st.success("✓ File uploaded")

# RIGHT COLUMN: Model Selection + Results
with col_right:
    if uploaded_file is not None:
        # Model selection at the top
        st.subheader("🤖 Select Model")
        selected_model = st.selectbox(
            "Choose a machine learning model:",
            list(models.keys()),
            label_visibility="collapsed"
        )

        # Predict button
        predict_btn = st.button("🎯 Predict", type="primary")

        if predict_btn:
            # Read and preprocess
            df = pd.read_csv(uploaded_file)
            target_col = preprocessing_info['target_col']
            has_labels = target_col in df.columns

            with st.spinner("Processing..."):
                X_processed, y_true = preprocess_data(df, label_encoders, target_col, is_prediction=not has_labels)
                X_processed = X_processed[feature_names]
                X_test_scaled = scaler.transform(X_processed)

                # Predict
                model = models[selected_model]
                predictions = model.predict(X_test_scaled)
                probabilities = model.predict_proba(X_test_scaled)[:, 1]

            # Prediction summary (top)
            st.subheader(f"Results: {selected_model}")

        col1, col2, col3 = st.columns(3)

        if target_encoder is not None:
            pred_labels = target_encoder.inverse_transform(predictions)
            pred_counts = pd.Series(pred_labels).value_counts()
            good_count = pred_counts.get('good', 0)
            bad_count = pred_counts.get('bad', 0)
        else:
            good_count = (predictions == 1).sum()
            bad_count = (predictions == 0).sum()

        col1.metric("Good Credit", good_count, delta=f"{good_count/len(predictions)*100:.1f}%")
        col2.metric("Bad Credit", bad_count, delta=f"{bad_count/len(predictions)*100:.1f}%")
        col3.metric("Total", len(predictions))

        # Metrics (if labels exist)
        if has_labels:
            if y_true.dtype == 'object':
                y_true_encoded = target_encoder.transform(y_true) if target_encoder else LabelEncoder().fit_transform(y_true)
            else:
                y_true_encoded = y_true.values

            acc = accuracy_score(y_true_encoded, predictions)
            prec = precision_score(y_true_encoded, predictions, zero_division=0)
            rec = recall_score(y_true_encoded, predictions, zero_division=0)
            f1 = f1_score(y_true_encoded, predictions, zero_division=0)
            auc = roc_auc_score(y_true_encoded, probabilities)
            mcc = matthews_corrcoef(y_true_encoded, predictions)

            st.subheader("📊 Evaluation Metrics")
            met1, met2, met3, met4, met5, met6 = st.columns(6)
            met1.metric("Accuracy", f"{acc:.3f}")
            met2.metric("Precision", f"{prec:.3f}")
            met3.metric("Recall", f"{rec:.3f}")
            met4.metric("F1", f"{f1:.3f}")
            met5.metric("AUC", f"{auc:.3f}")
            met6.metric("MCC", f"{mcc:.3f}")

            # Confusion Matrix
            cm = confusion_matrix(y_true_encoded, predictions)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')

            # Show confusion matrix in a compact column
            cm_col1, cm_col2 = st.columns([1, 2])
            with cm_col1:
                st.pyplot(fig, use_container_width=True)
            plt.close()

        # Predictions table (compact)
        st.subheader("🎯 Predictions")

        if target_encoder is not None:
            pred_labels = target_encoder.inverse_transform(predictions)
            results = pd.DataFrame({
                'Prediction': pred_labels,
                'Probability': probabilities.round(3)
            })
        else:
            results = pd.DataFrame({
                'Prediction': predictions,
                'Probability': probabilities.round(3)
            })

        if has_labels:
            results['Actual'] = y_true.values
            results['Match'] = ['✓' if predictions[i] == y_true_encoded[i] else '✗' for i in range(len(predictions))]

        # Show first 10 rows
        st.dataframe(results.head(10), use_container_width=True, height=250)

        # Download button
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        else:
            st.info("Click 'Predict' to see results")
    else:
        st.info("👈 Upload a CSV file to get started")
