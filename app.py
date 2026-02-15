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

# Page title
st.title("Credit Risk Classification System")
st.write("Upload your customer data and select a model to predict credit risk")

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
    """
    Preprocess raw data similar to training

    Args:
        df: Raw dataframe
        label_encoders: Dictionary of fitted LabelEncoders
        target_col: Name of target column
        is_prediction: If True, no target column processing

    Returns:
        X_processed: Processed features
        y: Target (if present), else None
    """
    df_clean = df.copy()

    # Handle missing values
    df_clean = df_clean.dropna()

    # Separate target if present
    has_target = target_col in df_clean.columns
    if has_target and not is_prediction:
        y = df_clean[target_col]
        X = df_clean.drop(target_col, axis=1)
    else:
        y = None
        X = df_clean.drop(target_col, axis=1) if target_col in df_clean.columns else df_clean

    # Encode categorical variables using the same encoders from training
    for col in label_encoders.keys():
        if col in X.columns:
            le = label_encoders[col]
            # Handle unseen categories by replacing with most frequent
            X[col] = X[col].astype(str)
            # Transform known values, unknown values become -1
            known_classes = set(le.classes_)
            X[col] = X[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            X[col] = le.transform(X[col])

    return X, y

# Load models
models, scaler, feature_names, label_encoders, preprocessing_info, target_encoder = load_models()

if models is None:
    st.error("Please run the training script first to generate model files.")
    st.stop()

st.write("---")

# Sidebar for downloading test data
st.sidebar.header("Download Test Data")
st.sidebar.write("Download sample test data to try the app:")

# Load test data for download
try:
    test_data_with_labels = pd.read_csv('test_data_with_labels.csv')
    test_data_features = pd.read_csv('test_data_features.csv')

    # Download buttons
    csv_with_labels = test_data_with_labels.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Test Data (with labels)",
        data=csv_with_labels,
        file_name="test_data_with_labels.csv",
        mime="text/csv"
    )

    csv_features = test_data_features.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Test Data (features only)",
        data=csv_features,
        file_name="test_data_features.csv",
        mime="text/csv"
    )

    st.sidebar.write(f"Test samples available: {len(test_data_features)}")
    st.sidebar.info("📝 Test data is RAW (before preprocessing). The app will handle preprocessing automatically.")
except:
    st.sidebar.warning("Test data files not found")

st.sidebar.write("---")
st.sidebar.info("Upload RAW CSV data. The app will automatically handle preprocessing and feature engineering.")

# Step 1: Upload CSV file
st.header("Step 1: Upload CSV File")
st.write("Upload your raw customer data. The app will automatically preprocess it.")
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    # Read the uploaded file
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully! Shape: {df.shape}")

        # Show preview
        st.write("Preview of uploaded data (RAW):")
        st.dataframe(df.head())

        # Check if target column exists
        target_col = preprocessing_info['target_col']
        has_labels = target_col in df.columns

        if has_labels:
            st.info(f"Labels detected in column '{target_col}' - Will calculate evaluation metrics")
        else:
            st.warning("No labels found - Only predictions will be shown")

        st.write("---")

        # Preprocess the data
        st.header("Data Preprocessing")
        with st.spinner("Preprocessing data..."):
            X_processed, y_true = preprocess_data(df, label_encoders, target_col, is_prediction=not has_labels)

            st.success("✓ Data preprocessed successfully!")
            st.write(f"- Missing values handled")
            st.write(f"- {len(label_encoders)} categorical features encoded")
            st.write(f"- Processed shape: {X_processed.shape}")

        # Validate features match training
        if X_processed.shape[1] != len(feature_names):
            st.error(f"Error: Expected {len(feature_names)} features after preprocessing, got {X_processed.shape[1]}")
            st.write(f"Expected features: {feature_names}")
            st.write(f"Got features: {list(X_processed.columns)}")
            st.stop()

        # Ensure column order matches training
        X_processed = X_processed[feature_names]

        st.write("---")

        # Step 2: Select model
        st.header("Step 2: Select Model")
        selected_model = st.selectbox(
            "Choose a machine learning model:",
            list(models.keys())
        )

        st.write("---")

        # Step 3: Make predictions
        st.header("Step 3: Make Predictions")

        if st.button("Predict"):
            # Scale the data
            X_test_scaled = scaler.transform(X_processed)

            # Get the selected model
            model = models[selected_model]

            # Make predictions
            predictions = model.predict(X_test_scaled)
            probabilities = model.predict_proba(X_test_scaled)[:, 1]

            st.success("Predictions completed!")

            # Show predictions
            st.subheader("Prediction Results")

            # Decode predictions if target encoder exists
            if target_encoder is not None:
                pred_labels = target_encoder.inverse_transform(predictions)
                results = pd.DataFrame({
                    'Prediction': pred_labels,
                    'Prediction (Encoded)': predictions,
                    'Risk Probability': probabilities
                })
            else:
                results = pd.DataFrame({
                    'Prediction': predictions,
                    'Risk Probability': probabilities
                })

            if has_labels:
                # Encode y_true if it's categorical
                if y_true.dtype == 'object':
                    if target_encoder is not None:
                        y_true_encoded = target_encoder.transform(y_true)
                    else:
                        # Create temporary encoder
                        temp_encoder = LabelEncoder()
                        y_true_encoded = temp_encoder.fit_transform(y_true)
                else:
                    y_true_encoded = y_true.values

                results['Actual'] = y_true.values
                results['Correct'] = (predictions == y_true_encoded)

            st.dataframe(results)

            # Download predictions
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

            st.write("---")

            # Show prediction summary
            st.subheader("Prediction Summary")
            col1, col2 = st.columns(2)

            if target_encoder is not None:
                # Show actual class names
                pred_counts = pd.Series(pred_labels).value_counts()
                for class_name, count in pred_counts.items():
                    if 'bad' in str(class_name).lower():
                        col1.metric(f"Bad Credit Risk", count)
                    else:
                        col2.metric(f"Good Credit Risk", count)
            else:
                pred_counts = pd.Series(predictions).value_counts()
                col1.metric("Class 0", pred_counts.get(0, 0))
                col2.metric("Class 1", pred_counts.get(1, 0))

            # If labels exist, calculate and display metrics
            if has_labels:
                st.write("---")
                st.header("Evaluation Metrics")

                # Calculate metrics using encoded values
                accuracy = accuracy_score(y_true_encoded, predictions)
                precision = precision_score(y_true_encoded, predictions, zero_division=0)
                recall = recall_score(y_true_encoded, predictions, zero_division=0)
                f1 = f1_score(y_true_encoded, predictions, zero_division=0)
                auc = roc_auc_score(y_true_encoded, probabilities)
                mcc = matthews_corrcoef(y_true_encoded, predictions)

                # Display metrics
                col1, col2, col3 = st.columns(3)

                col1.metric("Accuracy", f"{accuracy:.4f}")
                col1.metric("Precision", f"{precision:.4f}")

                col2.metric("Recall", f"{recall:.4f}")
                col2.metric("F1 Score", f"{f1:.4f}")

                col3.metric("AUC Score", f"{auc:.4f}")
                col3.metric("MCC Score", f"{mcc:.4f}")

                st.write("---")

                # Confusion Matrix
                st.header("Confusion Matrix")

                cm = confusion_matrix(y_true_encoded, predictions)

                # Get class names for labels
                if target_encoder is not None:
                    class_names = target_encoder.classes_
                else:
                    class_names = ['Class 0', 'Class 1']

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names,
                           yticklabels=class_names)
                ax.set_title(f'Confusion Matrix - {selected_model}')
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')

                st.pyplot(fig)

                st.write("---")

                # Classification Report
                st.header("Classification Report")

                report = classification_report(y_true_encoded, predictions,
                                              target_names=class_names,
                                              output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.write("**Error details:**")
        st.code(traceback.format_exc())
        st.write("Please make sure your CSV file has the correct format (RAW data with original column names).")

else:
    st.info("Please upload a CSV file to get started")

    st.write("**Expected CSV format:**")
    st.write("Upload RAW customer data with original column names from the dataset.")
    st.write(f"The app expects the following columns (before preprocessing):")

    # Show example of expected columns
    expected_cols = ['checking_status', 'duration', 'credit_history', 'purpose',
                     'credit_amount', 'savings_status', 'employment',
                     'installment_commitment', 'personal_status', 'other_parties',
                     'residence_since', 'property_magnitude', 'age',
                     'other_payment_plans', 'housing', 'existing_credits',
                     'job', 'num_dependents', 'own_telephone', 'foreign_worker']

    st.write(", ".join(expected_cols))
    st.write("Optional: 'class' column for evaluation (values: 'good' or 'bad')")

    st.write("\n**Download sample test data from the sidebar to try the app!**")

# Footer
st.write("---")
st.write("Credit Risk Classification - ML Assignment 2")
st.caption("The app automatically handles data preprocessing and feature engineering")
