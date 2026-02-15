"""
Credit Risk Classification - Complete ML Pipeline
ML Assignment 2 - Binary Classification

Dataset: Credit Risk Customers (Kaggle)
Features: 20 customer features
Target: Binary (0 = Low Risk, 1 = High Risk)
Models: 6 ML algorithms
"""

# =============================================================================
# STEP 1: IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import pickle
import os

print("="*70)
print("  CREDIT RISK CLASSIFICATION - ML ASSIGNMENT 2")
print("="*70)
print("\nLibraries imported successfully!")

# =============================================================================
# STEP 2: LOAD DATASET
# =============================================================================

print("\n" + "="*70)
print("LOADING DATASET")
print("="*70)

# Load the dataset
try:
    dataset_path = 'data/credit_customers.csv'

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset file not found at: {dataset_path}")
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(dataset_path)
    print(f"[OK] Dataset loaded from: {dataset_path}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

print(f"\nDataset Shape: {df.shape}")
print(f"Total Features: {df.shape[1]}")
print(f"Total Records: {df.shape[0]}")

# =============================================================================
# STEP 3: INITIAL DATA EXPLORATION
# =============================================================================

print("\n" + "="*70)
print("DATA EXPLORATION")
print("="*70)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(list(df.columns))

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# =============================================================================
# STEP 4: DATA PREPROCESSING
# =============================================================================

print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Target variable
target_col = 'class'

# Make a copy
df_clean = df.copy()

# 1. Handle missing values
print("\n1. Handling missing values...")
df_clean = df_clean.dropna()
print(f"   Rows after cleaning: {len(df_clean)}")

# 2. Encode categorical variables
print("\n2. Encoding categorical variables...")
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove(target_col)  # Remove target from features

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le
    print(f"   [OK] Encoded: {col}")

# 3. Encode target variable
print(f"\n3. Encoding target variable '{target_col}'...")
target_encoder = LabelEncoder()
df_clean[target_col] = target_encoder.fit_transform(df_clean[target_col])
print(f"   Mapping: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

print(f"\nTarget distribution:")
print(df_clean[target_col].value_counts())

# Display sample of preprocessed data
print("\nFirst 5 rows after preprocessing:")
print(df_clean.head())

# =============================================================================
# STEP 5: SEPARATE FEATURES AND TARGET
# =============================================================================

print("\n" + "="*70)
print("SEPARATING FEATURES AND TARGET")
print("="*70)

X = df_clean.drop(target_col, axis=1)
y = df_clean[target_col]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {list(X.columns)}")

# =============================================================================
# STEP 6: CORRELATION MATRIX AND FEATURE ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("CORRELATION MATRIX & FEATURE ANALYSIS")
print("="*70)

# Create correlation matrix
correlation_matrix = X.corr()

print("\nCorrelation Matrix Shape:", correlation_matrix.shape)

# Find features highly correlated with each other
print("\n--- High Correlation Pairs (|correlation| > 0.7) ---")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
else:
    print("  No highly correlated feature pairs found (threshold: 0.7)")

# Correlation with target
print("\n--- Feature Correlation with Target ---")
# Create a dataframe with features and target for correlation
df_with_target = X.copy()
df_with_target['target'] = y

target_correlation = df_with_target.corr()['target'].drop('target').sort_values(ascending=False)
print("\nTop 10 positively correlated features:")
print(target_correlation.head(10))
print("\nTop 10 negatively correlated features:")
print(target_correlation.tail(10))

# Visualize correlation matrix INCLUDING TARGET
print("\nGenerating correlation heatmap (including target)...")
plt.figure(figsize=(18, 16))
# Use df_with_target which includes the target column
correlation_with_target = df_with_target.corr()
sns.heatmap(correlation_with_target, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix (Including Target)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
print("[OK] Correlation matrix saved as 'correlation_matrix.png'")
plt.close()

# Feature selection based on correlation with target
print("\n--- FEATURE SELECTION BASED ON TARGET CORRELATION ---")
# Features with strong correlation (|r| > 0.2) with target
strong_features = target_correlation[abs(target_correlation) > 0.2]
print(f"\nFeatures with strong correlation to target (|r| > 0.2): {len(strong_features)} features")
print(strong_features.sort_values(ascending=False))

# Features with weak correlation (|r| < 0.05) - potential candidates for removal
weak_features = target_correlation[abs(target_correlation) < 0.05]
if len(weak_features) > 0:
    print(f"\n⚠ Features with weak correlation to target (|r| < 0.05): {len(weak_features)} features")
    print("These features might be candidates for removal:")
    print(weak_features)

# Feature importance based on variance
print("\n--- Feature Variance Analysis ---")
feature_variance = X.var().sort_values(ascending=False)
print("\nTop 10 features by variance:")
print(feature_variance.head(10))
print("\nBottom 10 features by variance:")
print(feature_variance.tail(10))

# =============================================================================
# STEP 7: TRAIN-TEST SPLIT (80-20)
# =============================================================================

print("\n" + "="*70)
print("TRAIN-TEST SPLIT (80-20)")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

print("\nClass distribution in training set:")
print(y_train.value_counts())
print("\nClass distribution in testing set:")
print(y_test.value_counts())

# =============================================================================
# STEP 8: FEATURE SCALING
# =============================================================================

print("\n" + "="*70)
print("FEATURE SCALING")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler (mean=0, std=1)")
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Testing set shape: {X_test_scaled.shape}")

# =============================================================================
# HELPER FUNCTION: PLOT CONFUSION MATRIX AND CLASSIFICATION REPORT
# =============================================================================

def plot_model_evaluation(model, model_name, X_train, y_train, X_test, y_test):
    """
    Plot confusion matrices and display classification reports for train and test sets
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Create figure with 2 subplots for confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training Confusion Matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
    axes[0].set_title(f'{model_name} - Training Set\nConfusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Testing Confusion Matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
    axes[1].set_title(f'{model_name} - Testing Set\nConfusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=100, bbox_inches='tight')
    print(f"[OK] Confusion matrices saved as '{model_name.lower().replace(' ', '_')}_confusion_matrix.png'")
    plt.close()

    # Classification Reports
    print(f"\n--- TRAINING SET CLASSIFICATION REPORT ---")
    print(classification_report(y_train, y_train_pred, target_names=['Bad Credit', 'Good Credit'], zero_division=0))

    print(f"\n--- TESTING SET CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_test_pred, target_names=['Bad Credit', 'Good Credit'], zero_division=0))

# =============================================================================
# STEP 9: MODEL 1 - LOGISTIC REGRESSION
# =============================================================================

print("\n" + "="*70)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*70)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, zero_division=0)
lr_recall = recall_score(y_test, lr_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)
lr_auc = roc_auc_score(y_test, lr_pred_proba)
lr_mcc = matthews_corrcoef(y_test, lr_pred)

print(f"\nTEST SET METRICS:")
print(f"  Accuracy:  {lr_accuracy:.4f}")
print(f"  AUC:       {lr_auc:.4f}")
print(f"  Precision: {lr_precision:.4f}")
print(f"  Recall:    {lr_recall:.4f}")
print(f"  F1 Score:  {lr_f1:.4f}")
print(f"  MCC:       {lr_mcc:.4f}")

# Plot confusion matrix and classification report
print("\n")
plot_model_evaluation(lr_model, "Logistic Regression", X_train_scaled, y_train, X_test_scaled, y_test)

# =============================================================================
# STEP 10: MODEL 2 - DECISION TREE
# =============================================================================

print("\n" + "="*70)
print("MODEL 2: DECISION TREE")
print("="*70)

dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)

dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_pred_proba = dt_model.predict_proba(X_test_scaled)[:, 1]

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, zero_division=0)
dt_recall = recall_score(y_test, dt_pred, zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, zero_division=0)
dt_auc = roc_auc_score(y_test, dt_pred_proba)
dt_mcc = matthews_corrcoef(y_test, dt_pred)

print(f"\nTEST SET METRICS:")
print(f"  Accuracy:  {dt_accuracy:.4f}")
print(f"  AUC:       {dt_auc:.4f}")
print(f"  Precision: {dt_precision:.4f}")
print(f"  Recall:    {dt_recall:.4f}")
print(f"  F1 Score:  {dt_f1:.4f}")
print(f"  MCC:       {dt_mcc:.4f}")

# Plot confusion matrix and classification report
print("\n")
plot_model_evaluation(dt_model, "Decision Tree", X_train_scaled, y_train, X_test_scaled, y_test)

# =============================================================================
# STEP 11: MODEL 3 - K-NEAREST NEIGHBORS
# =============================================================================

print("\n" + "="*70)
print("MODEL 3: K-NEAREST NEIGHBORS")
print("="*70)

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='minkowski',
    p=2
)

knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_pred_proba = knn_model.predict_proba(X_test_scaled)[:, 1]

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred, zero_division=0)
knn_recall = recall_score(y_test, knn_pred, zero_division=0)
knn_f1 = f1_score(y_test, knn_pred, zero_division=0)
knn_auc = roc_auc_score(y_test, knn_pred_proba)
knn_mcc = matthews_corrcoef(y_test, knn_pred)

print(f"\nTEST SET METRICS:")
print(f"  Accuracy:  {knn_accuracy:.4f}")
print(f"  AUC:       {knn_auc:.4f}")
print(f"  Precision: {knn_precision:.4f}")
print(f"  Recall:    {knn_recall:.4f}")
print(f"  F1 Score:  {knn_f1:.4f}")
print(f"  MCC:       {knn_mcc:.4f}")

# Plot confusion matrix and classification report
print("\n")
plot_model_evaluation(knn_model, "K-Nearest Neighbors", X_train_scaled, y_train, X_test_scaled, y_test)

# =============================================================================
# STEP 12: MODEL 4 - GAUSSIAN NAIVE BAYES
# =============================================================================

print("\n" + "="*70)
print("MODEL 4: GAUSSIAN NAIVE BAYES")
print("="*70)

gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)
gnb_pred = gnb_model.predict(X_test_scaled)
gnb_pred_proba = gnb_model.predict_proba(X_test_scaled)[:, 1]

gnb_accuracy = accuracy_score(y_test, gnb_pred)
gnb_precision = precision_score(y_test, gnb_pred, zero_division=0)
gnb_recall = recall_score(y_test, gnb_pred, zero_division=0)
gnb_f1 = f1_score(y_test, gnb_pred, zero_division=0)
gnb_auc = roc_auc_score(y_test, gnb_pred_proba)
gnb_mcc = matthews_corrcoef(y_test, gnb_pred)

print(f"\nTEST SET METRICS:")
print(f"  Accuracy:  {gnb_accuracy:.4f}")
print(f"  AUC:       {gnb_auc:.4f}")
print(f"  Precision: {gnb_precision:.4f}")
print(f"  Recall:    {gnb_recall:.4f}")
print(f"  F1 Score:  {gnb_f1:.4f}")
print(f"  MCC:       {gnb_mcc:.4f}")

# Plot confusion matrix and classification report
print("\n")
plot_model_evaluation(gnb_model, "Naive Bayes", X_train_scaled, y_train, X_test_scaled, y_test)

# =============================================================================
# STEP 13: MODEL 5 - RANDOM FOREST
# =============================================================================

print("\n" + "="*70)
print("MODEL 5: RANDOM FOREST (ENSEMBLE)")
print("="*70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
rf_auc = roc_auc_score(y_test, rf_pred_proba)
rf_mcc = matthews_corrcoef(y_test, rf_pred)

print(f"\nTEST SET METRICS:")
print(f"  Accuracy:  {rf_accuracy:.4f}")
print(f"  AUC:       {rf_auc:.4f}")
print(f"  Precision: {rf_precision:.4f}")
print(f"  Recall:    {rf_recall:.4f}")
print(f"  F1 Score:  {rf_f1:.4f}")
print(f"  MCC:       {rf_mcc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot confusion matrix and classification report
print("\n")
plot_model_evaluation(rf_model, "Random Forest", X_train_scaled, y_train, X_test_scaled, y_test)

# =============================================================================
# STEP 14: MODEL 6 - XGBOOST
# =============================================================================

print("\n" + "="*70)
print("MODEL 6: XGBOOST (ENSEMBLE)")
print("="*70)

from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred, zero_division=0)
xgb_recall = recall_score(y_test, xgb_pred, zero_division=0)
xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
xgb_mcc = matthews_corrcoef(y_test, xgb_pred)

print(f"\nTEST SET METRICS:")
print(f"  Accuracy:  {xgb_accuracy:.4f}")
print(f"  AUC:       {xgb_auc:.4f}")
print(f"  Precision: {xgb_precision:.4f}")
print(f"  Recall:    {xgb_recall:.4f}")
print(f"  F1 Score:  {xgb_f1:.4f}")
print(f"  MCC:       {xgb_mcc:.4f}")

# Plot confusion matrix and classification report
print("\n")
plot_model_evaluation(xgb_model, "XGBoost", X_train_scaled, y_train, X_test_scaled, y_test)

# =============================================================================
# STEP 15: MODEL COMPARISON
# =============================================================================

print("\n" + "="*90)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*90)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors',
              'Naive Bayes', 'Random Forest', 'XGBoost'],
    'Accuracy': [lr_accuracy, dt_accuracy, knn_accuracy, gnb_accuracy, rf_accuracy, xgb_accuracy],
    'AUC': [lr_auc, dt_auc, knn_auc, gnb_auc, rf_auc, xgb_auc],
    'Precision': [lr_precision, dt_precision, knn_precision, gnb_precision, rf_precision, xgb_precision],
    'Recall': [lr_recall, dt_recall, knn_recall, gnb_recall, rf_recall, xgb_recall],
    'F1': [lr_f1, dt_f1, knn_f1, gnb_f1, rf_f1, xgb_f1],
    'MCC': [lr_mcc, dt_mcc, knn_mcc, gnb_mcc, rf_mcc, xgb_mcc]
})

print(comparison_df.to_string(index=False))
print("="*90)

# Calculate rankings
comparison_df['Average'] = comparison_df[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']].mean(axis=1)
ranking = comparison_df.sort_values('Average', ascending=False).reset_index(drop=True)
ranking.insert(0, 'Rank', range(1, len(ranking) + 1))

print("\n" + "="*70)
print("OVERALL RANKING")
print("="*70)
print(ranking[['Rank', 'Model', 'Average']].to_string(index=False))

# =============================================================================
# STEP 16: SAVE MODELS AND DATA
# =============================================================================

print("\n" + "="*70)
print("SAVING MODELS AND DATA")
print("="*70)

# Create models directory
os.makedirs('models', exist_ok=True)

# Save all models
models_dict = {
    'logistic_regression_model.pkl': lr_model,
    'decision_tree_model.pkl': dt_model,
    'knn_model.pkl': knn_model,
    'naive_bayes_model.pkl': gnb_model,
    'random_forest_model.pkl': rf_model,
    'xgboost_model.pkl': xgb_model
}

for filename, model in models_dict.items():
    with open(f'models/{filename}', 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] Saved {filename}")

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("[OK] Saved scaler.pkl")

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("[OK] Saved feature_names.pkl")

# Save target encoder (if exists)
if target_encoder is not None:
    with open('models/target_encoder.pkl', 'wb') as f:
        pickle.dump(target_encoder, f)
    print("[OK] Saved target_encoder.pkl")

# Save label encoders for features
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("[OK] Saved label_encoders.pkl")

# Save preprocessing info
preprocessing_info = {
    'target_col': target_col,
    'categorical_cols': categorical_cols,
    'feature_names': list(X.columns)
}
with open('models/preprocessing_info.pkl', 'wb') as f:
    pickle.dump(preprocessing_info, f)
print("[OK] Saved preprocessing_info.pkl")

# Save RAW test data (before preprocessing) for Streamlit app
# Get the original indices from the test set
test_indices = X_test.index

# Get raw data from original dataframe
raw_test_data = df.loc[test_indices].copy()

# Save raw test data with original feature names
raw_test_features = raw_test_data.drop(target_col, axis=1)
raw_test_features.to_csv('test_data_features.csv', index=False)
print("[OK] Saved test_data_features.csv (raw data before preprocessing)")

raw_test_with_labels = raw_test_data.copy()
# Rename target column to 'class' (keep original name)
raw_test_with_labels.to_csv('test_data_with_labels.csv', index=False)
print("[OK] Saved test_data_with_labels.csv (raw data with labels)")

# Save metrics
metrics_data = {
    'comparison_df': comparison_df,
    'ranking': ranking,
    'metrics': {
        'Logistic Regression': {'accuracy': lr_accuracy, 'auc': lr_auc, 'precision': lr_precision,
                                'recall': lr_recall, 'f1': lr_f1, 'mcc': lr_mcc},
        'Decision Tree': {'accuracy': dt_accuracy, 'auc': dt_auc, 'precision': dt_precision,
                         'recall': dt_recall, 'f1': dt_f1, 'mcc': dt_mcc},
        'K-Nearest Neighbors': {'accuracy': knn_accuracy, 'auc': knn_auc, 'precision': knn_precision,
                                'recall': knn_recall, 'f1': knn_f1, 'mcc': knn_mcc},
        'Naive Bayes': {'accuracy': gnb_accuracy, 'auc': gnb_auc, 'precision': gnb_precision,
                       'recall': gnb_recall, 'f1': gnb_f1, 'mcc': gnb_mcc},
        'Random Forest': {'accuracy': rf_accuracy, 'auc': rf_auc, 'precision': rf_precision,
                         'recall': rf_recall, 'f1': rf_f1, 'mcc': rf_mcc},
        'XGBoost': {'accuracy': xgb_accuracy, 'auc': xgb_auc, 'precision': xgb_precision,
                   'recall': xgb_recall, 'f1': xgb_f1, 'mcc': xgb_mcc}
    }
}

with open('models/metrics.pkl', 'wb') as f:
    pickle.dump(metrics_data, f)
print("[OK] Saved metrics.pkl")

print("\n" + "="*70)
print("ALL FILES SAVED SUCCESSFULLY!")
print("="*70)
print("\nNext steps:")
print("1. Run Streamlit app: streamlit run app.py")
print("2. Upload to GitHub")
print("3. Deploy to Streamlit Cloud")
print("="*70)

print("\n[OK] TRAINING COMPLETE!")
