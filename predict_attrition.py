"""
predict_attrition.py
====================
Script untuk memprediksi 412 missing values pada kolom Attrition
menggunakan pipeline preprocessing dan model terbaik.

Output: employee_data_predicted.csv (dataset lengkap tanpa missing values)
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

# ============================================================
# 1. Load Dataset
# ============================================================
print("=" * 60)
print("STEP 1: Loading dataset...")
print("=" * 60)

df = pd.read_csv('employee_data.csv')
attrition_missing_mask = df['Attrition'].isna()
attrition_original = df['Attrition'].copy()
print(f"Dataset shape: {df.shape}")
print(f"Missing values in Attrition: {df['Attrition'].isna().sum()}")
print()

# ============================================================
# 2. Separate labeled vs unlabeled data
# ============================================================
print("=" * 60)
print("STEP 2: Separating labeled vs unlabeled data...")
print("=" * 60)

df_labeled = df[df['Attrition'].notna()].copy()
df_unlabeled = df[df['Attrition'].isna()].copy()

print(f"Labeled rows (train): {len(df_labeled)}")
print(f"Unlabeled rows (to predict): {len(df_unlabeled)}")
print(f"Attrition rate (labeled): {df_labeled['Attrition'].mean()*100:.1f}%")
print()

# ============================================================
# 3. Preprocessing
# ============================================================
print("=" * 60)
print("STEP 3: Preprocessing...")
print("=" * 60)

# 3a. Drop constant/useless columns
drop_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeId']
df_labeled.drop(columns=drop_cols, inplace=True)
df_unlabeled.drop(columns=drop_cols, inplace=True)
print(f"Dropped columns: {drop_cols}")

# 3b. Identify categorical columns
cat_cols = df_labeled.select_dtypes(include=['object', 'string']).columns.tolist()
print(f"Categorical columns to encode: {cat_cols}")

# 3c. Label Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([df_labeled[col], df_unlabeled[col]], ignore_index=True)
    le.fit(all_values)
    df_labeled[col] = le.transform(df_labeled[col])
    df_unlabeled[col] = le.transform(df_unlabeled[col])
    label_encoders[col] = le

print(f"Label encoded {len(cat_cols)} columns")

# 3d. Prepare features and target
X_labeled = df_labeled.drop(columns=['Attrition'])
y_labeled = df_labeled['Attrition'].astype(int)
X_unlabeled = df_unlabeled.drop(columns=['Attrition'])

print(f"Feature columns: {X_labeled.shape[1]}")
print(f"Training samples: {X_labeled.shape[0]}")
print(f"Samples to predict: {X_unlabeled.shape[0]}")

# 3e. StandardScaler
scaler = StandardScaler()
X_labeled_scaled = scaler.fit_transform(X_labeled)
X_unlabeled_scaled = scaler.transform(X_unlabeled)
print("Applied StandardScaler")
print()

# ============================================================
# 4. Evaluate multiple models with Cross-Validation
# ============================================================
print("=" * 60)
print("STEP 4: Evaluating models (5-Fold Stratified CV)...")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, random_state=42,
        use_label_encoder=False, eval_metric='logloss'
    ),
}

results = {}
print(f"\n{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
print("-" * 60)

for name, model in models.items():
    acc_scores = cross_val_score(model, X_labeled_scaled, y_labeled, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X_labeled_scaled, y_labeled, cv=cv, scoring='f1')
    auc_scores = cross_val_score(model, X_labeled_scaled, y_labeled, cv=cv, scoring='roc_auc')

    results[name] = {
        'accuracy': acc_scores.mean(),
        'f1': f1_scores.mean(),
        'auc': auc_scores.mean(),
        'model': model
    }

    print(f"{name:<25} {acc_scores.mean():>10.4f} {f1_scores.mean():>10.4f} {auc_scores.mean():>10.4f}")

# Select best model by ROC-AUC
best_name = max(results, key=lambda k: results[k]['auc'])
best_info = results[best_name]
print(f"\n>>> Best model (by ROC-AUC): {best_name}")
print(f"    Accuracy: {best_info['accuracy']:.4f}")
print(f"    F1-Score: {best_info['f1']:.4f}")
print(f"    ROC-AUC:  {best_info['auc']:.4f}")
print()

# ============================================================
# 5. Train best model on ALL labeled data
# ============================================================
print("=" * 60)
print(f"STEP 5: Training best model ({best_name}) on ALL labeled data...")
print("=" * 60)

best_model = best_info['model']
best_model.fit(X_labeled_scaled, y_labeled)
print(f"Training accuracy: {best_model.score(X_labeled_scaled, y_labeled):.4f}")
print()

# ============================================================
# 6. Predict missing Attrition values
# ============================================================
print("=" * 60)
print("STEP 6: Predicting missing Attrition values...")
print("=" * 60)

predictions = best_model.predict(X_unlabeled_scaled)
prediction_probabilities = None
if hasattr(best_model, 'predict_proba'):
    prediction_probabilities = best_model.predict_proba(X_unlabeled_scaled)[:, 1]

print(f"Predictions made: {len(predictions)}")
print(f"Predicted as No Attrition (0): {(predictions == 0).sum()}")
print(f"Predicted as Attrition (1): {(predictions == 1).sum()}")
print(f"Predicted Attrition rate: {predictions.mean()*100:.1f}%")
print()

# ============================================================
# 7. Fill predictions into original dataset
# ============================================================
print("=" * 60)
print("STEP 7: Filling predictions into original dataset...")
print("=" * 60)

df['Predicted_Row'] = attrition_missing_mask.astype(int)
df['Attrition_Original'] = attrition_original
df['Attrition_Predicted'] = np.nan
df['Attrition_Predicted_Yes'] = 0
df['Attrition_Source'] = np.where(attrition_missing_mask, 'Predicted', 'Original')

if prediction_probabilities is not None:
    df['Attrition_Prediction_Probability'] = np.nan

unlabeled_indices = df[attrition_missing_mask].index
df.loc[unlabeled_indices, 'Attrition'] = predictions.astype(float)
df.loc[unlabeled_indices, 'Attrition_Predicted'] = predictions.astype(float)
df.loc[unlabeled_indices, 'Attrition_Predicted_Yes'] = predictions.astype(int)

if prediction_probabilities is not None:
    df.loc[unlabeled_indices, 'Attrition_Prediction_Probability'] = prediction_probabilities

df['Attrition_Final'] = df['Attrition']
df['Attrition_Yes'] = (df['Attrition_Final'] == 1).astype(int)
df['Attrition_No'] = (df['Attrition_Final'] == 0).astype(int)

print(f"Filled {len(predictions)} missing values")
print(f"Missing values AFTER prediction: {df['Attrition'].isna().sum()}")
print(f"Rows marked as predicted: {df['Predicted_Row'].sum()}")
print(f"Predicted Attrition (metric-ready): {df['Attrition_Predicted_Yes'].sum()}")
print(f"Final Attrition Yes rows: {df['Attrition_Yes'].sum()}")
print(f"Final Attrition No rows: {df['Attrition_No'].sum()}")
print()

# ============================================================
# 8. Save complete dataset
# ============================================================
print("=" * 60)
print("STEP 8: Saving complete dataset...")
print("=" * 60)

output_file = 'employee_data_predicted.csv'
try:
    df.to_csv(output_file, index=False)
except PermissionError:
    output_file = 'employee_data_predicted_dashboard.csv'
    df.to_csv(output_file, index=False)
    print("Primary output file is locked; saved to fallback file instead.")
print(f"Saved to: {output_file}")
print()

# ============================================================
# Export Model and Preprocessing Objects
# ============================================================
print("=" * 60)
print("STEP 8.5: Exporting Model and Preprocessing Objects...")
print("=" * 60)
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
feature_names = X_labeled.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.pkl')
print("Model, scaler, and label encoders saved to 'models/' directory.")
print()

# ============================================================
# 9. Verification
# ============================================================
print("=" * 60)
print("VERIFICATION")
print("=" * 60)

df_verify = pd.read_csv(output_file)
print(f"Dataset shape: {df_verify.shape}")
print(f"Missing values in Attrition: {df_verify['Attrition'].isna().sum()}")
print(f"\nAttrition distribution:")
print(df_verify['Attrition'].value_counts().sort_index())
print(f"\nPredicted rows: {int(df_verify['Predicted_Row'].sum())}")
print(f"Predicted attrition count: {int(df_verify['Attrition_Predicted_Yes'].sum())}")
print(f"Final attrition yes count: {int(df_verify['Attrition_Yes'].sum())}")
print(f"Final attrition no count: {int(df_verify['Attrition_No'].sum())}")
print(f"\nAttrition rate (full dataset): {df_verify['Attrition'].mean()*100:.1f}%")

unique_vals = sorted(df_verify['Attrition'].unique())
print(f"\nUnique values in Attrition: {unique_vals}")
assert df_verify['Attrition'].isna().sum() == 0, "ERROR: Still has missing values!"
assert set(unique_vals) == {0.0, 1.0}, "ERROR: Unexpected values in Attrition!"

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"Best model: {best_name}")
print(f"CV Accuracy: {best_info['accuracy']:.4f}")
print(f"CV F1-Score: {best_info['f1']:.4f}")
print(f"CV ROC-AUC:  {best_info['auc']:.4f}")
print(f"Dataset saved to: {output_file}")
print("No missing values in Attrition column.")
