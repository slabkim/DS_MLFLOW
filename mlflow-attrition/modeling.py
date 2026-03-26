import os
import pandas as pd
import warnings
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('data/data_clean.csv')

X = df.drop(columns=['Attrition'])
y = df['Attrition'].astype(int)

# 2. Preprocessing
drop_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeId']
X.drop(columns=[col for col in drop_cols if col in X.columns], inplace=True)

cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. MLFlow Tracking Setup
mlflow.set_experiment("Attrition_Prediction_Experiment")

# Start an MLFlow run
with mlflow.start_run(run_name="XGBoost_Run"):
    # Define hyperparameters
    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    # Log Parameters
    mlflow.log_params(params)

    # Train Model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Log Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    # Log Model
    mlflow.sklearn.log_model(model, "model")

    print(f"Run completed. Metrics: Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # If the user sets credentials (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD),
    # this script will seamlessly track to remote DagsHub. Otherwise, it logs locally to ./mlruns.
