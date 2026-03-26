import joblib
import os
import pandas as pd

def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    encoders_path = os.path.join(base_dir, 'models', 'label_encoders.pkl')
    features_path = os.path.join(base_dir, 'models', 'feature_names.pkl')
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        feature_names = joblib.load(features_path)
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError:
        print("Warning: Artifacts not found. Run predict_attrition.py first and copy models dir.")
        return None, None, None, []

def predict_attrition(df_input, model, scaler, label_encoders):
    df = df_input.copy()
    
    # Label Encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen labels by mapping them to majority class or first class
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    # Ensure numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Scale
    df_scaled = scaler.transform(df)
    
    prediction = model.predict(df_scaled)[0]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_scaled)[0][1] # Probability of Class 1
    else:
        proba = float(prediction)
        
    return prediction, proba
