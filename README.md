import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["churn_probability"] = proba
    out["churn_prediction"] = pred
    return out
