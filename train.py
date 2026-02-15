import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression


DATA_PATH = os.path.join("data", "churn.csv")
MODEL_DIR = "models"
OUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def find_target_column(df: pd.DataFrame) -> str:
    # Change here if your target column has another name
    for c in ["Churn", "churn"]:
        if c in df.columns:
            return c
    raise ValueError("Target column not found. Rename your target column to 'Churn' or 'churn'.")


def normalize_target(y: pd.Series) -> pd.Series:
    # Supports: 0/1, True/False, Yes/No, Y/N, churned/not churned
    if y.dtype == "bool":
        return y.astype(int)

    if np.issubdtype(y.dtype, np.number):
        return (y.astype(float) > 0).astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    positive = {"yes", "y", "true", "1", "churn", "churned", "left"}
    return y_str.isin(positive).astype(int)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put your CSV there as churn.csv")

    df = pd.read_csv(DATA_PATH)

    target_col = find_target_column(df)
    y = normalize_target(df[target_col])
    X = df.drop(columns=[target_col])

    # Identify feature types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() == 2 else None
    )

    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None and y_test.nunique() == 2 else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "target_col": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }

    # Save pipeline (includes preprocessing + model)
    joblib.dump(clf, os.path.join(MODEL_DIR, "model.pkl"))

    # Also save just preprocessor (optional)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training complete.")
    print("✅ Saved: models/model.pkl")
    print("✅ Saved: outputs/metrics.json")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
