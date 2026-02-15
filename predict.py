import json
import os
import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = os.path.join("models", "model.pkl")
METRICS_PATH = os.path.join("outputs", "metrics.json")

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("📉 Customer Churn Prediction (ML + Streamlit)")
st.write("Upload a CSV and get churn probability predictions.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run: python train.py")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Model Info")
    if os.path.exists(METRICS_PATH):
        metrics = json.load(open(METRICS_PATH, "r", encoding="utf-8"))
        st.json({
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "rows": metrics.get("n_rows"),
            "features": metrics.get("n_features"),
        })
    else:
        st.info("metrics.json not found. Train the model first.")

with col1:
    st.subheader("Upload Data for Prediction")
    file = st.file_uploader("Upload CSV (same columns as training features)", type=["csv"])

    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    if file is not None:
        df = pd.read_csv(file)

        st.write("Preview:")
        st.dataframe(df.head(10))

        if st.button("Predict"):
            if not hasattr(model, "predict_proba"):
                st.error("Model does not support probabilities.")
                st.stop()

            proba = model.predict_proba(df)[:, 1]
            pred = (proba >= threshold).astype(int)

            out = df.copy()
            out["churn_probability"] = proba
            out["churn_prediction"] = pred

            st.success("✅ Predictions generated!")
            st.dataframe(out.head(50))

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
