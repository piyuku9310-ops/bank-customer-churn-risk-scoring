import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Customer Churn Risk Scoring", layout="centered")

st.title("🏦 Bank Customer Churn Risk Scoring")
st.write("Upload a CSV file of customers and get churn risk predictions.")

# ----- Load model -----
MODEL_PATH = os.path.join("models", "model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error("Model file not found: models/model.pkl")
    st.info("Fix: Make sure your trained model exists in the repo at models/model.pkl")
    st.stop()

# ----- File upload -----
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview")
st.dataframe(df.head())

st.write("✅ Tip: The model expects the same columns used during training.")

if st.button("Predict"):
    try:
        preds = model.predict(df)
        # If model supports predict_proba, show probability too
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[:, 1]

        out = df.copy()
        out["churn_pred"] = preds
        if proba is not None:
            out["churn_risk_score"] = np.round(proba, 4)

        st.subheader("Predictions")
        st.dataframe(out.head(20))

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error("Prediction failed. This usually happens if CSV columns don't match training columns.")
        st.write("Error details:", str(e))
