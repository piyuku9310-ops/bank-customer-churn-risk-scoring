# Customer Churn Prediction (ML + Streamlit)

## Overview
This project predicts customer churn using a machine learning pipeline that includes:
- Data preprocessing (missing values, scaling, one-hot encoding)
- Logistic Regression model (balanced class weights)
- Streamlit web app for interactive prediction

## Folder Structure
- `data/churn.csv` -> dataset
- `train.py` -> trains and saves model
- `app.py` -> Streamlit UI
- `models/model.pkl` -> saved trained pipeline
- `outputs/metrics.json` -> evaluation metrics

## Setup (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Dataset Requirement
Your CSV must contain a churn target column named `Churn` or `churn`.
If different, edit `train.py` -> `find_target_column()`.

## Output
- Saved model: `models/model.pkl`
- Metrics: `outputs/metrics.json`
- App provides downloadable predictions CSV
