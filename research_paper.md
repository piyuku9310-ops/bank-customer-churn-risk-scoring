# Predictive Modeling for Customer Churn Using Behavioral Engagement Signals

## Abstract
Customer churn is a major driver of revenue loss in subscription and repeat-purchase businesses. This project reframes churn prediction from a purely transactional view into a behavioral engagement perspective. We develop a supervised learning pipeline that predicts churn probability using customer features and engagement signals. The workflow includes data cleaning, feature preprocessing, model training, evaluation, and deployment via a Streamlit web application for real-time inference.

## 1. Introduction
Churn refers to customers discontinuing a service or reducing engagement such that they no longer generate value. Traditional churn methods often rely heavily on billing/payment history, but modern retention strategies benefit from early warning signals such as engagement decline and service interaction changes. The goal is to estimate churn risk and enable targeted retention actions.

## 2. Problem Statement
Given historical customer data with churn labels, predict the probability that a customer will churn. The solution should be:
- Accurate and interpretable
- Able to handle missing values and mixed feature types
- Deployable as a lightweight web app

## 3. Methodology
### 3.1 Data Preparation
- Load dataset from CSV
- Identify target column (Churn / churn)
- Convert labels into binary format (0/1)
- Separate numerical and categorical features

### 3.2 Feature Engineering & Preprocessing
- Numerical: median imputation + standard scaling
- Categorical: most-frequent imputation + one-hot encoding

### 3.3 Model Selection
We use Logistic Regression with class balancing to handle possible churn imbalance. Logistic Regression is chosen for:
- Strong baseline performance
- Probability outputs for risk scoring
- Interpretability

### 3.4 Evaluation
Metrics used:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC (when probability output is available)
- Confusion matrix

## 4. Deployment
A Streamlit app is provided to:
- Upload customer CSV
- Generate churn probabilities
- Choose a decision threshold
- Download predictions

## 5. Results
Results are saved in `outputs/metrics.json`. The model produces churn probabilities enabling business teams to rank customers by churn risk.

## 6. Conclusion
This project demonstrates an end-to-end churn analytics pipeline from training to deployment. By focusing on engagement-based signals and providing probability-based decisions, the system supports proactive retention strategies.

## References
- Scikit-learn documentation (Logistic Regression, Pipeline, ColumnTransformer)
- Standard ML classification evaluation metrics literature
