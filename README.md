# Credit Risk Prediction App

This is a **Streamlit-based web application** for predicting loan defaults using machine learning models. The app utilizes **XGBoost** and **Random Forest** models to analyze credit risk and provide actionable insights. It is designed to help financial institutions make better lending decisions by identifying high-risk applicants.

---

## Features

1. **Overview**:
   - A brief introduction to credit risk prediction and its importance.

2. **Data Exploration**:
   - Visualize the dataset.
   - Analyze missing values, feature distributions, and correlations.

3. **Model Insights**:
   - Understand feature importance.
   - Evaluate model performance metrics such as Accuracy, ROC-AUC, and Confusion Matrix.

4. **Make a Prediction**:
   - Input applicant details to predict the likelihood of loan default.
   - Choose between **XGBoost** and **Random Forest** for predictions.
   - Get default probability and prediction results.

5. **Insights & Recommendations**:
   - Key drivers of loan default.
   - Actionable strategies to minimize credit risk.

---

## Technology Stack

- **Frontend**: Streamlit
- **Backend Models**: 
  - XGBoost
  - Random Forest
- **Libraries**:
  - Python
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Imbalanced-learn
  - Joblib

---

## Project Setup

### Prerequisites

1. Python 3.8 or higher installed.
2. Required Python libraries:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib xgboost
