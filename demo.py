# credit_risk_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your saved models and scaler
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')  # Pre-fitted scaler used during training

# Load dataset for data exploration
data = pd.read_csv('credit_risk_dataset.csv')

# Add income_category to the dataset using quantiles
data['income_category'] = pd.qcut(data['person_income'], q=3, labels=['low', 'medium', 'high'])

# Title and Description
st.title("Credit Risk Prediction App")
st.write("""
This application uses machine learning models to predict the likelihood of a loan default based on applicant information.
You can explore the data, understand model insights, and make your own predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select a Section", ["Overview","Make a Prediction"])

# Overview Section
if option == "Overview":
    st.header("Overview")
    st.write("""
    **Credit Risk Prediction** is crucial for financial institutions to minimize losses due to defaults.
    This app demonstrates how machine learning models like **Random Forest** and **XGBoost** can be used to predict loan defaults.
    """)

# Make a Prediction Section
elif option == "Make a Prediction":
    st.header("Make a Prediction")
    st.write("Input applicant information to predict the likelihood of a loan default.")

    # Input fields for user data
    person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Person Income", min_value=1000, max_value=1000000, value=50000)
    person_emp_length = st.number_input("Employment Length (months)", min_value=0, max_value=480, value=12)
    loan_amnt = st.number_input("Loan Amount", min_value=100, max_value=50000, value=10000)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=40.0, value=10.0)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2)
    cb_person_default_on_file = st.selectbox("Historical Default on File", ['No', 'Yes'])
    cb_person_default_on_file = 1 if cb_person_default_on_file == 'Yes' else 0

    # Compute debt_to_income_ratio
    debt_to_income_ratio = loan_amnt / person_income

    # Calculate income category dynamically
    income_thresholds = data['person_income'].quantile([0.33, 0.67])
    if person_income <= income_thresholds[0.33]:
        income_category = 'low'
    elif person_income <= income_thresholds[0.67]:
        income_category = 'medium'
    else:
        income_category = 'high'

    # Encode income category
    medium = 1 if income_category == 'medium' else 0
    high = 1 if income_category == 'high' else 0

    # Prepare input data
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_emp_length': [person_emp_length],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'debt_to_income_ratio': [debt_to_income_ratio],
        'medium': [medium],
        'high': [high]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Select Model
    model_choice = st.radio("Choose Model for Prediction", ["XGBoost", "Random Forest"])

    # Prediction
    if st.button("Predict"):
        if model_choice == "XGBoost":
            prediction = xgb_model.predict(input_data_scaled)[0]
            probability = xgb_model.predict_proba(input_data_scaled)[0][1]
        else:
            prediction = rf_model.predict(input_data_scaled)[0]
            probability = rf_model.predict_proba(input_data_scaled)[0][1]

        result = 'Default' if prediction == 1 else 'No Default'
        st.write(f"### Prediction: {result}")
        st.write(f"### Probability of Default: {probability:.2%}")
