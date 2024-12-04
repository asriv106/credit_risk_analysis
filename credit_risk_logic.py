import pandas as pd
from sklearn.model_selection import train_test_split
import joblib  # To save and load models
from xgboost import XGBClassifier

# Load dataset
def load_data():
    data = pd.read_csv("credit_risk_dataset.csv")
    return data

# Preprocess dataset
def preprocess_data(data):
    # Handle missing values
    data['person_emp_length'].fillna(data['person_emp_length'].median(), inplace=True)
    data['loan_int_rate'].fillna(data['loan_int_rate'].mean(), inplace=True)

    # Encode categorical features
    data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

    # Create additional features
    data['debt_to_income_ratio'] = data['loan_amnt'] / data['person_income']

    # Select relevant columns
    features = data[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                     'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file', 
                     'debt_to_income_ratio']]
    target = data['loan_status']

    return features, target

# Train and save model
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Train XGBoost model
    model = XGBClassifier(max_depth=7, learning_rate=0.2, n_estimators=200, subsample=0.8, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and test data
    joblib.dump(model, "xgboost_model.pkl")
    joblib.dump((X_test, y_test), "test_data.pkl")

    return model

# Load trained model
def load_model():
    model = joblib.load("xgboost_model.pkl")
    X_test, y_test = joblib.load("test_data.pkl")
    return model, X_test, y_test

# Predict using the model
def predict(model, input_data):
    prediction = model.predict(input_data)[0]
    return "Default" if prediction == 1 else "No Default"
