import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

st.title("Customer Churn Prediction Bot")

# Load the cleaned dataset.
df = pd.read_csv("cleaned_dataset.csv")

# Define features and target.
features = ["tenure", "MonthlyCharges", "TotalCharges", "InternetService", "Contract"]
target = "Churn"
X = df[features]
y = df[target]

# Define which features are numeric and which are categorical.
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["InternetService", "Contract"]

# Build a pipeline to preprocess numeric features (impute and scale).
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Build a pipeline for categorical features (impute and one-hot encode).
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine numeric and categorical pipelines.
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create a pipeline with a Logistic Regression classifier.
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Split the dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Tune the regularization strength (parameter C) with cross-validation.
param_grid = {"classifier__C": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Display training and testing accuracy.
train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
st.write(f"Train Accuracy: {train_accuracy:.2f} | Test Accuracy: {test_accuracy:.2f}")

# Build the prediction interface.
st.header("Make a Prediction")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0, max_value=200, value=70)
total_charges = st.number_input("Total Charges ($)", min_value=0, max_value=10000, value=100)
internet_service = st.selectbox("Internet Service", options=df["InternetService"].unique())
contract = st.selectbox("Contract", options=df["Contract"].unique())

# Construct a DataFrame from the user inputs.
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "InternetService": [internet_service],
    "Contract": [contract]
})

if st.button("Predict Churn"):
    prediction = best_model.predict(input_data)[0]
    probability = best_model.predict_proba(input_data)[0][1]
    result = "Likely to churn" if prediction == 1 else "Unlikely to churn"
    st.success(f"Prediction: {result} (Churn probability: {probability:.2f})")

# Display class distribution.
st.write("Churn Class Distribution:")
st.write(df[target].value_counts())
