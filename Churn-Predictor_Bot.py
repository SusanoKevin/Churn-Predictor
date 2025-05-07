import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

st.title("Telco Customer Churn Prediction Bot")

# Load the cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")
st.write("Dataset loaded.")
st.write(df.head())

# Select features and target variable.
# Using features: tenure, MonthlyCharges, TotalCharges, InternetService, and Contract.
features = ["tenure", "MonthlyCharges", "TotalCharges", "InternetService", "Contract"]
target = "Churn"

X = df[features]
y = df[target]

st.write("Feature sample:")
st.write(X.head())

# Define numeric and categorical columns.
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["InternetService", "Contract"]

# Build a transformation pipeline.
# Numeric columns: impute missing values (if any) and scale them.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
# Categorical columns: impute and apply one-hot encoding.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create a modeling pipeline with Logistic Regression using L2 regularization.
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Split the data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Use GridSearchCV to tune the regularization strength (C).
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)

st.write(f"Train Accuracy: {train_accuracy:.2f} | Test Accuracy: {test_accuracy:.2f}")
st.write("Best hyperparameters:", grid_search.best_params_)

# Prediction user interface
st.header("Make a Prediction")

# Input fields for features.
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0, max_value=200, value=70)
total_charges = st.number_input("Total Charges ($)", min_value=0, max_value=10000, value=100)

# Use unique categorical values from the dataset.
internet_service = st.selectbox("Internet Service", options=df["InternetService"].unique())
contract = st.selectbox("Contract", options=df["Contract"].unique())

# Construct a DataFrame from user inputs.
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

# Display the class distribution in the dataset.
st.write("Churn Class Distribution:")
st.write(df[target].value_counts())
