import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

st.title("Customer Churn Prediction Bot (Random Forest + SMOTE)")

# Load the cleaned dataset.
df = pd.read_csv("cleaned_dataset.csv")

# Define features and target.
features = ["tenure", "MonthlyCharges", "TotalCharges", "InternetService", "Contract"]
target = "Churn"
X = df[features]
y = df[target]

# Identify numeric and categorical features.
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["InternetService", "Contract"]

# Build a pipeline for numeric features.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Pipeline for categorical features.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine numeric and categorical transformers.
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split the dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create an imblearn pipeline: preprocess -> SMOTE -> Random Forest Classifier.
pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=123)),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=123))
])

# Tune hyperparameters.
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [5, 7, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate model.
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

st.write(f"**Train Accuracy:** {train_accuracy:.2f} | **Test Accuracy:** {test_accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f} | **Recall:** {recall:.2f} | **F1-score:** {f1:.2f}")

# Display full classification report.
st.text("Classification Report (Test Data):")
st.text(classification_report(y_test, y_test_pred))

# Prediction user interface.
st.header("Make a Prediction")

# Input fields.
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=100.0)
internet_service = st.selectbox("Internet Service", options=df["InternetService"].unique())
contract = st.selectbox("Contract", options=df["Contract"].unique())

# Collect user inputs.
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "InternetService": [internet_service],
    "Contract": [contract]
})

# Predict churn.
if st.button("Predict Churn"):
    prediction = best_model.predict(input_data)[0]
    probability = best_model.predict_proba(input_data)[0][1]
    result = "Likely to churn" if prediction == 1 else "Unlikely to churn"
    st.success(f"📊 Prediction: **{result}** (Churn probability: {probability:.2f})")

# Show class distribution in training data after SMOTE.
st.write("🔍 Class Distribution After SMOTE (During Training):")
smote = SMOTE(random_state=123)
_, y_train_balanced = smote.fit_resample(X_train, y_train)
st.write(pd.Series(y_train_balanced).value_counts())
