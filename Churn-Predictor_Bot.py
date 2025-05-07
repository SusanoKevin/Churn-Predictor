

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data_path = "cleaned_dataset.csv"
df = pd.read_csv(data_path)

# Select key features and encode categorical variables
selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges']
categorical_columns = ['InternetService', 'Contract']

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Encode the target variable
target_column = "Churn"
target_encoder = LabelEncoder()
df[target_column] = target_encoder.fit_transform(df[target_column])

# Split data
X = df[selected_features]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = DecisionTreeClassifier(max_depth=5, random_state=123)
model.fit(X_train, y_train)

# Streamlit App UI
st.title("📊 Customer Churn Prediction Bot")

# Collect user input via UI
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
internet_service = st.selectbox("Internet Service", label_encoders['InternetService'].classes_)
contract = st.selectbox("Contract", label_encoders['Contract'].classes_)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)

# Prepare input for prediction
if st.button("Predict Churn"):
    user_input = pd.DataFrame([{
        'tenure': tenure,
        'InternetService': label_encoders['InternetService'].transform([internet_service])[0],
        'Contract': label_encoders['Contract'].transform([contract])[0],
        'MonthlyCharges': monthly_charges
    }])

    # Scale input
    user_input_scaled = scaler.transform(user_input)

    # Predict
    prediction = model.predict(user_input_scaled)[0]
    result = "Likely to churn" if prediction == 1 else "Unlikely to churn"

    st.success(f"Prediction: **{result}**")


