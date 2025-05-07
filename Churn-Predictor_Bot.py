import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the cleaned dataset
data_path = "cleaned_dataset.csv"
df = pd.read_csv(data_path)

# Confirm dataset loaded
st.write("✅ Dataset successfully loaded!")
st.write(df.head())

# Select key features and identify categorical columns
selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges']
categorical_columns = ['InternetService', 'Contract']

# Prepare encoders for categorical features
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target
target_column = "Churn"
X = df[selected_features]
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Decision Tree model with class_weight balanced
model = DecisionTreeClassifier(max_depth=7, random_state=123, class_weight='balanced')
model.fit(X_train, y_train)

# Check and display model accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
st.write(f"🔍 Train Accuracy: {train_accuracy:.2f} | Test Accuracy: {test_accuracy:.2f}")

# Streamlit app UI
st.title("📊 Telco Customer Churn Prediction Bot")

# Collect user inputs
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
internet_service = st.selectbox("Internet Service", label_encoders['InternetService'].classes_)
contract = st.selectbox("Contract Type", label_encoders['Contract'].classes_)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)

# Prediction button
if st.button("Predict Churn"):
    user_input = pd.DataFrame([{
        'tenure': tenure,
        'InternetService': label_encoders['InternetService'].transform([internet_service])[0],
        'Contract': label_encoders['Contract'].transform([contract])[0],
        'MonthlyCharges': monthly_charges
    }])

    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    result = "Likely to churn" if prediction == 1 else "Unlikely to churn"

    st.success(f"📢 Prediction: **{result}**")

# Show class balance in dataset
st.write("📊 Churn Class Distribution in Dataset:")
st.write(df['Churn'].value_counts())
