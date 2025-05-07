import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

st.title("📊 Telco Customer Churn Prediction Bot")

# Load and Preview the Dataset

data_path = "cleaned_dataset.csv"
df = pd.read_csv(data_path)

st.write("✅ Dataset successfully loaded!")
st.write(df.head())

# Feature Selection & Categorical Encoding (LabelEncoder)

selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges']
categorical_columns = ['InternetService', 'Contract']

# Encode categorical features. 
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate Features and Target Variable

target_column = "Churn"
X = df[selected_features]
y = df[target_column]

# impute missing values if any
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# Split into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Scale Numeric Features using StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Decision Tree Classifier with Hyperparameter Tuning

param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3]
}

# GridSearchCV 
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=123, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

train_accuracy = best_model.score(X_train_scaled, y_train)
test_accuracy = best_model.score(X_test_scaled, y_test)
st.write(f"🔍 Train Accuracy: {train_accuracy:.2f} | Test Accuracy: {test_accuracy:.2f}")

# Streamlit User Interface for Prediction

st.header("Make a Prediction")
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
internet_service = st.selectbox("Internet Service", label_encoders['InternetService'].classes_)
contract = st.selectbox("Contract Type", label_encoders['Contract'].classes_)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)

if st.button("Predict Churn"):
    # Create a DataFrame from user input
    user_input = pd.DataFrame([{
        'tenure': tenure,
        'InternetService': label_encoders['InternetService'].transform([internet_service])[0],
        'Contract': label_encoders['Contract'].transform([contract])[0],
        'MonthlyCharges': monthly_charges
    }])
    
    # Scale the user input in the same way as the training data
    user_input_scaled = scaler.transform(user_input)
    
    # Generate the prediction
    prediction = best_model.predict(user_input_scaled)[0]
    result = "Likely to churn" if prediction == 1 else "Unlikely to churn"
    st.success(f"📢 Prediction: **{result}**")

# Display Class Distribution

st.write("📊 Churn Class Distribution in Dataset:")
st.write(df['Churn'].value_counts())
