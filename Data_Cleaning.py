import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset.
data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)
print("Dataset loaded.")
print(df.head())

# Drop irrelevant columns such as customerID.
df.drop(columns=["customerID"], inplace=True, errors='ignore')

# Convert 'TotalCharges' to numeric. Non-numeric values will be set as NaN.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Show missing values before imputation.
print("\nMissing values before imputation:")
print(df.isnull().sum())

# Impute missing values for numeric columns using the mean.
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Fill missing entries for categorical columns with the most frequent value.
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode the target column "Churn" if it's in a string format.
if df["Churn"].dtype == "object":
    le = LabelEncoder()
    df["Churn"] = le.fit_transform(df["Churn"])

# Remove duplicate rows.
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"\nFound {duplicates} duplicate rows; removing them.")
    df.drop_duplicates(inplace=True)

# Convert all numeric values into whole numbers.
# For each numeric column, round values and if any value is negative, set it to 0.
df[numeric_cols] = df[numeric_cols].apply(lambda col: col.map(lambda x: int(round(x)) if x >= 0 else 0))

# Verify that there are no missing values left.
print("\nMissing values after imputation and conversion:")
print(df.isnull().sum())

# Save the cleaned dataset.
output_path = "cleaned_dataset.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved as '{output_path}'!")
