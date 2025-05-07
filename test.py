
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
try:
    df = pd.read_csv(data_path)
    print("Data successfully loaded!")
except FileNotFoundError:
    print("Error: File not found. Please check the path and try again.")

print("\nDataset Overview:")
print(df.head())
print("\nDataset Summary:")
print(df.info())

irrelevant_columns = ["RowNumber", "CustomerId", "Surname"]  # Adjust as per your dataset
df.drop(columns=irrelevant_columns, inplace=True, errors='ignore')
print("\nColumns after removing irrelevant ones:")
print(df.columns)

print("\nMissing Values Summary:")
print(df.isnull().sum())

imputer = SimpleImputer(strategy="mean")
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

categorical_columns = df.select_dtypes(include=["object"]).columns
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

print("\nData after encoding categorical variables:")
print(df.head())

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\nData after scaling numerical features:")
print(df.head())

duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    print(f"\nFound {duplicate_count} duplicate rows. Removing them.")
    df = df.drop_duplicates()
else:
    print("\nNo duplicate rows found.")

print("\nCleaned Dataset Preview:")
print(df.head())
print("\nCleaned Dataset Summary:")
print(df.info())

output_path = "cleaned_dataset.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to '{output_path}'!")
