import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

# Remove irrelevant columns
irrelevant_columns = ["customerID"]
df.drop(columns=irrelevant_columns, inplace=True, errors='ignore')

# Convert TotalCharges to numeric (fix blank strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Show missing values
print("\nMissing Values Summary:")
print(df.isnull().sum())

# Impute missing numerical values
imputer = SimpleImputer(strategy="mean")
# Compute numerical columns before label encoding (target still as object)
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

# Encode target separately
target_encoder = LabelEncoder()
df['Churn'] = target_encoder.fit_transform(df['Churn'])

# Encode other categorical variables
categorical_columns = df.select_dtypes(include=["object"]).columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Scale numerical features (exclude target)
# Recompute numerical columns after encoding so that "Churn" is now numeric
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
# Remove the target column if it is present
if "Churn" in numerical_columns:
    numerical_columns.remove("Churn")
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Remove duplicates
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    print(f"\nFound {duplicate_count} duplicate rows. Removing them.")
    df = df.drop_duplicates()

# Save cleaned data
output_path = "cleaned_dataset.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to '{output_path}'!")
