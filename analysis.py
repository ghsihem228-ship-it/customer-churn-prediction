import pandas as pd

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Show basic info
print("Initial shape:", df.shape)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Drop missing values
df = df.dropna()

# Drop unnecessary column
df = df.drop("customerID", axis=1)

# Show cleaned data
print("\nCleaned shape:", df.shape)

# Churn distribution
print("\nChurn distribution:\n", df["Churn"].value_counts())
