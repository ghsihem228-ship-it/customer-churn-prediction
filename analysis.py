import pandas as pd

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Show first rows
print(df.head())

# Show info
print(df.info())

# Check missing values
print(df.isnull().sum())
