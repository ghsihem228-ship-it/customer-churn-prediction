import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📉 Customer Churn Prediction Dashboard")
st.write("A machine learning dashboard to analyze and predict customer churn.")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop("customerID", axis=1)

st.subheader("Dataset Preview")
st.dataframe(df.head(20))

st.subheader("Churn Distribution")
fig1, ax1 = plt.subplots()
df["Churn"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_xlabel("Churn")
ax1.set_ylabel("Count")
st.pyplot(fig1)

st.subheader("Monthly Charges Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(df["MonthlyCharges"], bins=30)
ax2.set_xlabel("Monthly Charges")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# Safe encoding
encoded_df = pd.get_dummies(df, drop_first=True)

X = encoded_df.drop("Churn_Yes", axis=1)
y = encoded_df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.metric("Logistic Regression Accuracy", round(accuracy, 3))

st.success("Dashboard loaded successfully.")
