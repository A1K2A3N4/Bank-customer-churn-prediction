import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "BankData.csv"

st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon="🏦")
st.title("Bank Customer Churn Prediction")
st.write("Enter customer details below to predict whether a customer is likely to churn.")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["CustomerId", "Surname", "Year"], axis=1)
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    df["Balance_to_Salary"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Product_Density"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Engagement_Product"] = df["IsActiveMember"] * df["NumOfProducts"]
    df["Age_Tenure"] = df["Age"] * df["Tenure"]
    return df

@st.cache_data
def train_model(df):
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X.columns.tolist()

try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find BankData.csv in the app directory.")
    st.stop()

model, scaler, feature_columns = train_model(df)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 90, value=35)
    credit_score = st.slider("Credit Score", 350, 850, value=650)
    tenure = st.slider("Tenure", 0, 10, value=3)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    is_active = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)
    has_credit_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Female", "Male"])

if st.button("Predict Churn"):
    row = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_credit_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }
    row["Balance_to_Salary"] = row["Balance"] / (row["EstimatedSalary"] + 1)
    row["Product_Density"] = row["NumOfProducts"] / (row["Tenure"] + 1)
    row["Engagement_Product"] = row["IsActiveMember"] * row["NumOfProducts"]
    row["Age_Tenure"] = row["Age"] * row["Tenure"]

    input_df = pd.DataFrame([row], columns=feature_columns)
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0, 1]
    prediction = model.predict(input_scaled)[0]

    st.write("### Prediction Result")
    st.write(f"Churn probability: {probability:.2f}")
    if prediction == 1:
        st.error("🔴 High risk: This customer is likely to churn.")
    else:
        st.success("🟢 Low risk: This customer is likely to stay.")

    fig, ax = plt.subplots()
    ax.bar(["Stay", "Churn"], [1 - probability, probability], color=["green", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)
