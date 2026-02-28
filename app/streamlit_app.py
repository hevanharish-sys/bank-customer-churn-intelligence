import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# ===============================
# Load Model & Feature Columns
# ===============================

model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="Bank Churn Intelligence System", layout="wide")

st.title("🏦 Predictive Modeling & Risk Scoring for Bank Customer Churn")
st.markdown("Advanced Churn Risk Intelligence Dashboard")

tabs = st.tabs(["📊 Risk Calculator", "📈 Model Insights", "🔮 What-If Simulator"])

# ==========================================================
# 🔹 TAB 1: RISK CALCULATOR
# ==========================================================
with tabs[0]:

    st.header("Customer Churn Risk Calculator")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80, 35, key="rc_age")
        credit_score = st.slider("Credit Score", 300, 900, 650, key="rc_credit")
        tenure = st.slider("Tenure (Years)", 0, 10, 3, key="rc_tenure")
        balance = st.number_input("Account Balance", value=50000.0, key="rc_balance")

    with col2:
        num_products = st.slider("Number of Products", 1, 4, 2, key="rc_products")
        has_card = st.selectbox("Has Credit Card", [0, 1], key="rc_card")
        is_active = st.selectbox("Is Active Member", [0, 1], key="rc_active")
        salary = st.number_input("Estimated Salary", value=60000.0, key="rc_salary")

    if st.button("Calculate Churn Risk", key="rc_button"):

        input_data = pd.DataFrame(columns=feature_columns)
        input_data.loc[0] = 0

        # Base features
        input_data["CreditScore"] = credit_score
        input_data["Age"] = age
        input_data["Tenure"] = tenure
        input_data["Balance"] = balance
        input_data["NumOfProducts"] = num_products
        input_data["HasCrCard"] = has_card
        input_data["IsActiveMember"] = is_active
        input_data["EstimatedSalary"] = salary

        # Engineered features
        input_data["BalanceToSalaryRatio"] = balance / (salary + 1)
        input_data["ProductDensity"] = num_products / (tenure + 1)
        input_data["AgeTenureInteraction"] = age * tenure
        input_data["EngagementScore"] = is_active * num_products

        probability = model.predict_proba(input_data)[0][1]

        if probability < 0.30:
            risk = "Low Risk 🟢"
        elif probability < 0.70:
            risk = "Medium Risk 🟡"
        else:
            risk = "High Risk 🔴"

        st.subheader(f"Churn Probability: {probability:.2%}")
        st.subheader(f"Risk Level: {risk}")

# ==========================================================
# 🔹 TAB 2: MODEL INSIGHTS
# ==========================================================
with tabs[1]:

    st.header("Model Insights & Explainability")

    st.subheader("Top Feature Importances")
    feature_img = Image.open("models/feature_importance.png")
    st.image(feature_img)

    st.subheader("SHAP Summary Plot")
    shap_img = Image.open("models/shap_summary.png")
    st.image(shap_img)

# ==========================================================
# 🔹 TAB 3: WHAT-IF SIMULATOR
# ==========================================================
with tabs[2]:

    st.header("What-If Scenario Analysis")

    st.markdown("Adjust financial and engagement variables to simulate churn probability.")

    sim_balance = st.slider("Balance", 0, 200000, 50000, key="sim_balance")
    sim_products = st.slider("Number of Products", 1, 4, 2, key="sim_products")
    sim_active = st.selectbox("Active Member", [0, 1], key="sim_active")

    if st.button("Run Simulation", key="sim_button"):

        sim_input = pd.DataFrame(columns=feature_columns)
        sim_input.loc[0] = 0

        # Fixed base profile
        sim_input["CreditScore"] = 650
        sim_input["Age"] = 35
        sim_input["Tenure"] = 3
        sim_input["Balance"] = sim_balance
        sim_input["NumOfProducts"] = sim_products
        sim_input["HasCrCard"] = 1
        sim_input["IsActiveMember"] = sim_active
        sim_input["EstimatedSalary"] = 60000

        # Engineered features
        sim_input["BalanceToSalaryRatio"] = sim_balance / (60000 + 1)
        sim_input["ProductDensity"] = sim_products / (3 + 1)
        sim_input["AgeTenureInteraction"] = 35 * 3
        sim_input["EngagementScore"] = sim_active * sim_products

        sim_prob = model.predict_proba(sim_input)[0][1]

        st.subheader(f"Predicted Churn Probability: {sim_prob:.2%}")
