
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Explainable Credit Risk Prediction System",
    page_icon="📊",
    layout="wide"
)

st.title("Explainable AI System for Credit Risk Prediction")
st.markdown(
    """
    This application predicts whether a borrower is likely to experience
    serious delinquency within two years and explains the prediction using SHAP.
    """
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("credit_risk_gradient_boosting_model.pkl")
    return model

model = load_model()

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Borrower Information")

revolving_utilization = st.sidebar.number_input(
    "Revolving Utilization Of Unsecured Lines",
    min_value=0.0,
    value=0.5,
    step=0.01
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    value=35,
    step=1
)

past_due_30_59 = st.sidebar.number_input(
    "Number Of Time 30-59 Days Past Due Not Worse",
    min_value=0,
    value=0,
    step=1
)

debt_ratio = st.sidebar.number_input(
    "Debt Ratio",
    min_value=0.0,
    value=0.4,
    step=0.01
)

monthly_income = st.sidebar.number_input(
    "Monthly Income",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

open_credit_lines = st.sidebar.number_input(
    "Number Of Open Credit Lines And Loans",
    min_value=0,
    value=5,
    step=1
)

times_90_days_late = st.sidebar.number_input(
    "Number Of Times 90 Days Late",
    min_value=0,
    value=0,
    step=1
)

real_estate_loans = st.sidebar.number_input(
    "Number Real Estate Loans Or Lines",
    min_value=0,
    value=1,
    step=1
)

past_due_60_89 = st.sidebar.number_input(
    "Number Of Time 60-89 Days Past Due Not Worse",
    min_value=0,
    value=0,
    step=1
)

dependents = st.sidebar.number_input(
    "Number Of Dependents",
    min_value=0,
    value=0,
    step=1
)

# -----------------------------
# Prepare input data
# -----------------------------
input_data = pd.DataFrame({
    "RevolvingUtilizationOfUnsecuredLines": [revolving_utilization],
    "age": [age],
    "NumberOfTime30-59DaysPastDueNotWorse": [past_due_30_59],
    "DebtRatio": [debt_ratio],
    "MonthlyIncome": [monthly_income],
    "NumberOfOpenCreditLinesAndLoans": [open_credit_lines],
    "NumberOfTimes90DaysLate": [times_90_days_late],
    "NumberRealEstateLoansOrLines": [real_estate_loans],
    "NumberOfTime60-89DaysPastDueNotWorse": [past_due_60_89],
    "NumberOfDependents": [dependents]
})

st.subheader("Input Data")
st.dataframe(input_data, use_container_width=True)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("The borrower is likely to experience serious delinquency (Default Risk).")
    else:
        st.success("The borrower is unlikely to experience serious delinquency (No Default Risk).")

    st.metric("Predicted Probability of Default", f"{probability:.2%}")

    # -----------------------------
    # SHAP explanation
    # -----------------------------
    st.subheader("SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Waterfall plot / force-style alternative
    st.write("Feature contribution to this prediction:")

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        input_data.iloc[0],
        show=False
    )
    st.pyplot(fig, clear_figure=True)

    # Bar plot for this single case
    st.subheader("Input Feature Values")
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
