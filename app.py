import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Explainable Credit Risk Prediction System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        .section-card {
            padding: 1.2rem 1.2rem;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .hero-box {
            padding: 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #eff6ff 0%, #ecfeff 100%);
            border: 1px solid #dbeafe;
            margin-bottom: 1rem;
        }
        .risk-good {
            padding: 1rem;
            border-radius: 14px;
            background-color: #ecfdf5;
            border-left: 6px solid #16a34a;
            color: #166534;
            font-weight: 600;
        }
        .risk-mid {
            padding: 1rem;
            border-radius: 14px;
            background-color: #fffbeb;
            border-left: 6px solid #d97706;
            color: #92400e;
            font-weight: 600;
        }
        .risk-high {
            padding: 1rem;
            border-radius: 14px;
            background-color: #fef2f2;
            border-left: 6px solid #dc2626;
            color: #991b1b;
            font-weight: 600;
        }
        .small-note {
            color: #64748b;
            font-size: 0.93rem;
        }
        .explain-box {
            padding: 1rem;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            margin-top: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_gradient_boosting_model.pkl")

model = load_model()

# -----------------------------
# Helper functions
# -----------------------------
def get_risk_band(probability: float):
    if probability >= 0.50:
        return "High Risk"
    elif probability >= 0.20:
        return "Moderate Risk"
    return "Lower Risk"

def feature_friendly_name(feature):
    mapping = {
        "RevolvingUtilizationOfUnsecuredLines": "credit utilization",
        "NumberOfTime30-59DaysPastDueNotWorse": "30–59 days past due history",
        "age": "borrower age",
        "NumberOfTimes90DaysLate": "90 days late history",
        "NumberOfTime60-89DaysPastDueNotWorse": "60–89 days past due history",
        "NumberOfOpenCreditLinesAndLoans": "number of open credit lines and loans",
        "MonthlyIncome": "monthly income",
        "DebtRatio": "debt ratio",
        "NumberRealEstateLoansOrLines": "number of real estate loans or lines",
        "NumberOfDependents": "number of dependents",
    }
    return mapping.get(feature, feature)

def factor_reason(feature, value, shap_value):
    fname = feature_friendly_name(feature)

    if feature == "NumberOfTimes90DaysLate":
        if value > 0:
            return f"The borrower has {int(value)} record(s) of being 90 days late, which strongly increases the estimated default risk."
        return "The borrower has no 90-days-late record, which helps reduce the estimated risk."

    if feature == "NumberOfTime30-59DaysPastDueNotWorse":
        if value > 0:
            return f"The borrower has {int(value)} instance(s) of 30–59 days past due payments, which raises concern about repayment behaviour."
        return "The borrower has no 30–59 days past due record, which supports a lower risk profile."

    if feature == "NumberOfTime60-89DaysPastDueNotWorse":
        if value > 0:
            return f"The borrower has {int(value)} instance(s) of 60–89 days past due payments, which contributes to a higher risk assessment."
        return "The borrower has no 60–89 days past due record, which reduces repayment concern."

    if feature == "RevolvingUtilizationOfUnsecuredLines":
        if value >= 0.8:
            return "Very high credit utilization suggests significant reliance on unsecured credit and strongly increases risk."
        elif value >= 0.5:
            return "Moderately high credit utilization increases the model’s assessment of repayment risk."
        return "Lower credit utilization supports a more stable credit profile."

    if feature == "DebtRatio":
        if value >= 1:
            return "A high debt ratio indicates heavy debt burden relative to available resources, which raises default risk."
        elif value >= 0.5:
            return "A moderate debt ratio still contributes to repayment pressure."
        return "A lower debt ratio supports a healthier repayment position."

    if feature == "MonthlyIncome":
        if value < 3000:
            return "Lower monthly income reduces repayment capacity and contributes to higher estimated risk."
        return "Higher monthly income supports the borrower’s repayment ability."

    if feature == "age":
        if value < 30:
            return "A younger borrower profile is associated with slightly higher predicted risk in this model."
        return "The borrower’s age contributes to a relatively more stable risk profile."

    if feature == "NumberOfOpenCreditLinesAndLoans":
        if value > 10:
            return "A high number of open credit lines may indicate larger credit exposure."
        return "The number of open credit lines is not excessively high."

    if feature == "NumberRealEstateLoansOrLines":
        return f"The borrower has {int(value)} real estate loan/credit line record(s), which has a smaller influence on the final decision."

    if feature == "NumberOfDependents":
        return f"The borrower has {int(value)} dependents, which has a relatively minor effect on the prediction."

    direction = "increases" if shap_value > 0 else "reduces"
    return f"This factor {direction} the estimated risk."

def make_decision_text(probability, top_positive_factors):
    if probability >= 0.50:
        return (
            "Based on the current borrower profile, the system classifies this case as high credit risk. "
            "The borrower may not qualify for a loan under a standard lending policy unless additional review, "
            "collateral, or stricter conditions are applied."
        )
    elif probability >= 0.20:
        return (
            "The borrower falls into a moderate-risk category. This may require closer manual review before a final "
            "loan decision is made, especially if the institution applies conservative risk standards."
        )
    return (
        "The borrower falls into a lower-risk category based on the available input features. "
        "This profile is less likely to trigger serious delinquency within two years, although final approval "
        "should still depend on institutional lending policy."
    )

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="hero-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="main-title">Explainable AI System for Credit Risk Prediction</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">This application predicts whether a borrower is likely to experience serious delinquency within two years and explains the prediction using SHAP and plain-language reasoning.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Borrower Information")
st.sidebar.markdown("Adjust the values below and generate a prediction.")

revolving_utilization = st.sidebar.number_input(
    "Revolving Utilization Of Unsecured Lines", min_value=0.0, value=0.50, step=0.01
)
age = st.sidebar.number_input("Age", min_value=18, value=35, step=1)
past_due_30_59 = st.sidebar.number_input("Number Of Times 30–59 Days Past Due", min_value=0, value=0, step=1)
debt_ratio = st.sidebar.number_input("Debt Ratio", min_value=0.0, value=0.40, step=0.01)
monthly_income = st.sidebar.number_input("Monthly Income", min_value=0.0, value=5000.0, step=100.0)
open_credit_lines = st.sidebar.number_input("Number Of Open Credit Lines And Loans", min_value=0, value=5, step=1)
times_90_days_late = st.sidebar.number_input("Number Of Times 90 Days Late", min_value=0, value=0, step=1)
real_estate_loans = st.sidebar.number_input("Number Of Real Estate Loans Or Lines", min_value=0, value=1, step=1)
past_due_60_89 = st.sidebar.number_input("Number Of Times 60–89 Days Past Due", min_value=0, value=0, step=1)
dependents = st.sidebar.number_input("Number Of Dependents", min_value=0, value=0, step=1)

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

# -----------------------------
# Layout top section
# -----------------------------
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Input Data")
    st.dataframe(input_data, use_container_width=True)
    st.markdown('<div class="small-note">The model uses these borrower features to estimate default probability.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("How to Use")
    st.write(
        """
        1. Enter borrower details in the sidebar.  
        2. Click **Predict Credit Risk**.  
        3. Review the predicted probability and risk category.  
        4. Read the SHAP-based explanation to understand the decision.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Credit Risk", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_band = get_risk_band(probability)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if risk_band == "High Risk":
        st.markdown(
            '<div class="risk-high">The borrower is likely to experience serious delinquency and is classified as High Risk.</div>',
            unsafe_allow_html=True
        )
    elif risk_band == "Moderate Risk":
        st.markdown(
            '<div class="risk-mid">The borrower falls into a Moderate Risk category and may require closer review.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="risk-good">The borrower is unlikely to experience serious delinquency and is classified as Lower Risk.</div>',
            unsafe_allow_html=True
        )

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Predicted Probability of Default", f"{probability:.2%}")
    with m2:
        st.metric("Risk Category", risk_band)

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # SHAP values
    # -----------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = np.array(shap_values)[0]

    feature_names = list(input_data.columns)
    feature_values = input_data.iloc[0].to_dict()

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Feature Value": [feature_values[f] for f in feature_names],
        "SHAP Value": sv
    })

    shap_df["Abs SHAP"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values("Abs SHAP", ascending=False)

    top_factors = shap_df.head(3)
    top_positive = shap_df[shap_df["SHAP Value"] > 0].sort_values("SHAP Value", ascending=False).head(3)

    # -----------------------------
    # Human explanation
    # -----------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Decision Explanation")

    st.write(make_decision_text(probability, top_positive))

    if len(top_factors) > 0:
        st.markdown("**Top factors influencing this prediction:**")
        for _, row in top_factors.iterrows():
            reason = factor_reason(row["Feature"], row["Feature Value"], row["SHAP Value"])
            direction = "increased" if row["SHAP Value"] > 0 else "reduced"
            st.markdown(
                f"- **{feature_friendly_name(row['Feature']).title()}** {direction} the estimated risk. {reason}"
            )

    if probability >= 0.50 and len(top_positive) > 0:
        key_factor_names = ", ".join([feature_friendly_name(f) for f in top_positive["Feature"].tolist()])
        st.warning(
            f"Loan decision rationale: this profile may not qualify under standard lending rules mainly because of {key_factor_names}."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # SHAP chart
    # -----------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("SHAP Feature Contribution")
    st.write("The chart below ranks the input features by how strongly they influenced this specific prediction.")

    plot_values = np.array([sv])

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.summary_plot(
        plot_values,
        input_data,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig, clear_figure=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # Feature table
    # -----------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Borrower Feature Values")
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
