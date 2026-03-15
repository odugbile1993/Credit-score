import streamlit as st
import pandas as pd
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

# -----------------------------
# Simple styling
# -----------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #4b5563;
            margin-bottom: 1.2rem;
        }
        .section-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            background-color: #f8fafc;
            border: 1px solid #e5e7eb;
            margin-bottom: 1rem;
        }
        .small-note {
            color: #6b7280;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">Explainable AI System for Credit Risk Prediction</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">This application predicts whether a borrower is likely to experience serious delinquency within two years and explains the prediction using SHAP.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_gradient_boosting_model.pkl")

model = load_model()

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Borrower Information")
st.sidebar.markdown("Enter the borrower details below to generate a prediction.")

revolving_utilization = st.sidebar.number_input(
    "Revolving Utilization Of Unsecured Lines",
    min_value=0.0,
    value=0.50,
    step=0.01,
    help="Ratio of credit used compared to total available unsecured credit."
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    value=35,
    step=1
)

past_due_30_59 = st.sidebar.number_input(
    "Number Of Times 30–59 Days Past Due",
    min_value=0,
    value=0,
    step=1
)

debt_ratio = st.sidebar.number_input(
    "Debt Ratio",
    min_value=0.0,
    value=0.40,
    step=0.01,
    help="Monthly debt obligations relative to income."
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
    "Number Of Real Estate Loans Or Lines",
    min_value=0,
    value=1,
    step=1
)

past_due_60_89 = st.sidebar.number_input(
    "Number Of Times 60–89 Days Past Due",
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

# -----------------------------
# Main layout
# -----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Input Data")
    st.dataframe(input_data, use_container_width=True)
    st.markdown(
        '<div class="small-note">Review the borrower profile before generating the prediction.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Quick Guide")
    st.write(
        """
        - Enter borrower values in the left sidebar.
        - Click **Predict Credit Risk**.
        - Review the prediction result and probability.
        - Use the SHAP chart to understand feature influence.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Credit Risk", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("The borrower is likely to experience serious delinquency (Default Risk).")
        risk_label = "High Risk"
    else:
        st.success("The borrower is unlikely to experience serious delinquency (No Default Risk).")
        risk_label = "Lower Risk"

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Predicted Probability of Default", f"{probability:.2%}")
    with metric_col2:
        st.metric("Risk Category", risk_label)

    if probability >= 0.50:
        st.warning("This borrower shows a relatively elevated probability of serious delinquency.")
    elif probability >= 0.20:
        st.info("This borrower shows a moderate level of estimated credit risk.")
    else:
        st.info("This borrower shows a relatively low estimated credit risk.")

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # SHAP explanation
    # -----------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("SHAP Explanation")
    st.write("The chart below shows how strongly each feature influences the prediction.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Handle possible list output for some SHAP/model versions
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.summary_plot(
        shap_values_to_plot,
        input_data,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig, clear_figure=True)

    st.subheader("Borrower Feature Values")
    feature_display = input_data.T.rename(columns={0: "Value"})
    st.dataframe(feature_display, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
