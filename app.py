
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk Assessment Engine", layout="wide")

@st.cache_resource
def load_artifacts():
    return joblib.load('credit_risk_model.joblib')

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']

st.title("💳 Commercial Credit Risk & Loan Default Scoring Engine")
st.markdown("Adjust borrower parameters on the left sidebar to evaluate default risk in real time.")
st.markdown("---")

st.sidebar.header("Borrower Financial Profile")
revolving_utilization = st.sidebar.slider("Revolving Line Utilization Rate", 0.0, 1.5, 0.35, step=0.05)
age = st.sidebar.slider("Borrower Age", 21, 75, 38)
debt_ratio = st.sidebar.slider("Debt-to-Income Ratio (DTI)", 0.0, 5.0, 0.45, step=0.1)
monthly_income = st.sidebar.number_input("Monthly Income ($)", 1000, 30000, 6500, step=500)
open_credit_lines = st.sidebar.slider("Number of Open Credit Lines", 1, 20, 5)
past_due_days = st.sidebar.selectbox("Times Past Due (30-59 Days)", [0, 1, 2, 3, 4])

# Column names match feature_names exactly (lowercase 'of' in NumberofOpenCreditLines)
raw_input = pd.DataFrame({
    'RevolvingUtilization': [revolving_utilization],
    'Age': [age],
    'DebtRatio': [debt_ratio],
    'MonthlyIncome': [monthly_income],
    'NumberofOpenCreditLines': [open_credit_lines],
    'PastDue30to59Days': [past_due_days]
})

st.subheader("Selected Applicant Overview")
st.dataframe(raw_input, use_container_width=True)

if st.button("Evaluate Credit Application", type="primary"):
    # Ensure columns match feature_names ordering
    scaled_input = scaler.transform(raw_input[feature_names])
    default_prob = model.predict_proba(scaled_input)[0][1]
    
    st.markdown("---")
    st.subheader("Underwriting Decision Output")
    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{default_prob * 100:.1f}%")
    
    if default_prob >= 0.40:
        col2.error("❌ Decision: REJECTED")
        col3.warning("Reason: Risk exceeds tolerance threshold (40%).")
    else:
        col2.success("✅ Decision: APPROVED")
        col3.info("Reason: Risk falls within acceptable guidelines.")
