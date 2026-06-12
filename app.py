import streamlit as st
import pickle
import numpy as np

# Load model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")
st.title("🏦 Loan Approval Prediction System")
st.write("Fill in the details below to check loan eligibility.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_term = st.selectbox("Loan Term (months)", [360, 120, 180, 240, 300, 480])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

def encode(gender, married, dependents, education, self_employed, property_area):
    g = 1 if gender == "Male" else 0
    m = 1 if married == "Yes" else 0
    d = 3 if dependents == "3+" else int(dependents)
    e = 0 if education == "Graduate" else 1
    s = 1 if self_employed == "Yes" else 0
    p = 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0)
    return g, m, d, e, s, p

if st.button("🔍 Predict"):
    g, m, d, e, s, p = encode(gender, married, dependents,
                               education, self_employed, property_area)
    total_income = applicant_income + coapplicant_income
    loan_log = np.log(loan_amount + 1)
    income_log = np.log(total_income + 1)

    features = np.array([[g, m, d, e, s, applicant_income,
                          coapplicant_income, loan_amount,
                          loan_term, credit_history, p,
                          total_income, loan_log, income_log]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Loan Approved! Congratulations!")
    else:
        st.error("❌ Loan Not Approved. Better luck next time.")