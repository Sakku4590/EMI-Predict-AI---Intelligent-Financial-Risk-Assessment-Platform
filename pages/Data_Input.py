import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_engineering import engineer_features

st.set_page_config(page_title="Data Input", page_icon="📝", layout="wide")

st.title("Enter Your Information")
st.markdown("---")

with st.form("user_input_form"):
    st.subheader("Personal Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        marital_status = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Married" if x == 1 else "Single")
    
    with col2:
        education = st.selectbox("Education", 
                                ["Graduate", "High School", "Post Graduate", "Professional"])
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, value=50000, step=1000)
        employment_type = st.selectbox("Employment Type", 
                                      ["Government", "Private", "Self-Employed"])
    
    with col3:
        years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
        company_type = st.selectbox("Company Type", 
                                   ["Large", "Mid-Size", "MNC", "Small", "Startup"])
        house_type = st.selectbox("House Type", ["Own", "Rented", "Other"])
    
    st.markdown("---")
    st.subheader("Financial Information")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, value=10000, step=500)
    
    with col5:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    
    with col6:
        bank_balance = st.number_input("Bank Balance (₹)", min_value=0, value=100000, step=5000)
    
    emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, value=50000, step=5000)
    
    st.markdown("---")
    submitted = st.form_submit_button("Submit & Calculate Features", use_container_width=True)
    
    if submitted:
        user_input = {
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'education': education,
            'monthly_salary': monthly_salary,
            'employment_type': employment_type,
            'years_of_employment': years_of_employment,
            'company_type': company_type,
            'house_type': house_type,
            'monthly_rent': monthly_rent,
            'credit_score': credit_score,
            'bank_balance': bank_balance,
            'emergency_fund': emergency_fund
        }
        
        df = pd.DataFrame([user_input])
        df_engineered = engineer_features(df)
        st.session_state.user_data = df_engineered
        
        st.success("Data submitted successfully! Navigate to 'Prediction' page to see results.")
        
        with st.expander("View Calculated Features"):
            st.dataframe(df_engineered.T, use_container_width=True)