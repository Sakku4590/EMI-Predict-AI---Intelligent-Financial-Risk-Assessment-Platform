import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title(" About EMI Prediction System")
st.markdown("---")

st.subheader("System Overview")
st.write("""
This EMI Prediction System uses machine learning models to predict:
1. **EMI Eligibility** (Classification)
2. **EMI Amount** (Regression)

The system takes basic user information and automatically calculates 40+ derived features
to make accurate predictions.
""")

st.markdown("---")
st.subheader("Models Used")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Classification Model**
    - Predicts approval/rejection
    - Provides probability scores
    - Uses 42 features
    """)

with col2:
    st.success("""
    **Regression Model**
    - Predicts EMI amount
    - Considers financial capacity
    - Uses standardized features
    """)

st.markdown("---")
st.subheader("Feature Engineering")
st.write("""
The system automatically calculates:
- Monthly expenses (school fees, travel, utilities, etc.)
- Financial ratios (debt-to-income, housing burden, etc.)
- Stability metrics (employment, financial stability)
- One-hot encoded categorical variables
""")

st.markdown("---")
st.subheader("Input Features (13)")
st.code("""
1. age
2. gender
3. marital_status
4. education
5. monthly_salary
6. employment_type
7. years_of_employment
8. company_type
9. house_type
10. monthly_rent
11. credit_score
12. bank_balance
13. emergency_fund
""")

st.markdown("---")
st.subheader("Calculated Features (29)")
st.code("""
- school_fees, college_fees, travel_expenses
- groceries_utilities, other_monthly_expenses
- total_monthly_expenses, savings_capacity
- max_monthly_emi, debt_to_income_ratio
- financial_stability, per_capita_income
- employment_stability, housing_burden_ratio
- loan_to_income_ratio, expance_to_income_ratio
- affordability_ratio
- One-hot encoded categorical features (13)
""")

st.markdown("---")
st.caption("Developed with Streamlit | Powered by Machine Learning")