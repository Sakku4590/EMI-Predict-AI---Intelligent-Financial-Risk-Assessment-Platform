import streamlit as st
import sys
import os
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import load_models
from utils.feature_engineering import prepare_features_for_prediction

st.set_page_config(page_title="Prediction", page_icon="🎯", layout="wide")

st.title(" EMI Prediction Results")
st.markdown("---")

if 'user_data' not in st.session_state or st.session_state.user_data is None:
    st.warning("Please enter your data in the 'Data Input' page first!")
else:
    classification_model, regression_model, scaler = load_models()
    
    if classification_model is None or regression_model is None or scaler is None:
        st.error("Models not loaded. Please check model files.")
    else:
        df = st.session_state.user_data
        X = prepare_features_for_prediction(df)
        
        # Handle different scaler types
        try:
            # Convert to numpy array first
            X_array = X.values if hasattr(X, 'values') else np.array(X)
            
            # If scaler has transform method (StandardScaler, MinMaxScaler, etc.)
            if hasattr(scaler, 'transform'):
                X_scaled = scaler.transform(X)
            # If scaler is a numpy array (manual scaling parameters)
            elif isinstance(scaler, np.ndarray):
                st.info(f"Scaler is numpy array with shape: {scaler.shape}")
                # Check if it's a 2D array with mean and std
                if scaler.shape[0] == 2:
                    mean = scaler[0]
                    std = scaler[1]
                    X_scaled = (X_array - mean) / std
                else:
                    st.warning("Unknown scaler array format, using raw features")
                    X_scaled = X_array
            # If scaler is a dict with mean and std
            elif isinstance(scaler, dict):
                X_scaled = (X_array - scaler['mean']) / scaler['std']
            else:
                st.warning(f"Unknown scaler type: {type(scaler)}, using raw features")
                X_scaled = X_array
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            st.info("Proceeding without scaling...")
            X_scaled = X.values if hasattr(X, 'values') else np.array(X)
        
        # Verify models are loaded correctly
        if not hasattr(classification_model, 'predict'):
            st.error(f"Classification model is not a valid model object. Type: {type(classification_model)}")
            st.stop()
        
        if not hasattr(regression_model, 'predict'):
            st.error(f"Regression model is not a valid model object. Type: {type(regression_model)}")
            st.stop()
        
        with st.spinner("Making predictions..."):
            classification_pred = classification_model.predict(X_scaled)[0]
            classification_proba = classification_model.predict_proba(X_scaled)[0]
            regression_pred = regression_model.predict(X_scaled)[0]
        
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Classification Result")
            if classification_pred == 1:
                st.success("EMI APPROVED")
                st.metric("Approval Probability", f"{classification_proba[1]*100:.2f}%")
            else:
                st.error("EMI REJECTED")
                st.metric("Rejection Probability", f"{classification_proba[0]*100:.2f}%")
        
        with col2:
            st.markdown("Predicted EMI Amount")
            st.info(f"### ₹ {regression_pred:,.2f}")
            st.caption("Estimated Monthly EMI")
        
        st.markdown("---")
        
        st.subheader("Financial Summary")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Monthly Salary", f"₹ {df['monthly_salary'].iloc[0]:,.2f}")
            st.metric("Total Expenses", f"₹ {df['total_monthly_expenses'].iloc[0]:,.2f}")
        
        with col4:
            st.metric("Savings Capacity", f"₹ {df['savings_capacity'].iloc[0]:,.2f}")
            st.metric("Max Affordable EMI", f"₹ {df['max_monthly_emi'].iloc[0]:,.2f}")
        
        with col5:
            st.metric("Credit Score", f"{df['credit_score'].iloc[0]}")
            st.metric("Bank Balance", f"₹ {df['bank_balance'].iloc[0]:,.2f}")
        
        st.markdown("---")
        
        st.subheader("Financial Ratios")
        
        ratio_col1, ratio_col2, ratio_col3, ratio_col4 = st.columns(4)
        
        with ratio_col1:
            st.metric("Debt-to-Income", f"{df['debt_to_income_ratio'].iloc[0]:.2%}")
        
        with ratio_col2:
            st.metric("Housing Burden", f"{df['housing_burden_ratio'].iloc[0]:.2%}")
        
        with ratio_col3:
            st.metric("Expense-to-Income", f"{df['expance_to_income_ratio'].iloc[0]:.2%}")
        
        with ratio_col4:
            st.metric("Affordability", f"{df['affordability_ratio'].iloc[0]:.2%}")
        
        st.markdown("---")
        st.subheader("Recommendations")
        
        if classification_pred == 1:
            st.success("""
            - Your EMI application is likely to be approved
            - Ensure timely repayment to maintain good credit score
            - Consider building emergency fund to 6 months of expenses
            """)
        else:
            st.warning("""
            - Consider improving your credit score
            - Reduce monthly expenses to increase savings capacity
            - Build a stronger emergency fund before applying
            - Consider requesting a lower loan amount
            """)