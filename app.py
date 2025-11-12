import streamlit as st

st.set_page_config(
    page_title="EMI Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" EMI Prediction System")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("Classification Model")
    st.write("Predicts EMI eligibility:")
    st.write("Approved (>60% probability)")
    st.write(" High Risk (40-60% probability)")
    st.write(" Rejected (<40% probability)")

with col2:
    st.success("Regression Model")
    st.write("Predicts the exact EMI amount")

with col3:
    st.warning(" Feature Engineering")
    st.write("Automatic calculation of derived features")

st.markdown("---")
st.subheader("How It Works")
st.write("""
1. **Data Input**: Enter your personal and financial information
2. **Feature Engineering**: System automatically calculates derived features
3. **Prediction**: Get EMI eligibility and amount prediction
4. **Results**: View detailed analysis and recommendations
""")

st.markdown("---")
st.info("👈 Use the sidebar to navigate between pages")

st.markdown("---")
st.caption("Developed with Streamlit | Powered by Machine Learning")