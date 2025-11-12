import pickle
import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_models():
    try:
        # Load classification model
        classification_model = None
        try:
            classification_model = joblib.load('EMI_classification_Best_Model.pkl')
            st.info(f"Classification model loaded with joblib - Type: {type(classification_model)}")
        except:
            try:
                with open('EMI_classification_Best_Model.pkl', 'rb') as f:
                    classification_model = pickle.load(f)
                st.info(f"Classification model loaded with pickle - Type: {type(classification_model)}")
            except Exception as e:
                st.error(f"Failed to load classification model: {e}")
        
        # Load regression model
        regression_model = None
        try:
            regression_model = joblib.load('EMI_regression_Best_Model.pkl')
            st.info(f"Regression model loaded with joblib - Type: {type(regression_model)}")
        except:
            try:
                with open('EMI_regression_Best_Model.pkl', 'rb') as f:
                    regression_model = pickle.load(f)
                st.info(f"Regression model loaded with pickle - Type: {type(regression_model)}")
            except Exception as e:
                st.error(f"Failed to load regression model: {e}")
        
        # Load scaler
        scaler = None
        try:
            scaler = joblib.load('scaler.pkl')
            st.info(f"Scaler loaded with joblib - Type: {type(scaler)}")
        except:
            try:
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                st.info(f"Scaler loaded with pickle - Type: {type(scaler)}")
            except:
                try:
                    scaler = np.load('scaler.pkl', allow_pickle=True)
                    st.info(f"Scaler loaded with numpy - Type: {type(scaler)}")
                except Exception as e:
                    st.error(f"Failed to load scaler: {e}")
        
        # Check if models and scaler are actually objects not strings
        if isinstance(classification_model, str):
            st.error("Classification model loaded as string - file may be corrupted")
            classification_model = None
        
        if isinstance(regression_model, str):
            st.error("Regression model loaded as string - file may be corrupted")
            regression_model = None
            
        if isinstance(scaler, str):
            st.error("Scaler loaded as string - file may be corrupted")
            scaler = None
        
        return classification_model, regression_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None