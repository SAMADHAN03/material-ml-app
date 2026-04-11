import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the brain of the app
model = joblib.load('hybrid_bandgap_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('model_features.pkl')

# UI Elements
st.title("Material Band Gap Predictor")
temp = st.number_input("Enter Temperature (K)", value=300)
# ... add other inputs like material choice ...

if st.button("Predict"):
    # Create input DataFrame
    input_df = pd.DataFrame([[temp, ...]], columns=features)
    
    # SCALE the data before predicting
    scaled_input = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(scaled_input)
    st.success(f"Hybrid Predicted Band Gap: {prediction[0]:.4f} eV")
