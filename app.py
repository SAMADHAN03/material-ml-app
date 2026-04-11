import streamlit as st
import joblib
import pandas as pd
from mp_api.client import MPRester

# 1. Load the exported files from your GitHub
model = joblib.load('hybrid_bandgap_model.pkl')
features = joblib.load('model_features.pkl')

st.title("🔬 Hybrid Materials Predictor")

# 2. User Input
mat_name = st.text_input("Enter Material Formula (e.g., ZnO)", "ZnO")
temp = st.slider("Temperature (K)", 100, 1000, 300)

if st.button("Calculate Hybrid Band Gap"):
    # Fetch real data from Materials Project
    with MPRester("3CX5U54ckg2IfJV2lK5zRIrS76Kx2rX2") as mpr:
        docs = mpr.materials.summary.search(formula=mat_name, fields=["band_gap"])
        api_gap = docs[0].band_gap if docs else 0.0

    # Prepare data for ML model
    # Note: Ensure your input_df matches the 'features' list exactly
    input_df = pd.DataFrame([[temp, api_gap]], columns=['temp', 'band_gap_api']) 
    
    ml_pred = model.predict(input_df)[0]
    
    # Final Hybrid Logic (The 60% error reduction formula)
    hybrid_val = (0.5 * ml_pred) + (0.5 * api_gap)
    
    st.success(f"Final Hybrid Result: {hybrid_val:.3f} eV")
