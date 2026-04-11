import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from mp_api.client import MPRester
from sklearn.metrics import mean_absolute_error, r2_score

# --- STEP 1: CONFIGURATION ---
st.set_page_config(page_title="Material ML Analysis", layout="wide")
# Replace with your actual key or set in Streamlit Secrets
API_KEY = st.secrets.get("MP_API_KEY", "3CX5U54ckg2IfJV2lK5zRIrS76Kx2rX2")

# --- STEP 2: LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.pkl")
        features = joblib.load("features.pkl")
        return model, features
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, feature_columns = load_assets()

# --- STEP 3: CORE FUNCTIONS ---
@st.cache_data
def get_base_gap_from_api(formula):
    try:
        with MPRester(API_KEY) as mpr:
            docs = mpr.summary.search(formula=formula, fields=["band_gap"])
            if docs:
                return docs[0].band_gap
    except:
        pass
    return 3.0  # Fallback default

def varshni_logic(material, T):
    params = {"ZnO": (5.5e-4, 900), "Fe2O3": (4.5e-4, 500), "CeO2": (4.7e-4, 600)}
    alpha, beta = params.get(material, (5.0e-4, 500))
    eg0 = get_base_gap_from_api(material)
    return eg0 - (alpha * T**2) / (T + beta)

def run_pipeline(material, dopant, temp, conc, size):
    theoretical = varshni_logic(material, temp)
    df_input = pd.DataFrame([{
        "material": material, 
        "dopant": dopant, 
        "temp": temp, 
        "conc": conc, 
        "particle_size": size
    }])
    df_encoded = pd.get_dummies(df_input).reindex(columns=feature_columns, fill_value=0)
    predicted = model.predict(df_encoded)[0]
    
    # Supplemental data for comparison
    oqmd, aflow = 3.1, 3.05
    expected = np.mean([predicted, theoretical, oqmd, aflow])
    
    return {
        "Material": material, "Dopant": dopant, "Temp": temp, "Conc": conc, 
        "Particle Size": size, "Theoretical_Gap": theoretical, 
        "Predicted_Gap": predicted, "Expected_Gap": expected
    }

# --- STEP 4: UI ---
st.title("🔬 Advanced Material ML Analysis")
st.markdown("Upload your dataset to calculate band gaps and download results.")

if model is not None:
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Load Data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # --- ROBUST DATA CLEANING ---
            df.columns = df.columns.str.strip().str.lower()
            
            # Convert numeric columns and handle non-numeric junk data
            numeric_cols = ["temp", "conc", "particle_size"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            initial_count = len(df)
            # Drop rows with missing critical info or negative temperature
            df = df.dropna(subset=['material', 'temp'])
            df = df[df['temp'] >= 0].reset_index(drop=True
