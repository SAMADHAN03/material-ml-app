import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- STEP 1: CONFIGURATION ---
st.set_page_config(page_title="Material ML Analysis", layout="wide")

# 2. VISUAL STYLING
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stApp::before {
        content: 'Au Ag Cu Fe Zn Li Ne';
        position: fixed; top: 15%; left: 5%; font-size: 8rem;
        font-weight: bold; color: rgba(0, 0, 0, 0.03);
        z-index: -1; transform: rotate(-15deg); white-space: nowrap;
    }
    .stButton>button {
        border-radius: 20px; background-color: #007bff; color: white;
        width: 100%; font-weight: bold; border: none; padding: 10px;
    }
    .stButton>button:hover { background-color: #0056b3; }
    </style>
    """, unsafe_allow_html=True)

# 3. LOAD MODEL AND FEATURES
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

# 4. CORE FUNCTIONS
def varshni(mat, T):
    params = {"ZnO": (3.44, 5.5e-4, 900), "Fe2O3": (2.2, 4.5e-4, 500), "CeO2": (3.2, 4.7e-4, 600)}
    Eg0, alpha, beta = params.get(mat, (3.0, 5.0e-4, 500))
    return Eg0 - (alpha * T**2) / (T + beta)

def run_pipeline(material, dopant, temp, conc, size):
    df_input = pd.DataFrame([{"material": material, "dopant": dopant, "temp": temp, "conc": conc, "particle_size": size}])
    df_input["band_gap_api"], df_input["band_gap_oqmd"], df_input["band_gap_aflow"] = 3.0, 3.1, 3.05
    df_input["band_gap_theoretical"] = df_input.apply(lambda x: varshni(x["material"], x["temp"]), axis=1)
    
    df_encoded = pd.get_dummies(df_input).reindex(columns=feature_columns, fill_value=0)
    df_input["band_gap_predicted"] = model.predict(df_encoded)
    df_input["band_gap_expected"] = df_input[["band_gap_predicted","band_gap_api","band_gap_oqmd","band_gap_aflow","band_gap_theoretical"]].mean(axis=1)
    
    k, sigma0 = 8.617e-5, 1e3
    df_input["conductivity"] = sigma0 * np.exp(-df_input["band_gap_expected"]
