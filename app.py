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
API_KEY = st.secrets.get("MP_API_KEY", "YOUR_FALLBACK_KEY")

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
    api_0k = get_base_gap_from_api(material)
    theoretical = varshni_logic(material, temp)
    
    df_input = pd.DataFrame([{"material": material, "dopant": dopant, "temp": temp, "conc": conc, "particle_size": size}])
    df_encoded = pd.get_dummies(df_input).reindex(columns=feature_columns, fill_value=0)
    predicted = model.predict(df_encoded)[0]
    
    oqmd, aflow = 3.1, 3.05
    expected = np.mean([predicted, theoretical, oqmd, aflow])
    
    # Return specific ordered dictionary
    return {
        "Material": material,
        "Dopant": dopant,
        "Temp": temp,
        "Conc": conc,
        "Particle Size": size,
        "Theoretical Band Gap": theoretical,
        "Predicted Bandgap": predicted,
        "Band Gap Expected": expected,
        "API Prediction (0K)": api_0k,
        "OQMD Band Gap": oqmd,
        "AFLOW Band Gap": aflow
    }

# --- STEP 4: UI & DATA CLEANING ---
st.title("🔬 Advanced Material ML Analysis")
st.divider()

if model is not None:
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
            mapping = {"temp": ["temp", "temperature"], "material": ["mat", "formula"], "dopant": ["dopedwith", "dopant"]}
            
            for std, aliases in mapping.items():
                for alias in aliases:
                    if alias in df.columns:
                        df = df.rename(columns={alias: std})
                        break
            
            # --- ROBUST DATA CLEANING ---
            initial_count = len(df)
            
            # 1. Convert columns to numeric, forcing errors to 'NaN' (None)
            for col in ["temp", "conc", "particle_size"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 2. Remove 'None' / NaN in essential columns
            df = df.dropna(subset=['material', 'temp'])
            
            # 3. Remove Invalid/Negative Temperatures (Physical Constraint for Kelvin)
            df = df[df['temp'] >= 0].reset_index(drop=True)
            
            # Notification of cleaned rows
            removed = initial_count - len(df)
            if removed > 0:
                st.warning(f"🧹 Cleaned {removed} invalid rows (missing data or negative temperatures).")

            if st.button("🚀 Run Analysis"):
                if df.empty:
                    st.error("❌ No valid data found. Check your file format or ensure valid temperatures.")
                else:
                    results = []
                    bar = st.progress(0.0)
                    for i, row in df.iterrows():
                        results.append(run_pipeline(
                            row['material'], 
                            row.get('dopant', 'None'), 
                            row['temp'], 
                            row.get('
