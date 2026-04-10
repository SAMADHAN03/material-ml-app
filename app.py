import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from mp_api.client import MPRester  # New Import

# --- STEP 1: CONFIGURATION ---
st.set_page_config(page_title="Material ML Analysis", layout="wide")
API_KEY = "3CX5U54ckg2IfJV2lK5zRIrS76Kx2rX2" # Get this from materialsproject.org

# --- LOAD ASSETS ---
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

# --- CORE FUNCTIONS ---
@st.cache_data
def get_base_gap_from_api(formula):
    """Fetches 0K Band Gap from Materials Project"""
    try:
        with MPRester(API_KEY) as mpr:
            docs = mpr.summary.search(formula=formula, fields=["band_gap"])
            if docs:
                return docs[0].band_gap
    except:
        pass
    return 3.0 # Fallback default

def varshni_logic(material, T):
    """Adjusts API band gap for temperature"""
    # Standard constants for fallback
    params = {"ZnO": (5.5e-4, 900), "Fe2O3": (4.5e-4, 500), "CeO2": (4.7e-4, 600)}
    alpha, beta = params.get(material, (5.0e-4, 500))
    
    eg0 = get_base_gap_from_api(material)
    return eg0 - (alpha * T**2) / (T + beta)

def run_pipeline(material, dopant, temp, conc, size):
    df_input = pd.DataFrame([{
        "material": material, "dopant": dopant, 
        "temp": temp, "conc": conc, "particle_size": size
    }])
    
    # 1. Theoretical Calculation (API + Varshni)
    df_input["band_gap_theoretical"] = varshni_logic(material, temp)
    
    # 2. Simulated Other Sources
    df_input["band_gap_oqmd"], df_input["band_gap_aflow"] = 3.1, 3.05
    
    # 3. ML Prediction
    df_encoded = pd.get_dummies(df_input).reindex(columns=feature_columns, fill_value=0)
    df_input["band_gap_predicted"] = model.predict(df_encoded)
    
    # 4. Final Expected Value (Mean)
    df_input["band_gap_expected"] = df_input[[
        "band_gap_predicted", "band_gap_theoretical", 
        "band_gap_oqmd", "band_gap_aflow"
    ]].mean(axis=1)
    
    return df_input

# --- USER INTERFACE ---
st.title("🔬 Advanced Material ML Analysis")
st.divider()

if model is not None:
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # AUTO-CLEANING: Handle delimiters and formats
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
            else:
                df = pd.read_excel(uploaded_file)
            
            # Standardize Columns
            df.columns = df.columns.str.strip().str.lower()
            mapping = {
                "temp": ["temp", "temperature", "temp_k", "t"],
                "material": ["mat", "formula", "composition"],
                "dopant": ["doped_with", "dopant_type"],
                "conc": ["conc", "concentration"],
                "particle_size": ["size", "particle_size"]
            }
            for standard, aliases in mapping.items():
                for alias in aliases:
                    if alias in df.columns:
                        df = df.rename(columns={alias: standard})
                        break
            
            # Numeric conversion
            for col in ["temp", "conc", "particle_size"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['material', 'temp'])

            if st.button("🚀 Run Analysis"):
                results = []
                bar = st.progress(0)
                for i, row in df.iterrows():
                    results.append(run_pipeline(row["material"], row["dopant"], row["temp"], row["conc"], row["particle_size"]))
                    bar.progress((i + 1) / len(df))
                
                df_results = pd.concat(results, ignore_index=True)

                # --- METRICS (Conductivity Removed) ---
                c1, c2 = st.columns(2)
                c1.metric("Avg Predicted Gap", f"{df_results['band_gap_predicted'].mean():.2f} eV")
                c2.metric("Total Samples", len(df_results))

                st.dataframe(df_results, use_container_width=True)

                # --- UPDATED GRAPH: BAND GAP VS TEMPERATURE ---
                st.subheader("📈 Band Gap vs. Temperature")
                df_plot = df_results.sort_values(by="temp")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_plot["temp"], df_plot["band_gap_theoretical"], label="Theoretical (Varshni)", color="#E63946", linestyle='--', marker='x')
                ax.plot(df_plot["temp"], df_plot["band_gap_predicted"], label="ML Predicted", color="#457B9D", marker='o', linewidth=2)
                ax.plot(df_plot["temp"], df_plot["band_gap_expected"], label="Combined Expected", color="#2A9D8F", marker='s', alpha=0.7)
                
                ax.set_xlabel("Temperature (K)", fontweight='bold')
                ax.set_ylabel("Band Gap (eV)", fontweight='bold')
                ax.legend(loc='best', shadow=True)
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"File Error: {e}")
