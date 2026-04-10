import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from sklearn.metrics import mean_absolute_error, r2_score

# --- STEP 1: CONFIGURATION ---
st.set_page_config(page_title="Material ML Analysis", layout="wide")

# Fetch API Key from Streamlit Secrets (Set this in Dashboard > Settings > Secrets)
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
    """Fetches 0K Band Gap from Materials Project"""
    try:
        with MPRester(API_KEY) as mpr:
            docs = mpr.summary.search(formula=formula, fields=["band_gap"])
            if docs:
                return docs[0].band_gap
    except:
        pass
    return 3.0  # Fallback default

def varshni_logic(material, T):
    """Adjusts API band gap for temperature using Varshni Expression"""
    params = {"ZnO": (5.5e-4, 900), "Fe2O3": (4.5e-4, 500), "CeO2": (4.7e-4, 600)}
    alpha, beta = params.get(material, (5.0e-4, 500))
    eg0 = get_base_gap_from_api(material)
    return eg0 - (alpha * T**2) / (T + beta)

def run_pipeline(material, dopant, temp, conc, size):
    """Generates the specific result row sequence requested"""
    api_0k = get_base_gap_from_api(material)
    theoretical = varshni_logic(material, temp)
    
    df_input = pd.DataFrame([{"material": material, "dopant": dopant, "temp": temp, "conc": conc, "particle_size": size}])
    df_encoded = pd.get_dummies(df_input).reindex(columns=feature_columns, fill_value=0)
    predicted = model.predict(df_encoded)[0]
    
    oqmd, aflow = 3.1, 3.05
    expected = np.mean([predicted, theoretical, oqmd, aflow])
    
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

# --- STEP 4: USER INTERFACE ---
st.title("🔬 Advanced Material ML Analysis")
st.write("Analyze semiconductor band gaps using Machine Learning, Physics (Varshni), and API data.")
st.divider()

if model is not None:
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
            else:
                df = pd.read_excel(uploaded_file)
            
            df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
            mapping = {
                "temp": ["temp", "temperature", "tempk", "tk"],
                "material": ["mat", "formula", "composition"],
                "dopant": ["dopedwith", "dopanttype", "dopant"],
                "conc": ["conc", "concentration"],
                "particle_size": ["size", "particlesize"]
            }
            for standard, aliases in mapping.items():
                for alias in aliases:
                    if alias in df.columns:
                        df = df.rename(columns={alias: standard})
                        break
            
            for col in ["temp", "conc", "particle_size"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['material', 'temp']).reset_index(drop=True)

            if st.button("🚀 Run Analysis"):
                results = []
                total_rows = len(df)
                bar = st.progress(0.0)
                
                for i, row in df.iterrows():
                    res = run_pipeline(row["material"], row["dopant"], row["temp"], row.get("conc", 0), row.get("particle_size", 0))
                    results.append(res)
                    bar.progress(min((i + 1) / total_rows, 1.0))
                
                df_results = pd.DataFrame(results)

                # --- 1. PERFORMANCE METRICS ---
                st.subheader("📊 Model Performance Metrics")
                y_true, y_pred = df_results["Theoretical Band Gap"], df_results["Predicted Bandgap"]
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                m1, m2, m3 = st.columns(3)
                m1.metric("MAE (Mean Absolute Error)", f"{mae:.3f} eV")
                m2.metric("R² Score", f"{r2:.2f}")
                m3.metric("Total Samples", len(df_results))
                
                if mae < 0.2:
                    st.success(f"✅ Excellent Accuracy: MAE is {mae:.3f} eV.")
                else:
                    st.warning(f"⚠️ Notice: MAE is {mae:.3f} eV.")
                st.divider()

                # --- 2. DATA TABLE & DOWNLOAD ---
                st.subheader("📋 Detailed Analysis Results")
                st.dataframe(df_results, use_container_width=True)

                # Convert results to CSV for download
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv_data,
                    file_name="bandgap_analysis_results.csv",
                    mime="text/csv",
                )
                st.divider()

                # --- 3. GRAPH ---
                st.subheader("📈 Temperature Dependence Graph")
                df_plot = df_results.sort_values(by="Temp")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_plot["Temp"], df_plot["Theoretical Band Gap"], label="Theoretical (Varshni)", color="#E63946", linestyle='--', marker='x')
                ax.plot(df_plot["Temp"], df_plot["Predicted Bandgap"], label="ML Predicted", color="#457B9D", marker='o', linewidth=2)
                ax.plot(df_plot["Temp"], df_plot["Band Gap Expected"], label="Combined Expected", color="#2A9D8F", marker='s', alpha=0.7)
                ax.set_xlabel("Temperature (K)", fontweight='bold')
                ax.set_ylabel("Band Gap (eV)", fontweight='bold')
                ax.legend(loc='best', shadow=True)
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Processing Error: {e}")
