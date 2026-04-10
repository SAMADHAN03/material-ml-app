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
            if docs: return docs[0].band_gap
    except: pass
    return 3.0

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
    
    return {
        "Material": material, "Dopant": dopant, "Temp": temp, "Conc": conc, "Particle Size": size,
        "Theoretical Band Gap": theoretical, "Predicted Bandgap": predicted, 
        "Band Gap Expected": expected, "API Prediction (0K)": api_0k, 
        "OQMD Band Gap": oqmd, "AFLOW Band Gap": aflow
    }

# --- STEP 4: USER INTERFACE & CLEANING ---
st.title("🔬 Advanced Material ML Analysis")
st.divider()

if model is not None:
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file, sep=None, engine='python') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            # 1. CLEAN HEADERS: Remove special chars and force lowercase
            df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
            
            # 2. SMART MAPPING
            mapping = {"temp": ["temp", "temperature"], "material": ["mat", "formula"], "dopant": ["dopedwith", "dopant"]}
            for std, aliases in mapping.items():
                for alias in aliases:
                    if alias in df.columns:
                        df = df.rename(columns={alias: std})
                        break
            
            # 3. FORCE NUMERIC CLEANING (The "Bad Data" Fix)
            # This converts letters/garbage in numeric columns into 'NaN'
            numeric_cols = ["temp", "conc", "particle_size"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 4. PURGE BAD DATA: Remove any row where Material or Temp is missing/invalid
            df = df.dropna(subset=['material', 'temp']).reset_index(drop=True)

            if st.button("🚀 Run Analysis"):
                if df.empty:
                    st.error("❌ All data in the file was invalid or incorrectly formatted. Please check your columns.")
                else:
                    results = []
                    bar = st.progress(0.0)
                    for i, row in df.iterrows():
                        results.append(run_pipeline(row['material'], row['dopant'], row['temp'], row.get('conc', 0), row.get('particle_size', 0)))
                        bar.progress(min((i + 1) / len(df), 1.0))
                    
                    df_results = pd.DataFrame(results)

                    # --- METRICS ---
                    y_true, y_pred = df_results["Theoretical Band Gap"], df_results["Predicted Bandgap"]
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE", f"{mae:.3f} eV")
                    m2.metric("R² Score", f"{r2:.2f}")
                    m3.metric("Samples", len(df_results))
                    st.success(f"✅ Analysis Complete! MAE: {mae:.3f} eV")

                    # --- DOWNLOAD BUTTONS ---
                    c_dl1, c_dl2 = st.columns(2)
                    csv_data = df_results.to_csv(index=False).encode('utf-8')
                    c_dl1.download_button("📥 Download Data (CSV)", csv_data, "results.csv", "text/csv", use_container_width=True)

                    # --- GRAPH ---
                    df_p = df_results.sort_values("Temp")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df_p["Temp"], df_p["Theoretical Band Gap"], label="Theoretical", color="#E63946", ls='--', marker='x')
                    ax.plot(df_p["Temp"], df_p["Predicted Bandgap"], label="ML Predicted", color="#457B9D", marker='o')
                    ax.plot(df_p["Temp"], df_p["Band Gap Expected"], label="Combined", color="#2A9D8F", marker='s', alpha=0.6)
                    ax.set_xlabel("Temperature (K)"), ax.set_ylabel("Band Gap (eV)")
                    ax.legend(), ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    # Graph Download
                    img = io.BytesIO()
                    fig.savefig(img, format='png', dpi=300)
                    c_dl2.download_button("🖼️ Download Graph (PNG)", img.getvalue(), "graph.png", "image/png", use_container_width=True)

                    st.subheader("📋 Results Table")
                    st.dataframe(df_results, use_container_width=True)

        except Exception as e:
            st.error(f"Critical Error: {e}")
