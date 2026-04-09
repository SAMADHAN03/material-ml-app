import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: CONFIGURATION ---
st.set_page_config(page_title="Material ML Analysis", layout="wide")

# 2. VISUAL STYLING (Background & Logos)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stApp::before {
        content: 'Au Ag Cu Fe Zn Li Ne';
        position: fixed;
        top: 15%;
        left: 5%;
        font-size: 8rem;
        font-weight: bold;
        color: rgba(0, 0, 0, 0.03);
        z-index: -1;
        transform: rotate(-15deg);
        white-space: nowrap;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #007bff;
        color: white;
        width: 100%;
        font-weight: bold;
        border: none;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
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
    # Dictionary of known material constants
    params = {
        "ZnO": (3.44, 5.5e-4, 900),
        "Fe2O3": (2.2, 4.5e-4, 500),
        "CeO2": (3.2, 4.7e-4, 600)
    }
    # If material is unknown, use a standard semiconductor default
    Eg0, alpha, beta = params.get(mat, (3.0, 5.0e-4, 500))
    return Eg0 - (alpha * T**2) / (T + beta)

def run_pipeline(material, dopant, temp, conc, size):
    df_input = pd.DataFrame([{
        "material": material, "dopant": dopant, "temp": temp, 
        "conc": conc, "particle_size": size 
    }])
    
    # Baseline comparison values
    df_input["band_gap_api"] = 3.0
    df_input["band_gap_oqmd"] = 3.1
    df_input["band_gap_aflow"] = 3.05
    df_input["band_gap_theoretical"] = df_input.apply(lambda x: varshni(x["material"], x["temp"]), axis=1)
    
    # ML Prediction
    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    df_input["band_gap_predicted"] = model.predict(df_encoded)
    
    # Expected value (Ensemble Mean)
    df_input["band_gap_expected"] = df_input[["band_gap_predicted","band_gap_api","band_gap_oqmd","band_gap_aflow","band_gap_theoretical"]].mean(axis=1)
    
    k = 8.617e-5 # Boltzmann constant
    sigma0 = 1e3 # Reference conductivity
    df_input["conductivity"] = sigma0 * np.exp(-df_input["band_gap_expected"] / (k * df_input["temp"]))
    return df_input

# 5. USER INTERFACE
st.title("🔬 Material ML Analysis Platform")
st.markdown("**Comparison of Theoretical, Predicted, and Expected Band Gap values**")
st.divider()

if model is not None:
    uploaded_file = st.file_uploader("📂 Upload Experimental Data (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded_file)
            
            df.columns = df.columns.str.strip().str.lower()
            
            mapping = {
                "temp": ["temp", "temp_k", "temperature"],
                "conc": ["conc", "concentration", "dopant_conc"],
                "particle_size": ["size", "particle_size", "radius"]
            }
            
            for standard, aliases in mapping.items():
                for alias in aliases:
                    if alias in df.columns:
                        df = df.rename(columns={alias: standard})
                        break

            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 Run Analysis"):
                results = []
                progress_bar = st.progress(0)
                
                for i, row in df.iterrows():
                    res = run_pipeline(row["material"], row["dopant"], row["temp"], row["conc"], row["particle_size"])
                    results.append(res)
                    progress_bar.progress((i + 1) / len(df))
                
                df_results = pd.concat(results, ignore_index=True)

                # 6. ENHANCED RESULTS DISPLAY
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Predicted Gap", f"{df_results['band_gap_predicted'].mean():.2f} eV")
                m2.metric("Total Samples", len(df_results))
                m3.metric("Avg Conductivity", f"{df_results['conductivity'].mean():.1e} S/m")

                st.subheader("✅ Processed Results")
                st.dataframe(df_results, use_container_width=True)
                
                # --- UPDATED GRAPH WITH EXACT LABELS ---
                st.subheader("📈 Band Gap Comparison Analysis")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 1. Theoretical Plot
                ax.plot(df_results.index, df_results["band_gap_theoretical"], 
                        label="band_gap_theoretical", color="#FF5733", linestyle='--', marker='x', alpha=0.7)
                
                # 2. Predicted Plot
                ax.plot(df_results.index, df_results["band_gap_predicted"], 
                        label="band_gap_predicted", color="#007bff", marker='o', linewidth=2)
                
                # 3. Expected Plot
                ax.plot(df_results.index, df_results["band_gap_expected"], 
                        label="band_gap_expected", color="#28a745", marker='s', alpha=0.8)

                ax.set_xlabel("Sample Index")
                ax.set_ylabel("Band Gap (eV)")
                ax.set_title("Theoretical vs Predicted vs Expected Comparison")
                
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(title="Band Gap Metrics", loc='upper right', frameon=True, shadow=True)
                
                st.pyplot(fig)
                
                # Download Button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Detailed Results", data=csv, file_name="material_analysis_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
else:
    st.warning("Please ensure model.pkl and features.pkl are present in the repository.")
