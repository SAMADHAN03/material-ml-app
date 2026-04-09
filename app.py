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
        width: 100%; font-weight: bold; border: none; padding: 12px;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #0056b3; transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

# 3. LOAD ASSETS
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
    df_input["conductivity"] = sigma0 * np.exp(-df_input["band_gap_expected"] / (k * df_input["temp"]))
    return df_input

# 5. USER INTERFACE
st.title("🔬 Material ML Analysis Platform")
st.markdown("**Predictive Analysis of Band Gap and Conductivity**")
st.divider()

if model is not None:
    uploaded_file = st.file_uploader("📂 Upload Experimental Data (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python') if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            
            mapping = {"temp": ["temp", "temp_k", "temperature"], "conc": ["conc", "concentration"], "particle_size": ["size", "particle_size"]}
            for standard, aliases in mapping.items():
                for alias in aliases:
                    if alias in df.columns:
                        df = df.rename(columns={alias: standard}); break

            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 Run Comprehensive Analysis"):
                results = []
                bar = st.progress(0)
                for i, row in df.iterrows():
                    results.append(run_pipeline(row["material"], row["dopant"], row["temp"], row["conc"], row["particle_size"]))
                    bar.progress((i + 1) / len(df))
                
                df_results = pd.concat(results, ignore_index=True)

                # Metrics Section
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Predicted Gap", f"{df_results['band_gap_predicted'].mean():.2f} eV")
                c2.metric("Total Samples", len(df_results))
                c3.metric("Avg Conductivity", f"{df_results['conductivity'].mean():.1e} S/m")

                # --- 1. DATA TABLE FIRST ---
                st.subheader("✅ Full Result Data")
                st.dataframe(df_results, use_container_width=True)

                # --- 2. GRAPH SECOND ---
                st.subheader("📈 Band Gap Comparison")
                df_results['Material Composition'] = df_results['material'].astype(str) + " (" + df_results['dopant'].astype(str) + ")"
                
                fig, ax = plt.subplots(figsize=(10, 5))
                x_vals = range(len(df_results))
                
                # Plotting calls correctly closed
                ax.plot(x_vals, df_results["band_gap_theoretical"], label="band_gap_theoretical", color="#E63946", linestyle='--', marker='x', alpha=0.8)
                ax.plot(x_vals, df_results["band_gap_predicted"], label="band_gap_predicted", color="#457B9D", marker='o', linewidth=2.5)
                ax.plot(x_vals, df_results["band_gap_expected"], label="band_gap_expected", color="#2A9D8F", marker='s', alpha=0.9)

                ax.set_xticks(x_vals)
                ax.set_xticklabels(df_results['Material Composition'], rotation=0)
                
                ax.set_ylabel("Band Gap (eV)", fontsize=10, fontweight='bold')
                ax.set_xlabel("Material Composition", fontsize=10, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='best', frameon=True, shadow=True)
                plt.tight_layout()
                st.pyplot(fig)

                # --- 3. DOWNLOAD CENTER ---
                st.divider()
                st.subheader("⬇️ Download Center")
                d1, d2 = st.columns(2)
                with d1:
                    csv_data = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Data (CSV)", data=csv_data, file_name="material_results.csv", use_container_width=True)
                with d2:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300)
                    st.download_button("🖼️ Download Graph (PNG)", data=buf.getvalue(), file_name="comparison_chart.png", use_container_width=True)

        except Exception as e:
            st.error(f"Analysis Error: {e}")
else:
    st.warning("Ensure model.pkl and features.pkl are in the repository.")
