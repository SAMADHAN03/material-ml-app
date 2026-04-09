import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: MUST BE THE FIRST STREAMLIT COMMAND ---
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
        border: none;
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
        st.error(f"Error loading model files: { e }")
        return None, None

model, feature_columns = load_assets()

# 4. CORE FUNCTIONS
def varshni(mat, T):
    params = {
        "ZnO": (3.44, 5.5e-4, 900),
        "Fe2O3": (2.2, 4.5e-4, 500),
        "CeO2": (3.2, 4.7e-4, 600)
    }
    Eg0, alpha, beta = params.get(mat, (3, 5e-4, 500))
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
    
    # Mean and Conductivity Calculation
    df_input["band_gap_expected"] = df_input[["band_gap_predicted","band_gap_api","band_gap_oqmd","band_gap_aflow","band_gap_theoretical"]].mean(axis=1)
    
    k = 8.617e-5 # Boltzmann constant
    sigma0 = 1e3 # Reference conductivity
    df_input["conductivity"] = sigma0 * np.exp(-df_input["band_gap_expected"] / (k * df_input["temp"]))
    return df_input

# 5. USER INTERFACE
st.title("🔬 Material ML Analysis Platform")
st.markdown("**Predicting Band Gap and Conductivity using specialized Machine Learning models**")
st.divider()

if model is not None:
    uploaded_file = st.file_uploader("📂 Upload Experimental Data (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # File Loading
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded_file)
            
            # Column Standardization
            df.columns = df.columns.str.strip().str.lower()
            
            # Flexible Mapping
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

                # 6. ATTRACTIVE RESULTS DISPLAY
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Band Gap", f"{df_results['band_gap_predicted'].mean():.2f} eV")
                m2.metric("Total Samples", len(df_results))
                m3.metric("Avg Conductivity", f"{df_results['conductivity'].mean():.1e} S/m")

                st.subheader("✅ Processed Results")
                st.dataframe(df_results, use_container_width=True)
                
                # Visualizations
                col_left, col_right = st.columns(2)
                with col_left:
                    fig1, ax1 = plt.subplots()
                    ax1.plot(df_results["band_gap_predicted"], marker='o', color='#007bff')
                    ax1.set_title("Predicted Band Gap")
                    ax1.set_ylabel("eV")
                    st.pyplot(fig1)
                
                with col_right:
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
else:
    st.warning("Please ensure model.pkl and features.pkl are present in the repository.")
