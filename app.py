import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Model and Features
try:
    model = joblib.load("model.pkl")
    feature_columns = joblib.load("features.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

def run_pipeline(material, dopant, temp, conc, size):
    df_input = pd.DataFrame([{
        "material": material, "dopant": dopant, "temp": temp, 
        "conc": conc, "particle_size": size 
    }])
    
    # Static API/Database values for comparison
    df_input["band_gap_api"] = 3.0
    df_input["band_gap_oqmd"] = 3.1
    df_input["band_gap_aflow"] = 3.05

    def varshni(mat, T):
        params = {
            "ZnO": (3.44, 5.5e-4, 900),
            "Fe2O3": (2.2, 4.5e-4, 500),
            "CeO2": (3.2, 4.7e-4, 600)
        }
        Eg0, alpha, beta = params.get(mat, (3, 5e-4, 500))
        return Eg0 - (alpha * T**2) / (T + beta)

    df_input["band_gap_theoretical"] = df_input.apply(lambda x: varshni(x["material"], x["temp"]), axis=1)
    
    # ML Prediction
    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    df_input["band_gap_predicted"] = model.predict(df_encoded)
    
    # Calculate Expected (Mean) and Conductivity
    df_input["band_gap_expected"] = df_input[["band_gap_predicted","band_gap_api","band_gap_oqmd","band_gap_aflow","band_gap_theoretical"]].mean(axis=1)
    
    k = 8.617e-5
    sigma0 = 1e3
    df_input["conductivity"] = sigma0 * np.exp(-df_input["band_gap_expected"] / (k * df_input["temp"]))
    df_input["log_conductivity"] = np.log10(df_input["conductivity"])
    
    return df_input

st.title("🔬 Advanced Material ML System")

uploaded_file = st.file_uploader("📂 Upload Experimental Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # 1. UNIVERSAL FILE READER
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded_file)
        
        # Standardize column casing
        df.columns = df.columns.str.strip().str.lower()
        
        # 2. AUTO COLUMN DETECTION
        column_map = {
            "material": ["material", "material_type"],
            "dopant": ["dopant", "dopant_type"],
            "temp": ["temp", "temperature", "temp_k"],
            "conc": ["conc", "concentration", "dopant_conc"],
            "particle_size": ["particle_size", "size", "particle"]
        }

        def find_column(possible_names):
            for col in df.columns:
                if col in possible_names:
                    return col
            return None

        mapped_cols = {}
        for key, options in column_map.items():
            col = find_column(options)
            if col:
                mapped_cols[key] = col
            else:
                st.error(f"❌ Missing required data for: {key}")
                st.stop()

        # Rename to standard format
        df = df.rename(columns={
            mapped_cols["material"]: "material",
            mapped_cols["dopant"]: "dopant",
            mapped_cols["temp"]: "temp",
            mapped_cols["conc"]: "conc",
            mapped_cols["particle_size"]: "particle_size"
        })

        # 3. DATA CLEANING
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
        df["particle_size"] = pd.to_numeric(df["particle_size"], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        
        # Validation filters
        df = df[(df["temp"] > 0) & (df["conc"] >= 0) & (df["particle_size"] > 0)]

        st.subheader("📊 Data Preview")
        st.dataframe(df)

        # 4. RUN ML PIPELINE
        if st.button("🚀 Run Predictions"):
            results = []
            progress_bar = st.progress(0)
            
            for i, row in df.iterrows():
                try:
                    res = run_pipeline(row["material"], row["dopant"], row["temp"], row["conc"], row["particle_size"])
                    results.append(res)
                except Exception as e:
                    st.warning(f"⚠ Skipping row {i}: {e}")
                progress_bar.progress((i + 1) / len(df))

            if results:
                df_results = pd.concat(results, ignore_index=True)
                
                st.subheader("✅ Prediction Results")
                st.dataframe(df_results)

                # Download Results
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(label="⬇ Download CSV", data=csv, file_name="material_predictions.csv", mime="text/csv")

                # Graphing
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_results["band_gap_predicted"], marker='s', label="Predicted", alpha=0.7)
                ax.plot(df_results["band_gap_expected"], marker='^', label="Expected (Avg)", alpha=0.7)
                ax.set_title("Band Gap Analysis")
                ax.set_ylabel("Band Gap (eV)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("❌ No valid results generated.")

    except Exception as e:
        st.error(f"❌ Processing error: {e}")

else:
    st.info("Please upload a CSV or Excel file to get started.")
