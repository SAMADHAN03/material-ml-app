import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("model.pkl")
feature_columns = joblib.load("features.pkl")

def run_pipeline(material, dopant, temp, conc, size):
    df_input = pd.DataFrame([{
        "material": material,
        "dopant": dopant,
        "temp": temp,
        "conc": conc,
        "particle_size": size
    }])

    df_input["band_gap_api"] = 3.0
    df_input["band_gap_oqmd"] = 3.1
    df_input["band_gap_aflow"] = 3.05

    def varshni(mat, T):
        params = {
            "ZnO": (3.44, 5.5e-4, 900),
            "Fe2O3": (2.2, 4.5e-4, 500),
            "CeO2": (3.2, 4.7e-4, 600)
        }
        Eg0, alpha, beta = params.get(mat, (3,5e-4,500))
        return Eg0 - (alpha*T**2)/(T+beta)

    df_input["band_gap_theoretical"] = df_input.apply(
        lambda x: varshni(x["material"], x["temp"]), axis=1
    )

    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    df_input["band_gap_predicted"] = model.predict(df_encoded)

    df_input["band_gap_expected"] = df_input[
        ["band_gap_predicted","band_gap_api","band_gap_oqmd","band_gap_aflow","band_gap_theoretical"]
    ].mean(axis=1)

    k = 8.617e-5
    sigma0 = 1e3

    df_input["conductivity"] = sigma0 * np.exp(
        -df_input["band_gap_expected"]/(k*df_input["temp"])
    )

    df_input["log_conductivity"] = np.log10(df_input["conductivity"])

    return df_input

st.title("🔬 Advanced Material ML System")
uploaded_file = st.file_uploader(
    "📂 Upload Experimental Data (CSV or Excel)",
    type=["csv", "xlsx"]
)
material = st.selectbox("Material", ["ZnO", "Fe2O3", "CeO2"])
dopant = st.selectbox("Dopant", ["Al", "Cu", "Ga", "Zn"])
temp = st.slider("Temperature (K)", 250, 500, 300)
conc = st.slider("Concentration (%)", 1.0, 10.0, 5.0)
size = st.slider("Particle Size (nm)", 5, 50, 20)

if st.button("Predict"):
    result = run_pipeline(material, dopant, temp, conc, size)
    st.dataframe(result)
