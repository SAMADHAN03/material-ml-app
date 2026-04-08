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
if uploaded_file is not None:

    import pandas as pd

    # Check file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Show data
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df)
    # -----------------------------
# STEP 3: CLEAN & VALIDATE DATA
# -----------------------------

# Standardize column names (very important)
df.columns = df.columns.str.strip().str.lower()

# Required columns
required_cols = ["material", "dopant", "temp", "conc", "particle_size"]

# Check missing columns
missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# Remove empty rows
df = df.dropna()

# Convert numeric columns
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
df["particle_size"] = pd.to_numeric(df["particle_size"], errors="coerce")

# Remove invalid numeric rows
df = df.dropna()

# Filter unrealistic values
df = df[df["temp"] > 0]
df = df[df["conc"] >= 0]
df = df[df["particle_size"] > 0]

# Reset index
df = df.reset_index(drop=True)

# Show cleaned data
st.subheader("🧹 Cleaned Data")
st.dataframe(df)
# -----------------------------
# STEP 4: RUN ML ON ALL DATA
# -----------------------------

results = []

# Loop through each row
for i, row in df.iterrows():
    try:
        res = run_pipeline(
            row["material"],
            row["dopant"],
            row["temp"],
            row["conc"],
            row["particle_size"]
        )
        results.append(res)

    except Exception as e:
        st.warning(f"⚠ Skipping row {i}: {e}")

# Combine all results
if len(results) > 0:
    df_results = pd.concat(results, ignore_index=True)

    st.subheader("🔬 Prediction Results")
    st.dataframe(df_results)
    # -----------------------------
# DOWNLOAD RESULTS
# -----------------------------

csv = df_results.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇ Download Results as CSV",
    data=csv,
    file_name="material_predictions.csv",
    mime="text/csv"
)

else:
    st.error("❌ No valid data to process")
material = st.selectbox("Material", ["ZnO", "Fe2O3", "CeO2"])
dopant = st.selectbox("Dopant", ["Al", "Cu", "Ga", "Zn"])
temp = st.slider("Temperature (K)", 250, 500, 300)
conc = st.slider("Concentration (%)", 1.0, 10.0, 5.0)
size = st.slider("Particle Size (nm)", 5, 50, 20)

if st.button("Predict"):
    result = run_pipeline(material, dopant, temp, conc, size)
    st.dataframe(result)
