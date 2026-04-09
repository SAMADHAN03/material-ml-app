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
        # -----------------------------
# UNIVERSAL FILE READER
# -----------------------------
try:
    if uploaded_file.name.endswith(".csv"):

        # Auto-detect separator (comma, semicolon, tab, etc.)
        df = pd.read_csv(uploaded_file, sep=None, engine='python')

    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    else:
        st.error("❌ Unsupported file format")
        st.stop()

except Exception as e:
    st.error(f"❌ File reading error: {e}")
    st.stop()
    # -----------------------------
# AUTO COLUMN DETECTION
# -----------------------------
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
        st.error(f"❌ Missing column for: {key}")
        st.stop()

# Rename to standard format
df = df.rename(columns={
    mapped_cols["material"]: "material",
    mapped_cols["dopant"]: "dopant",
    mapped_cols["temp"]: "temp",
    mapped_cols["conc"]: "conc",
    mapped_cols["particle_size"]: "particle_size"
})
# -----------------------------
# AUTO COLUMN DETECTION
# -----------------------------
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
        st.error(f"❌ Missing column for: {key}")
        st.stop()

# Rename to standard format
df = df.rename(columns={
    mapped_cols["material"]: "material",
    mapped_cols["dopant"]: "dopant",
    mapped_cols["temp"]: "temp",
    mapped_cols["conc"]: "conc",
    mapped_cols["particle_size"]: "particle_size"
})
# -----------------------------
# DATA FILTERING
# -----------------------------
df = df.dropna()

df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
df["particle_size"] = pd.to_numeric(df["particle_size"], errors="coerce")

df = df.dropna()

df = df[df["temp"] > 0]
df = df[df["conc"] >= 0]
df = df[df["particle_size"] > 0]

df = df.reset_index(drop=True)
else:
        df = pd.read_excel(uploaded_file)

    # Show data
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df)
    # -----------------------------
# -----------------------------
# STEP 3 + STEP 4 INSIDE UPLOAD BLOCK
# -----------------------------
if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Show data
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df)

    # -----------------------------
    # STEP 3: CLEAN DATA
    # -----------------------------
    df.columns = df.columns.str.strip().str.lower()

    required_cols = ["material", "dopant", "temp", "conc", "particle_size"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()

    df = df.dropna()

    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
    df["particle_size"] = pd.to_numeric(df["particle_size"], errors="coerce")

    df = df.dropna()

    df = df[df["temp"] > 0]
    df = df[df["conc"] >= 0]
    df = df[df["particle_size"] > 0]

    df = df.reset_index(drop=True)

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df)

    # -----------------------------
    # STEP 4: RUN ML
    # -----------------------------
    results = []

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

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------
    if len(results) > 0:

        df_results = pd.concat(results, ignore_index=True)

        # TABLE
        st.subheader("📊 Prediction Results")
        st.dataframe(df_results)

        # DOWNLOAD
        csv = df_results.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download All Results",
            data=csv,
            file_name="material_predictions.csv",
            mime="text/csv"
        )

        # GRAPH
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8,5))

        if "band_gap_exp" in df_results.columns:
            ax.plot(df_results["band_gap_exp"], marker='o', label="Experimental")

        ax.plot(df_results["band_gap_predicted"], marker='s', label="Predicted")
        ax.plot(df_results["band_gap_expected"], marker='^', label="Expected")

        ax.set_title("Band Gap Comparison")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Band Gap (eV)")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    else:
        st.error("❌ No valid data to process")
    
