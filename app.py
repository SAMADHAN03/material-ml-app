import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io

from sklearn.metrics import mean_absolute_error, r2_score

# PDF + Word
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Material ML System", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, features, scaler

model, feature_columns, scaler = load_assets()

# -----------------------------
# SMART FILE READER
# -----------------------------
def read_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            try:
                df = pd.read_csv(uploaded_file)
            except:
                df = pd.read_csv(uploaded_file, encoding="latin1")
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# -----------------------------
# CLEAN DATA
# -----------------------------
def clean_data(df):
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Material": "material",
        "Dopant": "dopant",
        "Dopant Concentration (%)": "conc",
        "Temperature (°C)": "temp_c",
        "Particle size (nm)": "particle_size",
        "Experimental band gap (eV)": "band_gap_exp",
        "Reference": "reference"
    })

    # Convert numeric
    for col in ["temp_c", "conc", "particle_size", "band_gap_exp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["temp_c", "band_gap_exp"])

    df = df[df["temp_c"] > 0]
    df = df[df["conc"] >= 0]
    df = df[df["particle_size"] > 0]

    df["temp"] = df["temp_c"] + 273

    return df

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def prepare_features(df):
    df["inv_temp"] = 1 / df["temp"]
    df["log_size"] = np.log(df["particle_size"])
    df["conc_sq"] = df["conc"] ** 2

    df_model = df.drop(columns=["reference"], errors="ignore")

    df_encoded = pd.get_dummies(df_model, columns=["material", "dopant"])

    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    X = df_encoded.astype(float)
    X_scaled = scaler.transform(X)

    return X_scaled

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🔬 Advanced Material ML System")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:

    df = read_file(uploaded_file)

    if df is not None:

        st.subheader("📊 Raw Data")
        st.dataframe(df)

        df = clean_data(df)

        st.subheader("🧹 Cleaned Data")
        st.dataframe(df)

        if st.button("🚀 Run Prediction"):

            X_scaled = prepare_features(df)

            predictions = model.predict(X_scaled)

            df["Predicted Band Gap"] = predictions

            df["Expected Band Gap"] = (
                df["Predicted Band Gap"] + df["band_gap_exp"]
            ) / 2

            # Metrics
            mae = mean_absolute_error(df["band_gap_exp"], predictions)
            r2 = r2_score(df["band_gap_exp"], predictions)

            st.subheader("📊 Results")
            st.dataframe(df)

            st.success(f"MAE: {mae:.4f}")
            st.success(f"R²: {r2:.4f}")

            # -----------------------------
            # GRAPH
            # -----------------------------
            fig, ax = plt.subplots()

            ax.plot(df["temp"], df["band_gap_exp"], label="Experimental", marker="o")
            ax.plot(df["temp"], df["Predicted Band Gap"], label="Predicted", marker="s")
            ax.plot(df["temp"], df["Expected Band Gap"], label="Expected", marker="^")

            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Band Gap (eV)")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

            # -----------------------------
            # DOWNLOAD CSV
            # -----------------------------
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download CSV",
                data=csv,
                file_name="results.csv"
            )

            # -----------------------------
            # PDF GENERATION
            # -----------------------------
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)

            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Material ML Results", styles["Title"]))
            elements.append(Spacer(1, 12))

            table_data = [df.columns.tolist()] + df.values.tolist()
            table = Table(table_data)
            elements.append(table)

            doc.build(elements)

            st.download_button(
                "⬇ Download PDF",
                data=buffer.getvalue(),
                file_name="results.pdf"
            )

            # -----------------------------
            # WORD FILE
            # -----------------------------
            docx = Document()
            docx.add_heading("Material ML Results", 0)

            for i in range(len(df)):
                docx.add_paragraph(str(df.iloc[i].to_dict()))

            doc_buffer = io.BytesIO()
            docx.save(doc_buffer)

            st.download_button(
                "⬇ Download Word",
                data=doc_buffer.getvalue(),
                file_name="results.docx"
            )
