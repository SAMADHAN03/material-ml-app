import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile

from sklearn.metrics import mean_absolute_error, r2_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Material ML Analysis", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_assets():
    return joblib.load("model.pkl"), joblib.load("features.pkl")

model, feature_columns = load_assets()

# -----------------------------
# VARSHNI MODEL (GENERIC)
# -----------------------------
def varshni(material, T):
    params = {
        "ZnO": (3.44, 5.5e-4, 900),
        "Fe2O3": (2.2, 4.5e-4, 500),
        "CeO2": (3.2, 4.7e-4, 600)
    }
    Eg0, alpha, beta = params.get(material, (3.0, 5e-4, 500))
    return Eg0 - (alpha * T**2) / (T + beta)

# -----------------------------
# PIPELINE
# -----------------------------
def run_pipeline(material, dopant, tempK, conc, size):

    df_input = pd.DataFrame([{
        "material": material,
        "dopant": dopant,
        "temp": tempK,
        "conc": conc,
        "particle_size": size
    }])

    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    predicted = model.predict(df_encoded)[0]
    theoretical = varshni(material, tempK)

    # External references (can upgrade later)
    oqmd = 3.10
    aflow = 3.05

    expected = np.mean([predicted, theoretical, oqmd, aflow])

    return theoretical, predicted, expected, oqmd, aflow

# -----------------------------
# PDF GENERATOR
# -----------------------------
def generate_pdf(df, fig):

    path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Material ML Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))

    data = [df.columns.tolist()] + df.values.tolist()
    elements.append(Table(data))

    img = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(img, bbox_inches='tight')
    elements.append(Spacer(1, 12))
    elements.append(Image(img, width=450, height=300))

    doc.build(elements)
    return path

# -----------------------------
# WORD GENERATOR
# -----------------------------
def generate_word(df):

    path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    doc = Document()

    doc.add_heading("Material ML Analysis Report", 0)

    table = doc.add_table(rows=1, cols=len(df.columns))

    for i, col in enumerate(df.columns):
        table.rows[0].cells[i].text = col

    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    doc.save(path)
    return path

# -----------------------------
# UI
# -----------------------------
st.title("🔬 Advanced Material ML System")

file = st.file_uploader("📂 Upload Your Dataset", type=["csv", "xlsx"])

if file:

    # READ DATA
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    st.subheader("📊 Raw Data")
    st.dataframe(df)

    # -----------------------------
    # COLUMN STANDARDIZATION
    # -----------------------------
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

    # -----------------------------
    # CLEAN DATA
    # -----------------------------
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
    df["particle_size"] = pd.to_numeric(df["particle_size"], errors="coerce")
    df["band_gap_exp"] = pd.to_numeric(df["band_gap_exp"], errors="coerce")

    df = df.dropna()
    df = df[df["temp_c"] > 0]

    # Convert °C → K
    df["temp_k"] = df["temp_c"] + 273

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df)

    # -----------------------------
    # RUN ANALYSIS
    # -----------------------------
    if st.button("🚀 Run Analysis"):

        results = []

        for _, row in df.iterrows():

            theo, pred, exp, oqmd, aflow = run_pipeline(
                row["material"],
                row["dopant"],
                row["temp_k"],
                row["conc"],
                row["particle_size"]
            )

            results.append({
                "Material": row["material"],
                "Dopant": row["dopant"],
                "Temperature (°C)": row["temp_c"],
                "Temperature (K)": row["temp_k"],
                "Conc (%)": row["conc"],
                "Particle Size (nm)": row["particle_size"],
                "Experimental (eV)": row["band_gap_exp"],
                "Predicted (eV)": pred,
                "Theoretical (eV)": theo,
                "Expected (eV)": exp,
                "OQMD (eV)": oqmd,
                "AFLOW (eV)": aflow,
                "Reference": row["reference"]
            })

        df_results = pd.DataFrame(results)

        st.subheader("📋 Results Table")
        st.dataframe(df_results)

        # -----------------------------
        # METRICS
        # -----------------------------
        mae = mean_absolute_error(df_results["Experimental (eV)"], df_results["Predicted (eV)"])
        r2 = r2_score(df_results["Experimental (eV)"], df_results["Predicted (eV)"])

        col1, col2 = st.columns(2)
        col1.metric("MAE", round(mae, 4))
        col2.metric("R²", round(r2, 4))

        # -----------------------------
        # GRAPH (PROPER SCALE)
        # -----------------------------
        fig, ax = plt.subplots()

        ax.plot(df_results["Temperature (K)"], df_results["Experimental (eV)"], marker='o', label="Experimental")
        ax.plot(df_results["Temperature (K)"], df_results["Predicted (eV)"], marker='s', label="Predicted")
        ax.plot(df_results["Temperature (K)"], df_results["Expected (eV)"], marker='^', label="Expected")

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Band Gap (eV)")
        ax.set_title("Band Gap vs Temperature")

        ax.legend()
        ax.grid(True)

        # Auto-scale
        y_min = df_results[["Experimental (eV)", "Predicted (eV)", "Expected (eV)"]].min().min()
        y_max = df_results[["Experimental (eV)", "Predicted (eV)", "Expected (eV)"]].max().max()
        ax.set_ylim(y_min - 0.2, y_max + 0.2)

        st.pyplot(fig)

        # -----------------------------
        # DOWNLOADS
        # -----------------------------
        csv = df_results.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv, "results.csv")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button("⬇ Download Graph", buf.getvalue(), "graph.png")

        pdf = generate_pdf(df_results, fig)
        with open(pdf, "rb") as f:
            st.download_button("📄 Download PDF", f, "report.pdf")

        word = generate_word(df_results)
        with open(word, "rb") as f:
            st.download_button("📝 Download Word", f, "report.docx")

        st.success("✅ Analysis Completed Successfully 🚀")
