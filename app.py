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
# VARSHNI MODEL
# -----------------------------
def varshni(T):
    Eg0, alpha, beta = 3.44, 5.5e-4, 900
    return Eg0 - (alpha * T**2) / (T + beta)

# -----------------------------
# PIPELINE
# -----------------------------
def run_pipeline(tempK, conc, size):

    df_input = pd.DataFrame([{
        "material": "ZnO",
        "dopant": "Al",
        "temp": tempK,
        "conc": conc,
        "particle_size": size
    }])

    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    predicted = model.predict(df_encoded)[0]
    theoretical = varshni(tempK)
    expected = np.mean([predicted, theoretical, 3.1, 3.05])

    return theoretical, predicted, expected

# -----------------------------
# PDF GENERATOR
# -----------------------------
def generate_pdf(df, fig):

    file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Material ML Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Table
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    elements.append(table)

    elements.append(Spacer(1, 12))

    # Graph
    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(img_path, bbox_inches='tight')

    elements.append(Image(img_path, width=450, height=300))

    doc.build(elements)
    return file_path

# -----------------------------
# WORD GENERATOR
# -----------------------------
def generate_word(df):

    file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    doc = Document()

    doc.add_heading("Material ML Analysis Report", 0)

    table = doc.add_table(rows=1, cols=len(df.columns))

    for i, col in enumerate(df.columns):
        table.rows[0].cells[i].text = col

    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    doc.save(file_path)
    return file_path

# -----------------------------
# UI
# -----------------------------
st.title("🔬 Advanced Material ML System")

file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:

    # READ
    df = pd.read_csv(file, sep=None, engine="python") if file.name.endswith(".csv") else pd.read_excel(file)

    st.subheader("📊 Raw Data")
    st.dataframe(df)

    # CLEAN + RENAME
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "temperature (°c)": "temp",
        "dopant (%)": "conc",
        "particle size (nm)": "particle_size",
        "band gap (ev)": "band_gap_exp"
    })

    # CONVERT TYPES
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
    df["particle_size"] = pd.to_numeric(df["particle_size"], errors="coerce")
    df["band_gap_exp"] = pd.to_numeric(df["band_gap_exp"], errors="coerce")

    df = df.dropna()
    df = df[df["temp"] > 0]

    # SAVE ORIGINAL °C
    df["temp (°C)"] = df["temp"]

    # CONVERT TO K
    df["temp (K)"] = df["temp"] + 273

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df)

    # RUN
    if st.button("🚀 Run Analysis"):

        results = []

        for _, row in df.iterrows():

            theo, pred, exp = run_pipeline(
                row["temp (K)"],
                row["conc"],
                row["particle_size"]
            )

            results.append({
                "Temperature (°C)": row["temp (°C)"],
                "Temperature (K)": row["temp (K)"],
                "Conc (%)": row["conc"],
                "Particle Size (nm)": row["particle_size"],
                "Experimental (eV)": row["band_gap_exp"],
                "Theoretical (eV)": theo,
                "Predicted (eV)": pred,
                "Expected (eV)": exp
            })

        df_results = pd.DataFrame(results)

        st.subheader("📋 Results Table")
        st.dataframe(df_results)

        # METRICS
        mae = mean_absolute_error(df_results["Experimental (eV)"], df_results["Predicted (eV)"])
        r2 = r2_score(df_results["Experimental (eV)"], df_results["Predicted (eV)"])

        st.metric("MAE", round(mae, 4))
        st.metric("R²", round(r2, 4))

        # GRAPH (PROPER SCALE)
        fig, ax = plt.subplots()

        ax.plot(df_results["Temperature (K)"], df_results["Experimental (eV)"], marker='o', label="Experimental")
        ax.plot(df_results["Temperature (K)"], df_results["Predicted (eV)"], marker='s', label="Predicted")
        ax.plot(df_results["Temperature (K)"], df_results["Expected (eV)"], marker='^', label="Expected")

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Band Gap (eV)")
        ax.set_title("Band Gap vs Temperature")

        ax.legend()
        ax.grid(True)

        # AUTO SCALE FIX
        ax.set_ylim(
            df_results[["Experimental (eV)", "Predicted (eV)", "Expected (eV)"]].min().min() - 0.2,
            df_results[["Experimental (eV)", "Predicted (eV)", "Expected (eV)"]].max().max() + 0.2
        )

        st.pyplot(fig)

        # ---------------- DOWNLOADS ----------------

        # CSV
        csv = df_results.to_csv(index=False).encode()
        st.download_button("⬇ Download Results CSV", csv, "results.csv")

        # GRAPH
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button("⬇ Download Graph", buf.getvalue(), "graph.png")

        # PDF
        pdf_file = generate_pdf(df_results, fig)
        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download PDF", f, "report.pdf")

        # WORD
        word_file = generate_word(df_results)
        with open(word_file, "rb") as f:
            st.download_button("📝 Download Word", f, "report.docx")

        st.success("✅ Analysis Complete")
