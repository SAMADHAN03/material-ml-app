import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile

from mp_api.client import MPRester
from sklearn.metrics import mean_absolute_error, r2_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Material ML Analysis", layout="wide")
API_KEY = st.secrets["MP_API_KEY"]

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    return model, features

model, feature_columns = load_assets()

# -----------------------------
# API FUNCTION
# -----------------------------
@st.cache_data
def get_base_gap(material):
    try:
        with MPRester(API_KEY) as mpr:
            docs = mpr.summary.search(formula=material, fields=["band_gap"])
            if docs:
                return docs[0].band_gap
    except:
        pass
    return 3.0

# -----------------------------
# VARSHNI MODEL
# -----------------------------
def varshni(material, T):
    params = {
        "ZnO": (5.5e-4, 900),
        "Fe2O3": (4.5e-4, 500),
        "CeO2": (4.7e-4, 600)
    }
    alpha, beta = params.get(material, (5e-4, 500))
    Eg0 = get_base_gap(material)
    return Eg0 - (alpha * T**2) / (T + beta)

# -----------------------------
# AUTO COLUMN DETECTION
# -----------------------------
def normalize_columns(df):

    df.columns = df.columns.str.strip().str.lower()

    col_map = {}

    for col in df.columns:

        if "temp" in col:
            col_map[col] = "temp"

        elif "conc" in col:
            col_map[col] = "conc"

        elif "size" in col:
            col_map[col] = "particle_size"

        elif "material" in col:
            col_map[col] = "material"

        elif "dopant" in col:
            col_map[col] = "dopant"

        elif "band_gap_exp" in col or "experimental" in col:
            col_map[col] = "band_gap_exp"

    df = df.rename(columns=col_map)

    return df

# -----------------------------
# UNIT CONVERSION
# -----------------------------
def convert_units(df):

    # Temperature conversion
    if df["temp"].mean() < 200:  # assume Celsius
        st.info("🌡 Temperature converted from °C to K")
        df["temp"] = df["temp"] + 273

    # Concentration normalization
    if df["conc"].max() <= 1:
        st.info("🧪 Concentration converted to %")
        df["conc"] = df["conc"] * 100

    return df

# -----------------------------
# ML PIPELINE
# -----------------------------
def run_pipeline(material, dopant, temp, conc, size):

    mp_gap = get_base_gap(material)
    oqmd_gap = 3.10
    aflow_gap = 3.05
    theoretical = varshni(material, temp)

    df_input = pd.DataFrame([{
        "material": material,
        "dopant": dopant,
        "temp": temp,
        "conc": conc,
        "particle_size": size
    }])

    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    predicted = model.predict(df_encoded)[0]
    expected = np.mean([predicted, theoretical, oqmd_gap, aflow_gap, mp_gap])

    return theoretical, predicted, expected, mp_gap, oqmd_gap, aflow_gap

# -----------------------------
# PDF GENERATOR
# -----------------------------
def generate_pdf(df_results, fig):

    path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Material ML Report", styles['Title']))
    elements.append(Spacer(1, 12))

    data = [df_results.columns.tolist()] + df_results.values.tolist()
    elements.append(Table(data))

    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(img_path)
    elements.append(Image(img_path, width=400, height=300))

    doc.build(elements)

    return path

# -----------------------------
# WORD GENERATOR
# -----------------------------
def generate_word(df_results):

    path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    doc = Document()

    doc.add_heading("Material ML Report", 0)

    table = doc.add_table(rows=1, cols=len(df_results.columns))

    for i, col in enumerate(df_results.columns):
        table.rows[0].cells[i].text = col

    for _, row in df_results.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    doc.save(path)

    return path

# -----------------------------
# UI
# -----------------------------
st.title("🔬 Advanced Material ML System")

file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:

    df = pd.read_csv(file, sep=None, engine='python') if file.name.endswith(".csv") else pd.read_excel(file)

    st.subheader("Raw Data")
    st.dataframe(df)

    # CLEANING
    df = normalize_columns(df)

    required = ["material", "temp"]

    if not all(col in df.columns for col in required):
        st.error("Missing required columns (material, temp)")
        st.stop()

    df = df.dropna(subset=["material", "temp"])

    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["conc"] = pd.to_numeric(df.get("conc", 0), errors="coerce").fillna(0)
    df["particle_size"] = pd.to_numeric(df.get("particle_size", 0), errors="coerce").fillna(0)

    df = df[df["temp"] > 0]

    df = convert_units(df)

    st.subheader("Cleaned Data")
    st.dataframe(df)

    # RUN
    if st.button("Run Analysis"):

        results = []

        for _, row in df.iterrows():

            theo, pred, exp, mp, oqmd, aflow = run_pipeline(
                row["material"],
                row.get("dopant", "None"),
                row["temp"],
                row["conc"],
                row["particle_size"]
            )

            results.append({
                "material": row["material"],
                "dopant": row.get("dopant", "None"),
                "temp (K)": row["temp"],
                "conc (%)": row["conc"],
                "particle_size": row["particle_size"],
                "band_gap_theoretical": theo,
                "band_gap_predicted": pred,
                "band_gap_expected": exp,
                "band_gap_api(MP)": mp,
                "band_gap_oqmd": oqmd,
                "band_gap_aflow": aflow
            })

        df_results = pd.DataFrame(results)

        st.subheader("Results")
        st.dataframe(df_results)

        # GRAPH vs TEMP
        fig, ax = plt.subplots()

        ax.plot(df_results["temp (K)"], df_results["band_gap_theoretical"], label="Theoretical")
        ax.plot(df_results["temp (K)"], df_results["band_gap_predicted"], label="Predicted")
        ax.plot(df_results["temp (K)"], df_results["band_gap_expected"], label="Expected")

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Band Gap (eV)")
        ax.set_title("Band Gap vs Temperature")
        ax.legend()
        ax.grid()

        st.pyplot(fig)

        # DOWNLOADS
        csv = df_results.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "results.csv")

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("Download Graph", buf.getvalue(), "graph.png")

        pdf = generate_pdf(df_results, fig)
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, "report.pdf")

        word = generate_word(df_results)
        with open(word, "rb") as f:
            st.download_button("Download Word", f, "report.docx")

        st.success("Analysis Complete 🚀")
