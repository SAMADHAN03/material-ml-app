import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
st.set_page_config(page_title="Advanced Material ML System", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_assets():
    return (
        joblib.load("model.pkl"),
        joblib.load("features.pkl"),
        joblib.load("scaler.pkl")
    )

model, feature_columns, scaler = load_assets()

# -----------------------------
# INTELLIGENT READER
# -----------------------------
def intelligent_reader(file):

    # FILE READ
    try:
        name = file.name.lower()

        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)

        else:
            try:
                df = pd.read_csv(file, sep=None, engine='python')
            except:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1', sep=None, engine='python')
                except:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='cp1252', sep=None, engine='python')

    except Exception as e:
        st.error(f"File read error: {e}")
        return None

    # CLEAN COLUMN NAMES
    df.columns = df.columns.str.strip().str.lower()

    # COLUMN DETECTION
    mapping = {}
    for col in df.columns:

        if "material" in col:
            mapping[col] = "material"

        elif "dopant" in col:
            mapping[col] = "dopant"

        elif "%" in col or "conc" in col:
            mapping[col] = "conc"

        elif "temp" in col:
            mapping[col] = "temp"

        elif "particle" in col:
            mapping[col] = "particle_size"

        elif "band" in col:
            mapping[col] = "band_gap_exp"

        elif "ref" in col:
            mapping[col] = "reference"

    df = df.rename(columns=mapping)

    # TYPE CONVERSION
    for col in ["temp", "conc", "particle_size", "band_gap_exp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["temp", "band_gap_exp"])

    # REMOVE INVALID
    df = df[df["temp"] > 0]

    # TEMP CONVERSION
    df["temp_c"] = df["temp"]
    df["temp_k"] = df["temp"] + 273

    # FILL MISSING
    df["material"] = df.get("material", "ZnO")
    df["dopant"] = df.get("dopant", "None")
    df["conc"] = df.get("conc", 0)
    df["particle_size"] = df.get("particle_size", 1)
    df["reference"] = df.get("reference", "N/A")

    return df.reset_index(drop=True)

# -----------------------------
# VARSHNI MODEL
# -----------------------------
def varshni(T):
    Eg0, alpha, beta = 3.44, 5.5e-4, 900
    return Eg0 - (alpha*T**2)/(T+beta)

# -----------------------------
# ML PIPELINE
# -----------------------------
def predict(material, dopant, tempK, conc, size):

    df_input = pd.DataFrame([{
        "material": material,
        "dopant": dopant,
        "temp": tempK,
        "conc": conc,
        "particle_size": size,
        "inv_temp": 1/tempK,
        "log_size": np.log(size),
        "conc_sq": conc**2
    }])

    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    df_scaled = scaler.transform(df_encoded)

    pred = model.predict(df_scaled)[0]
    theo = varshni(tempK)
    expected = np.mean([pred, theo, 3.1, 3.05])

    return pred, theo, expected

# -----------------------------
# PDF
# -----------------------------
def generate_pdf(df, fig):
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Material ML Report", styles['Title']))
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
# WORD
# -----------------------------
def generate_word(df):
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    doc = Document()

    doc.add_heading("Material ML Report", 0)

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
st.title("🔬 Advanced Material ML System (Ultimate)")

uploaded_file = st.file_uploader("Upload CSV / Excel")

if uploaded_file:

    df = intelligent_reader(uploaded_file)

    st.subheader("📊 Cleaned Data")
    st.dataframe(df)

    if st.button("🚀 Run Analysis"):

        results = []

        for _, row in df.iterrows():

            pred, theo, exp = predict(
                row["material"],
                row["dopant"],
                row["temp_k"],
                row["conc"],
                row["particle_size"]
            )

            results.append({
                "Material": row["material"],
                "Dopant": row["dopant"],
                "Temp (°C)": row["temp_c"],
                "Temp (K)": row["temp_k"],
                "Conc (%)": row["conc"],
                "Particle Size (nm)": row["particle_size"],
                "Experimental": row["band_gap_exp"],
                "Predicted": pred,
                "Theoretical": theo,
                "Expected": exp,
                "Reference": row["reference"]
            })

        df_results = pd.DataFrame(results)

        st.subheader("📋 Results")
        st.dataframe(df_results)

        # METRICS
        mae = mean_absolute_error(df_results["Experimental"], df_results["Predicted"])
        r2 = r2_score(df_results["Experimental"], df_results["Predicted"])

        col1, col2 = st.columns(2)
        col1.metric("MAE", round(mae, 4))
        col2.metric("R²", round(r2, 4))

        # GRAPH
        fig, ax = plt.subplots()

        ax.plot(df_results["Temp (K)"], df_results["Experimental"], marker='o', label="Experimental")
        ax.plot(df_results["Temp (K)"], df_results["Predicted"], marker='s', label="Predicted")
        ax.plot(df_results["Temp (K)"], df_results["Expected"], marker='^', label="Expected")

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Band Gap (eV)")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # DOWNLOADS
        csv = df_results.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv, "results.csv")

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("⬇ Download Graph", buf.getvalue(), "graph.png")

        pdf = generate_pdf(df_results, fig)
        with open(pdf, "rb") as f:
            st.download_button("📄 Download PDF", f, "report.pdf")

        word = generate_word(df_results)
        with open(word, "rb") as f:
            st.download_button("📝 Download Word", f, "report.docx")

        st.success("✅ Complete System Executed Successfully 🚀")
        
