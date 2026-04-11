import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. LOAD ASSETS ---
# Using the filenames verified in your GitHub repository
model = joblib.load('hybrid_bandgap_model.pkl')
features = joblib.load('model_features.pkl')

st.set_page_config(page_title="Material ML Predictor", layout="wide")
st.title("🔬 Advanced Material Band Gap Predictor")
st.write("Upload your experimental data to calculate ML-Predicted, Theoretical, and Hybrid Expected values.")

# --- 2. VARSHNI PHYSICS ENGINE ---
def calculate_varshni(row):
    """Calculates theoretical shift: Eg(T) = Eg(0) - (alpha*T^2)/(T + beta)"""
    # Material-specific constants
    params = {
        "ZnO": {"eg0": 3.44, "alpha": 5.5e-4, "beta": 900},
        "Fe2O3": {"eg0": 2.20, "alpha": 4.5e-4, "beta": 500},
        "CeO2": {"eg0": 3.20, "alpha": 4.7e-4, "beta": 600}
    }
    
    mat = str(row.get('material', 'ZnO'))
    T = row.get('temp', 300)
    
    p = params.get(mat, params["ZnO"]) # Default to ZnO if unknown
    return p['eg0'] - (p['alpha'] * T**2) / (T + p['beta'])

# --- 3. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # --- 4. AUTOMATIC CLEANING & MAPPING ---
    # Standardize column names (fixes case sensitivity and spaces)
    mapping = {
        'temperature': 'temp', 'temp_k': 'temp', 't': 'temp',
        'concentration': 'conc', 'c': 'conc',
        'size': 'particle_size', 'nm': 'particle_size',
        'experimental': 'band_gap_exp', 'actual': 'band_gap_exp'
    }
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=mapping)

    # Ensure required features exist for ML model
    for col in features:
        if col not in df.columns:
            df[col] = 0.0 # Default value to prevent crash

    # --- 5. HYBRID CALCULATIONS ---
    # A. ML Prediction
    df['ML_Predicted'] = model.predict(df[features])

    # B. Varshni Theoretical Prediction
    df['Theoretical_Gap'] = df.apply(calculate_varshni, axis=1)

    # C. Hybrid Expected (Combining ML + Physics)
    # Using the weighted logic that gave you the best results in Colab
    df['Hybrid_Expected'] = (0.6 * df['ML_Predicted']) + (0.4 * df['Theoretical_Gap'])

    # --- 6. RESULTS & METRICS ---
    st.subheader("📊 Processed Results")
    st.dataframe(df)

    if 'band_gap_exp' in df.columns:
        mae = mean_absolute_error(df['band_gap_exp'], df['Hybrid_Expected'])
        r2 = r2_score(df['band_gap_exp'], df['Hybrid_Expected'])
        
        c1, c2 = st.columns(2)
        c1.metric("Hybrid System MAE", f"{mae:.4f} eV")
        c2.metric("R² Accuracy Score", f"{r2:.4f}")

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df['band_gap_exp'], df['Hybrid_Expected'], color='blue', label='Predictions')
        ax.plot([df['band_gap_exp'].min(), df['band_gap_exp'].max()], 
                [df['band_gap_exp'].min(), df['band_gap_exp'].max()], 'r--', label='Perfect Match')
        ax.set_xlabel("Experimental (Actual)")
        ax.set_ylabel("Hybrid Expected")
        ax.legend()
        st.pyplot(fig)

    # --- 7. EXPORT ---
    st.download_button(
        label="📥 Download Results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='material_predictions.csv',
        mime='text/csv',
    )
