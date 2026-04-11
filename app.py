import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. LOAD ASSETS ---
model = joblib.load('hybrid_bandgap_model.pkl')
features = joblib.load('model_features.pkl')

st.title("🔬 Advanced Material ML Predictor")

# --- 2. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    # --- 3. AUTOMATIC CLEANING & MAPPING ---
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    mapping = {'conc': 'concentration', 'temp': 'temperature', 'psize': 'particle_size'}
    df = df.rename(columns=mapping)

    # Force relevant columns to be numeric and remove "None" or unrelated text
    cols_to_fix = [c for c in ['temperature', 'concentration', 'particle_size'] if c in df.columns]
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=cols_to_fix).reset_index(drop=True)

    # --- 4. HYBRID LOGIC & FEATURE ALIGNMENT ---
    # Generate Varshni Theoretical Gap
    if 'temperature' in df.columns:
        alpha, beta = 5.5e-4, 900
        df['theoretical_gap'] = 3.44 - (alpha * df['temperature']**2) / (df['temperature'] + beta)

    # Align with ML Model Features (Handles the KeyError)
    df_encoded = pd.get_dummies(df)
    X_input = df_encoded.reindex(columns=features, fill_value=0)

    # --- 5. RESULTS & OUTPUT ---
    if not X_input.empty:
        df['predicted_gap'] = model.predict(X_input)
        
        # Calculate Final Expected Hybrid Result
        # Using typical fallback values for database gaps if not provided in Excel
        mp_gap = df['band_gap_mp'] if 'band_gap_mp' in df.columns else 3.20
        df['expected_gap'] = (0.4 * df['predicted_gap'] + 0.4 * df['theoretical_gap'] + 0.2 * mp_gap)

        st.success(f"✅ Processed {len(df)} valid rows.")
        st.write("### Final Results Table")
        st.dataframe(df)

        # --- 6. VISUALIZATION ---
        st.write("### Temperature vs Band Gap")
        fig, ax = plt.subplots()
        df_sorted = df.sort_values('temperature')
        ax.plot(df_sorted['temperature'], df_sorted['theoretical_gap'], 'g--', label='Theory')
        ax.plot(df_sorted['temperature'], df_sorted['predicted_gap'], 'b-o', label='ML Predicted')
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Band Gap (eV)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("❌ No valid data found. Check your column headers (temp, conc, psize).")
