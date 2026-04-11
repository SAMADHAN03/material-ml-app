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
st.write("Upload your experimental data to calculate ML-Predicted, Theoretical, and Hybrid Expected Band Gaps.")

# --- 2. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # --- 3. AUTOMATIC CLEANING & MAPPING ---
    # Standardize column names (lowercase, no spaces)
    df.columns = df.columns.str.strip().str.lower()
    
    # Expanded mapping to catch common Excel naming variations
    mapping = {
        'conc': 'concentration', 'concentration (%)': 'concentration',
        'temp': 'temperature', 'temperature (k)': 'temperature',
        'psize': 'particle_size', 'size (nm)': 'particle_size'
    }
    df = df.rename(columns=mapping)

    # Initialize required columns with NaN to prevent KeyError
    df['theoretical_gap'] = np.nan
    df['predicted_gap'] = np.nan
    df['expected_gap'] = np.nan

    # Force relevant columns to be numeric; non-numeric values become NaN
    cols_to_fix = [c for c in ['temperature', 'concentration', 'particle_size'] if c in df.columns]
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows where essential data is missing
    df = df.dropna(subset=cols_to_fix).reset_index(drop=True)

    # --- 4. CALCULATIONS ---
    # Varshni Theoretical Gap Calculation
    if 'temperature' in df.columns:
        alpha, beta = 5.5e-4, 900
        # Formula: Eg(T) = Eg(0) - (alpha * T^2) / (T + beta)
        # Using 3.44 eV (e.g., for GaN/ZnO type materials) as base
        df['theoretical_gap'] = 3.44 - (alpha * df['temperature']**2) / (df['temperature'] + beta)

    # Align with ML Model Features
    # get_dummies handles 'material' column if present
    df_encoded = pd.get_dummies(df)
    X_input = df_encoded.reindex(columns=features, fill_value=0)

    if not X_input.empty:
        # ML Prediction
        df['predicted_gap'] = model.predict(X_input)
        
        # Determine fallback for Database values (Materials Project)
        mp_gap = df['band_gap_mp'] if 'band_gap_mp' in df.columns else 3.20
        
        # HYBRID CALCULATION (Weighted Average)
        # We use .fillna(0) here as a safety net for the math
        df['expected_gap'] = (
            (0.4 * df['predicted_gap'].fillna(0)) + 
            (0.4 * df['theoretical_gap'].fillna(df['predicted_gap'])) + 
            (0.2 * mp_gap)
        )

        # --- 5. DISPLAY RESULTS ---
        st.success(f"✅ Successfully processed {len(df)} rows of data.")
        
        # Display key columns
        display_cols = ['temperature', 'concentration', 'theoretical_gap', 'predicted_gap', 'expected_gap']
        st.subheader("📊 Result Summary")
        st.dataframe(df[[c for c in display_cols if c in df.columns]].style.highlight_max(axis=0))

        # --- 6. VISUALIZATION ---
        if 'temperature' in df.columns and len(df) > 1:
            st.subheader("📈 Temperature vs. Band Gap")
            df_sorted = df.sort_values('temperature')
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_sorted['temperature'], df_sorted['theoretical_gap'], 'g--', label='Theoretical (Varshni)')
            ax.plot(df_sorted['temperature'], df_sorted['predicted_gap'], 'b-o', label='ML Predicted')
            ax.plot(df_sorted['temperature'], df_sorted['expected_gap'], 'r-s', label='Hybrid Expected', linewidth=2)
            
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Band Gap (eV)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        # Download Option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "predicted_materials.csv", "text/csv")
