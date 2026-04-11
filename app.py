import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# --- LOAD ASSETS ---
model = joblib.load('hybrid_bandgap_model.pkl')
features = joblib.load('model_features.pkl')

st.title("🔬 Advanced Material ML Predictor")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # 1. Load Data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    # 2. THE FIX: REMOVE NONE/MISSING DATA
    # Drop rows where critical features are missing
    initial_count = len(df)
    df = df.dropna(subset=features).reset_index(drop=True)
    
    removed_count = initial_count - len(df)
    if removed_count > 0:
        st.warning(f"⚠️ Automatically removed {removed_count} rows containing missing (None/NaN) values.")

    # 3. PREDICTION
    if not df.empty and all(col in df.columns for col in features):
        preds = model.predict(df[features])
        df['Predicted_Gap'] = preds
        
        st.write("### ✅ Prediction Results", df)

        # 4. ACCURACY & GRAPHS
        # Only if the file has ground truth data
        if 'band_gap_exp' in df.columns:
            # Clean experimental column too just in case
            valid_idx = df['band_gap_exp'].notnull()
            clean_exp = df.loc[valid_idx, 'band_gap_exp']
            clean_pred = df.loc[valid_idx, 'Predicted_Gap']

            mae = mean_absolute_error(clean_exp, clean_pred)
            r2 = r2_score(clean_exp, clean_pred)
            
            st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
            st.metric("R² Score", f"{r2:.4f}")

            # Plotting
            fig, ax = plt.subplots()
            ax.scatter(clean_exp, clean_pred, color='blue', alpha=0.6, label="Data Points")
            ax.plot([clean_exp.min(), clean_exp.max()], [clean_exp.min(), clean_exp.max()], 'r--', label="Perfect Match")
            ax.set_xlabel("Experimental Value (eV)")
            ax.set_ylabel("Predicted Value (eV)")
            ax.legend()
            st.pyplot(fig)
    else:
        st.error(f"❌ Error: Your file is either empty after cleaning or missing columns: {features}")
