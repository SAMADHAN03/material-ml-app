import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Load your model assets
model = joblib.load('hybrid_bandgap_model.pkl')
features = joblib.load('model_features.pkl')

st.title("🔬 Hybrid Material Predictor")

# FIXED LINE:
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Handle CSV or Excel
    df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    # Check if required columns exist
    if all(col in df_input.columns for col in features):
        preds = model.predict(df_input[features])
        df_input['Predicted_Gap'] = preds
        
        st.write("### Prediction Results", df_input)

        # Calculate Accuracy if Experimental data is present
        if 'band_gap_exp' in df_input.columns:
            mae = mean_absolute_error(df_input['band_gap_exp'], preds)
            r2 = r2_score(df_input['band_gap_exp'], preds)
            
            st.metric("MAE", f"{mae:.4f}")
            st.metric("R² Score", f"{r2:.4f}")

            # Plotting
            fig, ax = plt.subplots()
            ax.scatter(df_input['band_gap_exp'], preds, color='blue', alpha=0.5)
            ax.plot([df_input['band_gap_exp'].min(), df_input['band_gap_exp'].max()],
                    [df_input['band_gap_exp'].min(), df_input['band_gap_exp'].max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)
