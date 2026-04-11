import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load your trained assets from GitHub
model = joblib.load('hybrid_bandgap_model.pkl')
features = joblib.load('model_features.pkl')

st.title("🔬 Material Property Predictor & Analyzer")

# 2. File Uploader
uploaded_file = st.file_file_uploader("Upload your Excel or CSV file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)

    st.write("### Data Preview", df_input.head())

    # 3. Prediction Logic
    # Ensure your uploaded file has the same column names as your 'features' list
    if all(col in df_input.columns for col in features):
        X_input = df_input[features]
        predictions = model.predict(X_input)
        df_input['Predicted_Band_Gap'] = predictions

        st.write("### Prediction Results", df_input)

        # 4. Metrics & Graphs (Only if 'band_gap_exp' exists in the file)
        if 'band_gap_exp' in df_input.columns:
            mae = mean_absolute_error(df_input['band_gap_exp'], predictions)
            r2 = r2_score(df_input['band_gap_exp'], predictions)

            col1, col2 = st.columns(2)
            col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
            col2.metric("R² Score", f"{r2:.4f}")

            # 5. Visualization
            fig, ax = plt.subplots()
            ax.scatter(df_input['band_gap_exp'], predictions, alpha=0.5)
            ax.plot([df_input['band_gap_exp'].min(), df_input['band_gap_exp'].max()],
                    [df_input['band_gap_exp'].min(), df_input['band_gap_exp'].max()], 'r--')
            ax.set_xlabel("Experimental Value")
            ax.set_ylabel("Predicted Value")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
        else:
            st.warning("Upload a file with a 'band_gap_exp' column to see MAE and R².")
    else:
        st.error(f"File must contain these columns: {features}")
