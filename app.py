from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import joblib
st.set_page_config(
    page_title="Parkinson's Disease Predictor",
    layout="centered"
)
BASE_DIR = Path(__file__).resolve().parent
@st.cache_resource
def load_model_files():
    model = joblib.load(BASE_DIR / "model.joblib")
    imputer = joblib.load(BASE_DIR / "imputer.joblib")
    scaler = joblib.load(BASE_DIR / "scaler.joblib")
    selector = joblib.load(BASE_DIR / "selector.joblib")
    feature_columns = joblib.load(
        BASE_DIR / "feature_columns.joblib"
    )
    return model, imputer, scaler, selector, feature_columns
try:
    model, imputer, scaler, selector, feature_columns = (
        load_model_files()
    )
except Exception as e:
    st.error(
        "Could not load the model files. Check that all five "
        "files exist in the application folder and are compatible."
    )
    st.exception(e)
    st.stop()
st.title("Parkinson's Disease Prediction App")
st.markdown("""
Enter the biomedical voice measurements below.
This application is an educational machine-learning project,
not a substitute for professional medical diagnosis.
""")
feature_info = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency in Hz.",
    "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency in Hz.",
    "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency in Hz.",
    "MDVP:Jitter(%)": "Variation in fundamental frequency.",
    "MDVP:Jitter(Abs)": "Absolute jitter measurement.",
    "MDVP:RAP": "Relative average perturbation.",
    "MDVP:PPQ": "Five-point period perturbation quotient.",
    "Jitter:DDP": "Difference of differences of periods.",
    "MDVP:Shimmer": "Variation in voice amplitude.",
    "MDVP:Shimmer(dB)": "Shimmer measured in decibels.",
    "Shimmer:APQ3": "Three-point amplitude perturbation quotient.",
    "Shimmer:APQ5": "Five-point amplitude perturbation quotient.",
    "MDVP:APQ": "Amplitude perturbation quotient.",
    "Shimmer:DDA": "Average absolute difference of consecutive amplitudes.",
    "NHR": "Noise-to-harmonics ratio.",
    "HNR": "Harmonics-to-noise ratio.",
    "RPDE": "Recurrence period density entropy.",
    "DFA": "Detrended fluctuation analysis.",
    "spread1": "Nonlinear fundamental-frequency measure.",
    "spread2": "Nonlinear fundamental-frequency measure.",
    "D2": "Correlation dimension measure.",
    "PPE": "Pitch period entropy."
}
st.info(
    f"The model expects {len(feature_columns)} original "
    "measurement features."
)
input_data = {}
with st.form("parkinsons_prediction_form"):
    st.subheader("Enter Voice Measurements")
    for start in range(0, len(feature_columns), 2):
        cols = st.columns(2)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx >= len(feature_columns):
                break
            feature = feature_columns[idx]
            with col:
                input_data[feature] = st.text_input(
                    feature,
                    placeholder="Enter numeric value",
                    help=feature_info.get(
                        feature,
                        f"Enter the measured value for {feature}."
                    ),
                    key=f"input_{idx}"
                )
    submitted = st.form_submit_button("🧪 Predict")
if submitted:
    missing_inputs = [
        feature
        for feature in feature_columns
        if not input_data.get(feature, "").strip()
    ]
    if missing_inputs:
        st.warning(
            "Please enter values for all required features."
        )
    else:
        try:
            values = {
                feature: float(input_data[feature])
                for feature in feature_columns
            }
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError(
                    "Please enter finite numeric values only."
                )
            # Preserve training feature names and order
            input_df = pd.DataFrame(
                [values],
                columns=feature_columns
            )
            # Match the preprocessing used during training
            X_imputed = imputer.transform(input_df)
            X_scaled = scaler.transform(X_imputed)
            X_selected = selector.transform(X_scaled)
            # Predict with the new SVM
            prediction = int(model.predict(X_selected)[0])
            st.subheader("🩺 Prediction Result")
            if prediction == 1:
                st.error(
                    "⚠️ The person have Parkinson's Disease"
                )
            else:
                st.success(
                    "✅ The person does not havw Parkinson's Disease"
                )
        except ValueError as e:
            st.warning(f"Please check your input values: {e}")
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
