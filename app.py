import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Parkinson's Disease Predictor", layout="centered")

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Parkinson's Disease Prediction App")

st.markdown("""
This app predicts that the person have **Parkinson's Disease** based on vocal measurement features.
Please enter the following values carefully.
""")
feature_info = {
    'MDVP:Fo(Hz)': 'Average vocal frequency (e.g., 120 to 180 Hz)',
    'MDVP:Flo(Hz)': 'Minimum vocal frequency (e.g., 70 to 130 Hz)',
    'MDVP:Jitter(%)': 'Frequency variation (e.g., 0.001-0.01)',
    'MDVP:Jitter(Abs)': 'Absolute jitter (e.g., 0.00001-0.0001)',
    'Jitter:DDP': 'Difference of differences of periods (e.g., 0.00001-0.002)',
    'MDVP:APQ': 'Amplitude perturbation quotient (e.g., 0.001-0.02)',
    'NHR': 'Noise-to-harmonics ratio (e.g., 0.01-0.2)',
    'spread1': 'Nonlinear signal measure (e.g., -7 to 0)',
    'spread2': 'Nonlinear signal measure (e.g., 0 to 3)',
    'PPE': 'Pitch period entropy (e.g., 0.1-1.0)'
}
user_input = []

features = list(feature_info.keys())
for i in range(0, len(features), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(features):
            feature = features[i + j]
            with cols[j]:
                value = st.text_input(
                    f"{feature}",
                    placeholder=f"({feature_info[feature]})",
                    help=feature_info[feature]
                )
                user_input.append(value)
if st.button("🧪 Predict"):
    try:
        input_array = np.array(user_input, dtype=float).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        st.subheader("🩺 Prediction Result")
        if prediction[0] == 1:
            st.error("⚠️ The person have Parkinson's Disease.")
        else:
            st.success("✅ The person does not have Parkinson Disease.")
    except ValueError:
        st.warning("❗ Please enter valid numeric values for all features.")