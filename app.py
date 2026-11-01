from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

from pymongo import MongoClient, DESCENDING


# =====================================================
# 1. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Parkinson's Disease Predictor",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>

/* Feature names */
[data-testid="stWidgetLabel"] p {
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Input text */
.stTextInput input {
    font-size: 17px !important;
    height: 48px !important;
}

/* Placeholder text */
.stTextInput input::placeholder {
    font-size: 16px !important;
}

/* Predict button text */
.stFormSubmitButton button,
.stFormSubmitButton button p {
    font-size: 20px !important;
    font-weight: 700 !important;
}

.stFormSubmitButton button {
    min-height: 58px !important;
}

</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

st.title("Parkinson's Disease Prediction App", anchor=False)


# =====================================================
# 2. LOAD EXISTING ML MODEL FILES
# =====================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(BASE_DIR / "model.joblib")

    imputer = joblib.load(BASE_DIR / "imputer.joblib")

    scaler = joblib.load(BASE_DIR / "scaler.joblib")

    feature_columns = joblib.load(
        BASE_DIR / "feature_columns.joblib"
    )

    return model, imputer, scaler, feature_columns


try:

    model, imputer, scaler, feature_columns = (
        load_model_files()
    )

except Exception as e:

    st.error(
        "Could not load the model files. "
        "Check that all four required files exist "
        "and are compatible with the deployment environment."
    )

    st.exception(e)
    st.stop()


# =====================================================
# 3. MONGODB ATLAS CONNECTION
# =====================================================

@st.cache_resource
def get_mongodb_database():

    uri = st.secrets.get("MONGODB_URI", "")

    if not uri:
        return None

    database_name = st.secrets.get(
        "MONGODB_DATABASE",
        "ParkinsonDB"
    )

    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000
    )

    # Verify the connection
    client.admin.command("ping")

    return client[database_name]


database = None
database_error = None

try:

    database = get_mongodb_database()

except Exception as e:

    database_error = e


deployment_collection = None
metrics_collection = None

if database is not None:

    deployment_collection = database[
        "deployment_predictions"
    ]

    metrics_collection = database[
        "model_metrics"
    ]


# =====================================================
# 4. MODEL PERFORMANCE AND INPUT FORM
# =====================================================

left_panel, right_panel = st.columns(
    [0.5, 2.5],
    gap="large"
)


# =====================================================
# LEFT SIDE: MODEL INFORMATION
# =====================================================

with left_panel:

    st.subheader("Model Performance", anchor=False)

    with st.container(border=True):

        st.metric(
            label="Model Accuracy",
            value="91.67%"
        )

    with st.container(border=True):

        st.metric(
            label="Test Samples Used",
            value="1,992"
        )

    with st.container(border=True):

        st.metric(
            label="Model Used",
            value="SVM"
        )


# =====================================================
# RIGHT SIDE: INPUT FEATURES
# =====================================================

with right_panel:

    feature_info = {

        "MDVP:Jitter(Abs)": "Absolute jitter measurement.",
        "MDVP:Shimmer": "Variation in voice amplitude.",
        "MDVP:Shimmer(dB)": "Shimmer measured in decibels.",
        "Shimmer:APQ3": "Three-point amplitude perturbation quotient.",
        "Shimmer:APQ5": "Five-point amplitude perturbation quotient.",
        "MDVP:APQ": "Amplitude perturbation quotient.",
        "Shimmer:DDA": "Average absolute amplitude difference.",
        "spread1": "Nonlinear fundamental-frequency measure.",
        "spread2": "Nonlinear fundamental-frequency measure.",
        "PPE": "Pitch period entropy."
    }

    input_data = {}

    with st.form("parkinsons_prediction_form"):

        st.subheader("Enter Voice Measurements", anchor=False)

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
                        autocomplete="on",
                        key=f"input_{idx}"
                    )

        submitted = st.form_submit_button(
            "🧪 Predict",
            use_container_width=True
        )


# =====================================================
# 7. GENERATE PREDICTION AND SAVE TO ATLAS
# =====================================================

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

            # Convert input strings into numerical values
            values = {

                feature: float(input_data[feature])

                for feature in feature_columns

            }

            # Validate input values
            if not all(np.isfinite(v) for v in values.values()):

                raise ValueError(
                    "Please enter finite numeric values only."
                )

            # Preserve training feature names and order
            input_df = pd.DataFrame(
                [values],
                columns=feature_columns
            )

            # Existing preprocessing pipeline
            X_imputed = imputer.transform(input_df)

            X_scaled = scaler.transform(X_imputed)

            # Existing trained SVM model
            prediction = int(
                model.predict(X_scaled)[0]
            )

            # Prediction result
            st.divider()
            st.subheader("🩺 Prediction Result")

            if prediction == 1:

                prediction_label = "Parkinson's"

                st.error(
                    "⚠️ The model predicted Parkinson's Disease."
                )

            else:

                prediction_label = "Non-Parkinson's"

                st.success(
                    "✅ The model predicted Non-Parkinson's."
                )


            # ---------------------------------------------
            # Save deployment prediction in MongoDB Atlas
            # ---------------------------------------------

            if deployment_collection is not None:

                prediction_document = {

                    "model": "SVM",

                    "input_features": values,

                    "predicted_status": prediction,

                    "prediction_label": prediction_label,

                    "created_at": datetime.now(timezone.utc)

                }

                try:

                    result = deployment_collection.insert_one(
                        prediction_document
                    )

                    st.success(
                        "Prediction successfully saved "
                        "to MongoDB Atlas."
                    )

                except Exception:

                    st.warning(
                        "Prediction was generated, but could "
                        "not be saved to MongoDB Atlas."
                    )

            else:

                st.warning(
                    "MongoDB Atlas is not connected. "
                    "This prediction has not been saved."
                )


        except ValueError as e:

            st.warning(
                f"Please check your input values: {e}"
            )

        except Exception as e:

            st.error("Prediction failed.")

            st.exception(e)


# =====================================================
# 8. PREDICTION ANALYTICS DASHBOARD
# =====================================================

st.divider()

st.header("📊 Prediction Analytics Dashboard", anchor=False)

if deployment_collection is None:

    if database_error is not None:

        st.error(
            "Could not connect to MongoDB Atlas. "
            "Check your Streamlit secrets and Atlas settings."
        )

    else:

        st.warning(
            "MongoDB Atlas is not configured. "
            "Analytics will become available after connecting it."
        )

else:

    try:

        # Retrieve actual deployment predictions
        records = list(

            deployment_collection.find(
                {},
                {"_id": 0}
            ).sort(
                "created_at",
                DESCENDING
            )

        )

        if not records:

            st.info(
                "No deployment predictions are available yet. "
                "Make a prediction to populate the dashboard."
            )

        else:

            # Convert MongoDB documents into a DataFrame
            analytics_df = pd.DataFrame(records)

            analytics_df["created_at"] = pd.to_datetime(
                analytics_df["created_at"],
                utc=True
            )

            # ---------------------------------------------
            # Total predictions
            # ---------------------------------------------

            st.metric(
                "Total Deployment Predictions",
                len(analytics_df)
            )

            # ---------------------------------------------
            # Prediction trends and distribution side by side
            # ---------------------------------------------

            left_col, right_col = st.columns(2, gap="large")

            with left_col:

                st.subheader("Prediction Trends Over Time", anchor=False)

                analytics_df["date"] = (
                    analytics_df["created_at"].dt.date
                )

                trend_data = (
                    analytics_df.groupby(
                        ["date", "prediction_label"]
                    )
                    .size()
                    .reset_index(name="prediction_count")
                )

                fig_trends = px.line(
                    trend_data,
                    x="date",
                    y="prediction_count",
                    color="prediction_label",
                    markers=True,
                    title="Daily Prediction Trends"
                )

                fig_trends.update_layout(
                    xaxis_title="Prediction Date",
                    yaxis_title="Number of Predictions",
                    legend_title="Prediction Result",
                    height=400
                )

                st.plotly_chart(
                    fig_trends,
                    use_container_width=True
                )


            with right_col:

                st.subheader("Prediction Distribution", anchor=False)

                distribution = (
                    analytics_df["prediction_label"]
                    .value_counts()
                    .reset_index()
                )

                distribution.columns = [
                    "Prediction",
                    "Count"
                ]

                fig_distribution = px.pie(
                    distribution,
                    names="Prediction",
                    values="Count",
                    hole=0.4,
                    title="Parkinson's vs Non-Parkinson's Predictions"
                )

                fig_distribution.update_layout(
                    height=400,
                    legend_title="Prediction Result"
                )

                st.plotly_chart(
                    fig_distribution,
                    use_container_width=True
                )


            # ---------------------------------------------
            # Prediction history
            # ---------------------------------------------

            st.subheader("Prediction History", anchor=False)

            history_df = pd.json_normalize(
                records,
                sep="."
            )

            # Format timestamps for display
            if "created_at" in history_df.columns:

                history_df["created_at"] = (

                    pd.to_datetime(
                        history_df["created_at"],
                        utc=True
                    )
                    .dt.strftime("%Y-%m-%d %H:%M:%S UTC")

                )

            st.dataframe(

                history_df,

                use_container_width=True,

                hide_index=True

            )


            # ---------------------------------------------
            # CSV EXPORT
            # ---------------------------------------------

            csv_data = history_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            title_col, gap_col, button_col, empty_col = st.columns(
                [1.0, 0.02, 0.7, 5],
                vertical_alignment="center"
            )

            with title_col:
                st.subheader("Get Records", anchor=False)

            with button_col:
                st.download_button(
                    label="Get CSV",
                    data=csv_data,
                    file_name="parkinsons_prediction_history.csv",
                    mime="text/csv",
                    help="Download prediction records as CSV",
                    key="download_prediction_csv"
                )


    except Exception as e:

        st.error(
            "Unable to retrieve prediction analytics from Atlas."
        )

        st.exception(e)
