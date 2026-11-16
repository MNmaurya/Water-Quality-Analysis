
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Water Quality Assessment",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Explainable AI-Based Water Quality Assessment System")

st.write(
    "Predict water-quality categories using machine learning "
    "and inspect the features contributing to the prediction."
)

model = joblib.load(BASE_DIR / "water_quality_model.joblib")

with open(BASE_DIR / "metadata.json", "r") as file:
    metadata = json.load(file)

df = pd.read_csv(BASE_DIR / "water_quality_dataset.csv")

target = metadata["target_column"]
feature_columns = metadata["feature_columns"]

# Match the model's training feature types
X_reference = df[feature_columns].copy()

numeric_features = X_reference.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = [
    col for col in feature_columns
    if col not in numeric_features
]

st.sidebar.header("Water Parameters")
st.sidebar.caption("Enter values for a water-quality assessment.")

user_input = {}

for col in feature_columns:

    if col in numeric_features:
        series = pd.to_numeric(X_reference[col], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()

        if series.empty:
            user_input[col] = 0.0
            continue

        min_value = float(series.min())
        max_value = float(series.max())
        median_value = float(series.median())

        # Sliders need a nonzero range
        if min_value == max_value:
            user_input[col] = min_value
        else:
            user_input[col] = st.sidebar.slider(
                col,
                min_value=min_value,
                max_value=max_value,
                value=min(max(median_value, min_value), max_value)
            )

    else:
        options = sorted(
            X_reference[col].dropna().astype(str).unique().tolist()
        )

        if not options:
            user_input[col] = "Unknown"
        else:
            user_input[col] = st.sidebar.selectbox(
                col,
                options=options
            )

input_df = pd.DataFrame(
    [user_input],
    columns=feature_columns
)

st.subheader("Assessment Input")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Water Quality", type="primary"):

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction")
    st.success(f"Predicted Water Quality Class: {prediction}")

    st.warning(
        "This is a model prediction, not a laboratory certification "
        "of drinking-water safety."
    )


    # Recommended uses based on predicted water quality class
    water_use_recommendations = {
        "Excellent": {
            "status": "More suitable for general water uses",
            "uses": [
                "Gardening and irrigation",
                "Washing and cleaning",
                "Bathing and recreational use, subject to safety standards"
            ],
            "caution": "Drinking use still requires appropriate treatment and laboratory verification."
        },
        "Good": {
            "status": "Suitable for some general uses, subject to testing",
            "uses": [
                "Gardening and irrigation, where suitable",
                "Washing and cleaning",
                "Other non-potable uses after checking relevant requirements"
            ],
            "caution": "Do not assume it is safe for drinking or direct contact without proper testing."
        },
        "Poor": {
            "status": "Limited potential uses",
            "uses": [
                "Consider restricted uses only after parameter-specific assessment",
                "Further treatment and testing before domestic use"
            ],
            "caution": "Avoid drinking and recreational contact unless treatment and testing confirm safety."
        },
        "Very Poor": {
            "status": "Not recommended for direct use",
            "uses": [
                "Further water-quality assessment",
                "Appropriate treatment before considering any use"
            ],
            "caution": "Avoid direct domestic, drinking, bathing, and recreational use."
        }
    }

    recommendation = water_use_recommendations.get(str(prediction))

    st.markdown("---")
    st.subheader("💧 Recommended Water Uses")
    st.caption("Indicative recommendations based on the predicted category.")

    if recommendation:
        st.markdown(f"### {recommendation['status']}")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Potential uses**")
            for use in recommendation["uses"]:
                st.markdown(f"✓ {use}")

        with col2:
            st.markdown("**Important note**")
            st.info(recommendation["caution"])
    else:
        st.info("No use recommendations are available for this predicted category.")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]

        probability_df = pd.DataFrame({
            "Class": model.classes_,
            "Predicted Probability": probabilities
        }).sort_values(
            "Predicted Probability",
            ascending=False
        )

        st.subheader("Model Class Probabilities")
        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Explainable AI — SHAP")

    try:
        preprocessor = model.named_steps["preprocessor"]
        tree_model = model.named_steps["model"]

        transformed_input = preprocessor.transform(input_df)
        transformed_columns = preprocessor.get_feature_names_out()

        transformed_df = pd.DataFrame(
            transformed_input,
            columns=transformed_columns
        )

        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(transformed_df)

        classes = list(tree_model.classes_)
        class_index = classes.index(prediction)

        if isinstance(shap_values, list):
            contributions = np.asarray(
                shap_values[class_index]
            )[0]

        else:
            shap_array = np.asarray(shap_values)

            if shap_array.ndim == 3:
                if shap_array.shape[2] == len(classes):
                    contributions = shap_array[0, :, class_index]
                elif shap_array.shape[0] == len(classes):
                    contributions = shap_array[class_index, 0, :]
                else:
                    raise ValueError("Unexpected SHAP output shape.")

            elif shap_array.ndim == 2:
                contributions = shap_array[0]

            else:
                raise ValueError("Unexpected SHAP output dimensions.")

        explanation_df = pd.DataFrame({
            "Feature": transformed_columns,
            "SHAP Contribution": contributions
        })

        explanation_df["Absolute Contribution"] = (
            explanation_df["SHAP Contribution"].abs()
        )

        explanation_df = explanation_df.sort_values(
            "Absolute Contribution",
            ascending=False
        ).head(10).sort_values("SHAP Contribution")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            explanation_df["Feature"],
            explanation_df["SHAP Contribution"]
        )

        ax.set_xlabel("SHAP contribution to the predicted class")
        ax.set_title("Top Feature Contributions")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "SHAP contributions explain how features influence the "
            "model's output. They do not establish causation or "
            "independently verify water safety."
        )

    except Exception as error:
        st.error(
            "The prediction succeeded, but the SHAP explanation "
            f"could not be generated: {error}"
        )

st.divider()

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Monitoring Stations", len(df))
col2.metric("Input Features", len(feature_columns))
col3.metric("Quality Classes", df[target].nunique())

st.subheader("Water Quality Class Distribution")

class_counts = df[target].value_counts()

st.bar_chart(class_counts)

# st.subheader("Model Comparison")

# comparison_path = BASE_DIR / "model_comparison.csv"

# if comparison_path.exists():
#     comparison_df = pd.read_csv(comparison_path)
#     st.dataframe(
#         comparison_df,
#         use_container_width=True,
#         hide_index=True
#     )
