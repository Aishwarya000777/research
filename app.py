

import streamlit as st
import joblib
import time
import numpy as np
from scipy import sparse

from src.preprocessing import preprocess_text, preprocess_official_text
from src.thresholding import predict_thresholded


# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------


def official_predict_values(model, features):
    bias = model.bias

    bias_col = np.full(
        (features.shape[0], 1 if bias > 0 else 0),
        bias
    )

    num_features = model.weights.shape[0]
    num_features -= 1 if bias > 0 else 0

    if features.shape[1] < num_features:
        x = sparse.hstack(
            [
                features,
                np.zeros((features.shape[0], num_features - features.shape[1])),
                bias_col
            ],
            format="csr"
        )
    else:
        x = sparse.hstack(
            [
                features[:, :num_features],
                bias_col
            ],
            format="csr"
        )

    return np.asarray(x @ model.weights) + model.thresholds
@st.cache_resource
def load_artifacts():
    original = joblib.load("results/model_artifacts.pkl")
    official = joblib.load("results/official/official_model_artifacts.pkl")
    return original, official


original_artifacts, official_artifacts = load_artifacts()

original_vectorizer = original_artifacts["vectorizer"]
official_vectorizer = official_artifacts["vectorizer"]

original_models = {
    "Baseline Linear SVM": original_artifacts["baseline_model"],
    "Cost-Sensitive Linear SVM": original_artifacts["weighted_model"],
    "Thresholded Linear SVM": original_artifacts["threshold_model"]
}

official_models = {
    "Official OVR (LibMultiLabel)": official_artifacts["models"]["official_baseline"],
    "Official Thresholding (LibMultiLabel)": official_artifacts["models"]["official_thresholded"],
    "Official Cost-Sensitive (LibMultiLabel)": official_artifacts["models"]["official_cost_sensitive"]
}

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Linear Classifier - Legal Text Classification",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Text Classification")

st.caption(
    "Linear Classifier: An Often-Forgotten Baseline for Text Classification"
)

st.markdown("""
Classify Supreme Court legal judgments using linear classification
methods implemented with both Scikit-learn and LibMultiLabel.
""")

# --------------------------------------------------
# INPUT
# --------------------------------------------------

st.subheader("Enter Legal Document")

document = st.text_area(
    "Paste your legal judgment below:",
    height=250,
    placeholder="Paste the contents of a legal judgment here..."
)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.subheader("Select Classification Model")

model_options = {
    **original_models,
    **official_models
}

selected_model = st.selectbox(
    "Choose a model:",
    list(model_options.keys())
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Classify Document", type="primary"):

    if not document.strip():
        st.warning("Please enter a legal document first.")

    else:
        start_time = time.time()

        if selected_model in original_models:

            processed_text = preprocess_text(document)
            features = original_vectorizer.transform([processed_text])
            model = original_models[selected_model]

            if selected_model == "Thresholded Linear SVM":

                prediction = predict_thresholded(
                    model,
                    features,
                    original_artifacts["threshold_offsets"]
                )[0]

                scores = (
                    model.decision_function(features)
                    + original_artifacts["threshold_offsets"]
                )[0]

            else:
                prediction = model.predict(features)[0]
                scores = model.decision_function(features)[0]

        else:

            processed_text = preprocess_official_text(document)
            features = official_vectorizer.transform([processed_text])
            model = official_models[selected_model]

            scores = official_predict_values(model, features)
            scores = np.squeeze(scores)

            prediction = int(np.argmax(scores))

        processing_time = time.time() - start_time

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        st.success("Classification completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Legal Category",
                f"Class {int(prediction) + 1}"
            )

        with col2:
            st.metric(
                "Processing Time",
                f"{processing_time:.4f} seconds"
            )

        st.subheader("Decision Scores by Class")

        st.bar_chart({
            f"Class {i + 1}": float(score)
            for i, score in enumerate(scores)
        })

        st.info(
            "Decision scores indicate relative classifier scores. "
            "They are not calibrated probabilities."
        )
