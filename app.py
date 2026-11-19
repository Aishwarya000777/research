
import streamlit as st
import joblib
import time

from src.preprocessing import preprocess_text
from src.thresholding import predict_thresholded


# Load trained artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load("results/model_artifacts.pkl")


artifacts = load_artifacts()

vectorizer = artifacts["vectorizer"]

models = {
    "Baseline Linear SVM": artifacts["baseline_model"],
    "Cost-Sensitive Linear SVM": artifacts["weighted_model"],
    "Thresholded Linear SVM": artifacts["threshold_model"]
}

offsets = artifacts["threshold_offsets"]


# Page configuration
st.set_page_config(
    page_title="Linear Classifier - Legal Text Classification",
    page_icon="⚖️",
    layout="wide"
)


# Application heading
st.title("⚖️ Legal Text Classification")
st.caption(
    "Linear Classifier: An Often-Forgotten Baseline for Text Classification"
)

st.markdown("""
Classify Supreme Court legal judgments using three different
Linear SVM approaches trained on the SCOTUS dataset.
""")


# Input section
st.subheader("Enter Legal Document")

document = st.text_area(
    "Paste your legal judgment below:",
    height=250,
    placeholder="Paste the contents of a legal judgment here..."
)


# Model selection
st.subheader("Select Classification Model")

selected_model = st.selectbox(
    "Choose a model:",
    list(models.keys())
)


# Prediction
if st.button("Classify Document", type="primary"):

    if not document.strip():
        st.warning("Please enter a legal document first.")

    else:
        start_time = time.time()

        # Preprocess document
        processed_text = preprocess_text(document)

        # Extract TF-IDF features
        features = vectorizer.transform([processed_text])

        # Generate prediction
        if selected_model == "Thresholded Linear SVM":

            prediction = predict_thresholded(
                models[selected_model],
                features,
                offsets
            )[0]

            scores = (
                models[selected_model].decision_function(features)
                + offsets
            )[0]

        else:

            model = models[selected_model]

            prediction = model.predict(features)[0]
            scores = model.decision_function(features)[0]

        processing_time = time.time() - start_time

        # Display results
        st.success("Classification completed!")

        st.metric(
            "Predicted Legal Category",
            f"Class {int(prediction) + 1}"
        )

        st.metric(
            "Processing Time",
            f"{processing_time:.4f} seconds"
        )

        # Display class decision scores
        st.subheader("Decision Scores by Class")

        st.bar_chart({
            f"Class {i + 1}": float(score)
            for i, score in enumerate(scores)
        })

        st.info(
            "Decision scores indicate relative classifier confidence. "
            "They are not calibrated probabilities."
        )
