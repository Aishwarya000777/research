

import os
import joblib
import pandas as pd

from src.dataset import load_scotus_dataset
from src.preprocessing import preprocess_text
from src.feature_extraction import extract_tfidf_features
from src.models import train_linear_svm, train_weighted_svm
from src.thresholding import train_thresholded_svm, predict_thresholded
from src.evaluation import (
    evaluate_model,
    evaluate_predictions,
    save_confusion_matrix
)


def run_experiment():

    os.makedirs("results", exist_ok=True)

    # 1. Load dataset
    train, validation, test = load_scotus_dataset()

    # 2. Combine training and validation
    train = pd.concat([train, validation], ignore_index=True)

    print("\nCombined training dataset:", train.shape)
    print("Final test dataset:", test.shape)

    # 3. Preprocess text
    train_texts = train["text"].apply(preprocess_text).tolist()
    test_texts = test["text"].apply(preprocess_text).tolist()

    y_train = train["label"].values
    y_test = test["label"].values

    # 4. TF-IDF feature extraction
    X_train, X_test, vectorizer = extract_tfidf_features(
        train_texts, test_texts
    )

    # 5. Train baseline OVR Linear SVM
    baseline_model = train_linear_svm(X_train, y_train)
    baseline_metrics = evaluate_model(
        baseline_model, X_test, y_test, "Baseline OVR Linear SVM"
    )

    # 6. Train cost-sensitive approximation
    weighted_model = train_weighted_svm(X_train, y_train)
    weighted_metrics = evaluate_model(
        weighted_model, X_test, y_test, "Cost-Sensitive Approximation"
    )

    # 7. Train thresholded model
    threshold_model, offsets = train_thresholded_svm(
        X_train, y_train
    )

    threshold_predictions = predict_thresholded(
        threshold_model, X_test, offsets
    )

    threshold_metrics = evaluate_predictions(
        y_test,
        threshold_predictions,
        "Thresholded Linear SVM"
    )

    # 8. Save test results
    results = pd.DataFrame([
        {"Model": "Baseline OVR", **baseline_metrics},
        {"Model": "Cost-Sensitive Approximation", **weighted_metrics},
        {"Model": "Thresholded SVM", **threshold_metrics}
    ])

    results.to_csv("results/test_metrics.csv", index=False)

    # 9. Save confusion matrix
    save_confusion_matrix(
        y_test,
        threshold_predictions,
        "results/test_confusion_matrix.png",
        "Thresholded Linear SVM - SCOTUS Test Set"
    )

    # 10. Save models and vectorizer
    artifacts = {
        "vectorizer": vectorizer,
        "baseline_model": baseline_model,
        "weighted_model": weighted_model,
        "threshold_model": threshold_model,
        "threshold_offsets": offsets
    }

    joblib.dump(artifacts, "results/model_artifacts.pkl")

    print("\nExperiment completed!")
    print("Test metrics saved to results/test_metrics.csv")
    print("Model artifacts saved successfully!")


if __name__ == "__main__":
    run_experiment()
