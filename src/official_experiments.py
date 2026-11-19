
import os
import joblib
import pandas as pd
from src.preprocessing import preprocess_official_text
from src.dataset import load_scotus_dataset
from src.preprocessing import preprocess_text
from src.feature_extraction import extract_tfidf_features

from src.official_models import (
    train_official_models,
    predict_official_model
)

from src.evaluation import (
    evaluate_predictions,
    save_confusion_matrix
)


def run_official_experiment():

    os.makedirs("results/official", exist_ok=True)

    # 1. Load dataset
    train, validation, test = load_scotus_dataset()

    # 2. Combine training and validation
    train = pd.concat(
        [train, validation],
        ignore_index=True
    )

    print("\nCombined training dataset:", train.shape)
    print("Final test dataset:", test.shape)

    # 3. Preprocess text
    train_texts = train["text"].apply(
        preprocess_official_text
    ).tolist()

    test_texts = test["text"].apply(
        preprocess_official_text
    ).tolist()

    y_train = train["label"].values
    y_test = test["label"].values

    # 4. TF-IDF
    X_train, X_test, vectorizer = extract_tfidf_features(
        train_texts,
        test_texts
    )

    # 5. Train official LibMultiLabel models
    models = train_official_models(
        X_train,
        y_train
    )

    results = []

    # 6. Evaluate each official model
    for model_name, model in models.items():

        print(f"\nEvaluating {model_name}...")

        predictions, scores = predict_official_model(
            model,
            X_test
        )

        metrics = evaluate_predictions(
            y_test,
            predictions,
            model_name
        )

        results.append({
            "Model": model_name,
            **metrics
        })

        # Save separate confusion matrix
        save_confusion_matrix(
            y_test,
            predictions,
            f"results/official/{model_name}_confusion_matrix.png",
            f"{model_name} - SCOTUS Test Set"
        )

    # 7. Save comparison
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/official/official_metrics.csv",
        index=False
    )

    # 8. Save official model artifacts
    artifacts = {
        "vectorizer": vectorizer,
        "models": models
    }

    joblib.dump(
        artifacts,
        "results/official/official_model_artifacts.pkl"
    )

    print("\nOfficial experiments completed!")
    print("Results saved in results/official/")


if __name__ == "__main__":
    run_official_experiment()
