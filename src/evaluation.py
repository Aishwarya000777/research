
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)


def evaluate_predictions(y, predictions, split_name="Test"):

    accuracy = accuracy_score(y, predictions)

    micro_f1 = f1_score(
        y, predictions, average="micro", zero_division=0
    )

    macro_f1 = f1_score(
        y, predictions, average="macro", zero_division=0
    )

    weighted_f1 = f1_score(
        y, predictions, average="weighted", zero_division=0
    )

    precision, recall, _, _ = precision_recall_fscore_support(
        y, predictions, average="macro", zero_division=0
    )

    print(f"\n{split_name} Results")
    print("-" * 35)
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Micro-F1:     {micro_f1:.4f}")
    print(f"Macro-F1:     {macro_f1:.4f}")
    print(f"Weighted-F1:  {weighted_f1:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")

    return {
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_macro": precision,
        "recall_macro": recall
    }


def evaluate_model(model, X, y, split_name="Test"):

    predictions = model.predict(X)

    return evaluate_predictions(
        y, predictions, split_name
    )


def save_confusion_matrix(
    y_true,
    predictions,
    filename,
    title
):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(13))
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Confusion matrix saved: {filename}")
