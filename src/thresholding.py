
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


def create_ovr_model():

    return OneVsRestClassifier(
        LinearSVC(
            C=1.0,
            random_state=42,
            max_iter=5000
        )
    )


def generate_oof_scores(X_train, y_train):

    print("\nGenerating cross-validation decision scores...")

    classes = np.unique(y_train)

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    oof_scores = np.zeros((X_train.shape[0], len(classes)))

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train), start=1
    ):

        model = create_ovr_model()

        model.fit(X_train[train_idx], y_train[train_idx])

        oof_scores[val_idx] = model.decision_function(
            X_train[val_idx]
        )

        print(f"Completed fold {fold}")

    return oof_scores, classes


def tune_threshold_offsets(oof_scores, y_train, classes):

    print("\nTuning class-specific threshold offsets...")

    offsets = np.zeros(len(classes))

    candidates = np.linspace(-2.0, 2.0, 21)

    # Coordinate-wise search for offsets
    for _ in range(3):

        for class_idx in range(len(classes)):

            best_score = -1
            best_offset = offsets[class_idx]

            for candidate in candidates:

                trial_offsets = offsets.copy()
                trial_offsets[class_idx] = candidate

                predictions = classes[
                    np.argmax(
                        oof_scores + trial_offsets,
                        axis=1
                    )
                ]

                score = f1_score(
                    y_train,
                    predictions,
                    average="macro",
                    zero_division=0
                )

                if score > best_score:
                    best_score = score
                    best_offset = candidate

            offsets[class_idx] = best_offset

    print("Threshold offsets:", offsets)

    return offsets


def train_thresholded_svm(X_train, y_train):

    # 1. Generate out-of-fold scores
    oof_scores, classes = generate_oof_scores(
        X_train,
        y_train
    )

    # 2. Optimize class-specific offsets
    offsets = tune_threshold_offsets(
        oof_scores,
        y_train,
        classes
    )

    # 3. Train final One-vs-Rest model
    print("\nTraining final thresholded SVM...")

    model = create_ovr_model()
    model.fit(X_train, y_train)

    print("Thresholded SVM training completed!")

    return model, offsets


def predict_thresholded(model, X, offsets):

    scores = model.decision_function(X)

    adjusted_scores = scores + offsets

    predictions = model.classes_[
        np.argmax(adjusted_scores, axis=1)
    ]

    return predictions
