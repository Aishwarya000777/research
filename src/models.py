
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


def train_linear_svm(X_train, y_train):

    model = OneVsRestClassifier(
        LinearSVC(
            C=1.0,
            random_state=42,
            max_iter=5000
        )
    )

    print("\nTraining One-vs-Rest Linear SVM...")

    model.fit(X_train, y_train)

    print("One-vs-Rest training completed!")

    return model


def train_weighted_svm(X_train, y_train):

    model = OneVsRestClassifier(
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=5000
        )
    )

    print("\nTraining Cost-Sensitive Linear SVM approximation...")

    model.fit(X_train, y_train)

    print("Weighted OVR training completed!")

    return model

