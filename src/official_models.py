
import numpy as np
from scipy.sparse import csr_matrix

from libmultilabel import linear


def convert_labels_to_sparse(y, num_classes=13):
    """
    Convert integer class labels into a one-hot CSR matrix.
    """
    y = np.asarray(y, dtype=int)

    rows = np.arange(len(y))
    columns = y

    data = np.ones(len(y), dtype=np.float64)

    return csr_matrix(
        (data, (rows, columns)),
        shape=(len(y), num_classes)
    )


def train_official_models(X_train, y_train):

    Y_train = convert_labels_to_sparse(y_train)

    print("\nTraining official LibMultiLabel OVR...")
    baseline = linear.train_1vsrest(
        Y_train,
        X_train,
        multiclass=True,
        options="-s 2 -B 1 -e 0.0001 -q"
    )

    print("\nTraining official LibMultiLabel Thresholding...")
    thresholded = linear.train_thresholding(
        Y_train,
        X_train,
        multiclass=True,
        options="-s 2 -B 1 -e 0.0001 -q"
    )

    print("\nTraining official LibMultiLabel Cost-Sensitive...")
    cost_sensitive = linear.train_cost_sensitive(
        Y_train,
        X_train,
        multiclass=True,
        options="-s 2 -B 1 -e 0.0001 -q"
    )

    return {
        "official_baseline": baseline,
        "official_thresholded": thresholded,
        "official_cost_sensitive": cost_sensitive
    }


def predict_official_model(model, X):

    scores = model.predict_values(X)

    predictions = np.argmax(scores, axis=1)

    return predictions, scores
