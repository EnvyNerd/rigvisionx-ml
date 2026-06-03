"""Model evaluation metrics"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error


def classification_metrics(y_true, y_pred) -> dict:
    """Calculate classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def regression_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
    }
