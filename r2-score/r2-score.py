import numpy as np

def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Handle constant-target case
    if ss_tot == 0.0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0

    return float(1.0 - ss_res / ss_tot)