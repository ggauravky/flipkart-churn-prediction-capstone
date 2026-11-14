"""Split, logistic regression, KNN and metrics shared by Parts 2 and 4.

Importing this module does not run any Part.
"""

import numpy as np
import data_loader


def get_stratified_split():
    """Create the stratified 70/30 train/test split preserving file order.

    Split rule:
    - First floor(7 * count / 10) of outcome 0 rows go to training, rest to test.
    - First floor(7 * count / 10) of outcome 1 rows go to training, rest to test.
    - Both subsets preserve original file order.
    - Features X1..X7 are standardized using training mean and sample SD (ddof=1).

    Returns:
        dict containing:
            - 'train_indices', 'test_indices'
            - 'X_train', 'y_train', 'X_test', 'y_test'
            - 'X_train_std', 'X_test_std'
            - 'train_mean', 'train_sd'
    """
    dataset = data_loader.load_dataset()
    X = dataset["X"]
    y = dataset["y"]
    n = len(y)

    idx_0 = [i for i in range(n) if y[i] == 0]
    idx_1 = [i for i in range(n) if y[i] == 1]

    n_train_0 = (7 * len(idx_0)) // 10
    n_train_1 = (7 * len(idx_1)) // 10

    train_idx_0 = idx_0[:n_train_0]
    test_idx_0 = idx_0[n_train_0:]

    train_idx_1 = idx_1[:n_train_1]
    test_idx_1 = idx_1[n_train_1:]

    train_indices = sorted(train_idx_0 + train_idx_1)
    test_indices = sorted(test_idx_0 + test_idx_1)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Use training statistics for both sets to prevent test-data leakage.
    train_mean = np.mean(X_train, axis=0)
    train_sd = np.std(X_train, axis=0, ddof=1)

    X_train_std = (X_train - train_mean) / train_sd
    X_test_std = (X_test - train_mean) / train_sd

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_std": X_train_std,
        "X_test_std": X_test_std,
        "train_mean": train_mean,
        "train_sd": train_sd
    }


def sigmoid(z):
    """Numerically stable sigmoid, 1 / (1 + e^-z), for a scalar or an array."""
    # For very negative z, e^-z overflows; there the equivalent form
    # e^z / (1 + e^z) is used instead. Both branches give the same value.
    z_arr = np.asarray(z, dtype=np.float64)
    safe = z_arr >= -500.0
    z_safe = np.where(safe, z_arr, 0.0)
    z_neg = np.where(safe, 0.0, z_arr)
    exp_neg = np.exp(z_neg)
    res = np.where(safe, 1.0 / (1.0 + np.exp(-z_safe)), exp_neg / (1.0 + exp_neg))
    return float(res.item()) if np.ndim(z) == 0 else res


def bce_loss(y_true, y_prob):
    """Mean binary cross-entropy: -(1/n) sum[y ln p + (1 - y) ln(1 - p)].

    Probabilities are clipped away from 0 and 1 so that ln() stays finite.
    """
    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def train_logistic_regression(X_std, y, eta=0.1, epochs=1000):
    """Train logistic regression by batch gradient descent, starting from zero.

    Returns (w, b, checkpoint_losses), where checkpoint_losses maps an update
    count (0, 1, 10, 100, 200, 500, 1000) to the training BCE at that point.
    """
    n_samples, n_features = X_std.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    checkpoint_losses = {}
    prob_0 = sigmoid(X_std @ w + b)
    checkpoint_losses[0] = bce_loss(y, prob_0)

    for step in range(1, epochs + 1):
        z = X_std @ w + b
        p = sigmoid(z)
        error = p - y

        # BCE gradients: dL/dw = (1/n) X^T (p - y), dL/db = (1/n) sum(p - y)
        grad_w = (1.0 / n_samples) * (X_std.T @ error)
        grad_b = (1.0 / n_samples) * np.sum(error)

        w -= eta * grad_w
        b -= eta * grad_b

        if step in [1, 10, 100, 200, 500, 1000]:
            z_curr = X_std @ w + b
            checkpoint_losses[step] = bce_loss(y, sigmoid(z_curr))

    return w, b, checkpoint_losses


def predict_logistic_prob(X_std, w, b):
    """Compute predicted positive class probabilities from logistic regression."""
    return sigmoid(X_std @ w + b)


def predict_knn(ref_X_std, ref_y, query_X_std, K):
    """K-nearest-neighbour classification on standardised features.

    A query is positive when at least (K + 1) // 2 of its K nearest reference
    rows are positive. Returns (predictions, positive_shares), where the share
    k_pos / K serves as the KNN "probability" in Part 4's soft vote.
    """
    # Euclidean distances, shape (n_query, n_ref)
    dists = np.sqrt(np.sum((query_X_std[:, np.newaxis, :] - ref_X_std[np.newaxis, :, :]) ** 2, axis=2))

    preds = []
    pos_shares = []
    threshold = (K + 1) // 2

    for i in range(len(query_X_std)):
        # Stable sort: equal distances keep file order, as the brief pins.
        ranked_idx = np.argsort(dists[i], kind="stable")
        top_k_labels = ref_y[ranked_idx[:K]]
        k_pos = int(np.sum(top_k_labels == 1))
        preds.append(1 if k_pos >= threshold else 0)
        pos_shares.append(k_pos / K)

    return np.array(preds, dtype=int), np.array(pos_shares, dtype=float)


def evaluate_confusion_metrics(y_true, y_pred):
    """Confusion-matrix cells and derived metrics; any 0/0 ratio is reported as 0."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr
    }


def compute_financial_cost(y_true, y_prob, tau, cost_fp=50, cost_fn=400):
    """Cost in INR at threshold tau: 50 per false alarm plus 400 per missed churner."""
    y_pred = (y_prob >= tau).astype(int)
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    cost = cost_fp * fp + cost_fn * fn
    return cost, fp, fn


def find_best_knn_k(X_train_std, y_train, odd_ks=None):
    """Choose K by 5-fold cross-validation on the training rows only.

    A row's fold is its 0-based position in the training set modulo 5. Each K
    is scored by its mean F1 over the five held-out folds. Features are not
    re-standardised inside a fold, as the brief requires.
    Returns (best_k, [(K, mean_f1), ...]).
    """
    if odd_ks is None:
        odd_ks = list(range(1, 23, 2))  # 1, 3, 5, ..., 21

    n_train = len(y_train)
    folds = np.array([i % 5 for i in range(n_train)])
    best_k = None
    best_mean_f1 = -1.0
    cv_records = []

    for K in odd_ks:
        fold_f1_scores = []
        for f in range(5):
            val_mask = (folds == f)
            ref_mask = ~val_mask

            ref_X = X_train_std[ref_mask]
            ref_y = y_train[ref_mask]
            val_X = X_train_std[val_mask]
            val_y = y_train[val_mask]

            val_preds, _ = predict_knn(ref_X, ref_y, val_X, K)
            val_m = evaluate_confusion_metrics(val_y, val_preds)
            fold_f1_scores.append(val_m["f1"])

        mean_f1 = float(np.mean(fold_f1_scores))
        cv_records.append((K, mean_f1))

        # K is tried in ascending order, so strict ">" keeps the smaller K on a tie.
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_k = K

    return best_k, cv_records


