"""Part 2: Classification and evaluation.

Flipkart E-commerce Decision Intelligence Capstone.
This script performs:
1. Stratified 70/30 train/test split and standardized preprocessing.
2. Logistic regression from scratch with binary cross-entropy loss.
3. Learning-rate sensitivity analysis (eta in {0.01, 0.1, 1.0}).
4. Feature odds ratio calculation.
5. Out-of-sample evaluation at thresholds 0.5 and 0.3, plus blind accuracy.
6. Threshold sweep across 0.1..0.9 generating outputs/part2_thresholds.csv.
7. K-Nearest Neighbours from scratch with 5-fold cross-validation generating outputs/part2_knn_cv.csv.
8. Financial cost optimization (INR 50 false alarm, INR 400 missed churn).

Outputs written to:
- outputs/part2_results.txt
- outputs/part2_thresholds.csv
- outputs/part2_knn_cv.csv
"""

import csv
import math
import os
import numpy as np
import part2_models


def format_value(val):
    """Format a value according to the capstone specification:
    - booleans as yes / no (checked before int since bool subclasses int)
    - integers as plain integers
    - 4 decimal places for floating-point numbers
    - strings as given
    """
    if isinstance(val, (bool, np.bool_)):
        return "yes" if val else "no"
    elif isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (float, np.floating)):
        formatted = f"{val:.4f}"
        return "0.0000" if formatted == "-0.0000" else formatted
    return str(val)


def run_part2():
    # -------------------------------------------------------------------------
    # Task 2.1: Split and standardise
    # -------------------------------------------------------------------------
    split_data = part2_models.get_stratified_split()
    X_train_std = split_data["X_train_std"]
    y_train = split_data["y_train"]
    X_test_std = split_data["X_test_std"]
    y_test = split_data["y_test"]

    train_rows = len(y_train)
    train_positives = int(np.sum(y_train == 1))
    test_rows = len(y_test)
    test_positives = int(np.sum(y_test == 1))

    results = {}
    results["train_rows"] = train_rows
    results["train_positives"] = train_positives
    results["test_rows"] = test_rows
    results["test_positives"] = test_positives

    # -------------------------------------------------------------------------
    # Task 2.2: Logistic regression from scratch (eta = 0.1, 1000 updates)
    # -------------------------------------------------------------------------
    w, b, checkpoints_1000 = part2_models.train_logistic_regression(
        X_train_std, y_train, eta=0.1, epochs=1000
    )

    results["loss_epoch_0"] = checkpoints_1000[0]
    results["loss_epoch_1"] = checkpoints_1000[1]
    results["loss_epoch_10"] = checkpoints_1000[10]
    results["loss_epoch_100"] = checkpoints_1000[100]
    results["loss_epoch_1000"] = checkpoints_1000[1000]

    for j in range(7):
        results[f"weight_X{j+1}"] = float(w[j])
    results["bias"] = float(b)

    # Test BCE loss with final parameters
    test_probs = part2_models.predict_logistic_prob(X_test_std, w, b)
    test_bce = part2_models.bce_loss(y_test, test_probs)
    results["test_bce"] = test_bce

    # -------------------------------------------------------------------------
    # Task 2.3: Learning-rate effect (200 updates with eta in {0.01, 0.1, 1.0})
    # -------------------------------------------------------------------------
    for eta_val in [0.01, 0.1, 1.0]:
        _, _, chk_200 = part2_models.train_logistic_regression(
            X_train_std, y_train, eta=eta_val, epochs=200
        )
        results[f"loss_epoch_200_eta_{eta_val}"] = chk_200[200]

    # -------------------------------------------------------------------------
    # Task 2.4: Odds ratios (e^weight)
    # -------------------------------------------------------------------------
    for j in range(7):
        results[f"odds_ratio_X{j+1}"] = float(np.exp(w[j]))

    # -------------------------------------------------------------------------
    # Task 2.5: Evaluate on test rows at thresholds 0.5 and 0.3, plus blind accuracy
    # -------------------------------------------------------------------------
    for tau in [0.5, 0.3]:
        pred_labels = (test_probs >= tau).astype(int)
        m = part2_models.evaluate_confusion_metrics(y_test, pred_labels)
        prefix = f"tau_{tau}_"
        results[prefix + "tp"] = m["tp"]
        results[prefix + "fp"] = m["fp"]
        results[prefix + "tn"] = m["tn"]
        results[prefix + "fn"] = m["fn"]
        results[prefix + "accuracy"] = m["accuracy"]
        results[prefix + "precision"] = m["precision"]
        results[prefix + "recall"] = m["recall"]
        results[prefix + "f1"] = m["f1"]
        results[prefix + "specificity"] = m["specificity"]

    blind_accuracy = float(np.sum(y_test == 0) / len(y_test))
    results["blind_accuracy"] = blind_accuracy

    # -------------------------------------------------------------------------
    # Task 2.6: Threshold sweep (0.1 to 0.9) -> outputs/part2_thresholds.csv
    # -------------------------------------------------------------------------
    thresholds_rows = []
    for k in range(1, 10):
        tau_val = k / 10.0
        p_labels = (test_probs >= tau_val).astype(int)
        m = part2_models.evaluate_confusion_metrics(y_test, p_labels)
        thresholds_rows.append([
            f"{tau_val:.4f}",
            m["tp"],
            m["fp"],
            m["tn"],
            m["fn"],
            f"{m['tpr']:.4f}",
            f"{m['fpr']:.4f}"
        ])

    os.makedirs("outputs", exist_ok=True)
    thresholds_csv_path = os.path.join("outputs", "part2_thresholds.csv")
    with open(thresholds_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        writer.writerows(thresholds_rows)
    print(f"Generated {thresholds_csv_path} with {len(thresholds_rows)} rows.")

    # -------------------------------------------------------------------------
    # Task 2.7: KNN with 5-fold cross-validation on 560 training rows
    # -------------------------------------------------------------------------
    best_k, cv_records = part2_models.find_best_knn_k(X_train_std, y_train)
    knn_cv_rows = [[K, f"{mean_f1:.4f}"] for K, mean_f1 in cv_records]

    knn_cv_csv_path = os.path.join("outputs", "part2_knn_cv.csv")
    with open(knn_cv_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])
        writer.writerows(knn_cv_rows)
    print(f"Generated {knn_cv_csv_path} with {len(knn_cv_rows)} rows. Best K = {best_k}.")

    # Refit KNN with all 560 training rows on test set
    test_knn_preds, _ = part2_models.predict_knn(X_train_std, y_train, X_test_std, best_k)
    km = part2_models.evaluate_confusion_metrics(y_test, test_knn_preds)

    results["knn_best_k"] = best_k
    results["knn_tp"] = km["tp"]
    results["knn_fp"] = km["fp"]
    results["knn_tn"] = km["tn"]
    results["knn_fn"] = km["fn"]
    results["knn_accuracy"] = km["accuracy"]
    results["knn_precision"] = km["precision"]
    results["knn_recall"] = km["recall"]
    results["knn_f1"] = km["f1"]

    # -------------------------------------------------------------------------
    # Task 2.8: Choose the alert threshold with money
    # -------------------------------------------------------------------------
    train_probs = part2_models.predict_logistic_prob(X_train_std, w, b)
    best_tau = None
    min_train_cost = float("inf")

    for step in range(1, 20):
        cand_tau = step / 20.0
        cost, _, _ = part2_models.compute_financial_cost(y_train, train_probs, cand_tau)
        if cost < min_train_cost:
            min_train_cost = cost
            best_tau = cand_tau

    test_cost_at_tau, _, _ = part2_models.compute_financial_cost(y_test, test_probs, best_tau)
    test_cost_at_05, _, _ = part2_models.compute_financial_cost(y_test, test_probs, 0.5)
    test_cost_flag_nobody = 400 * int(np.sum(y_test == 1))

    results["cost_tau"] = best_tau
    results["cost_train_at_tau"] = int(min_train_cost)
    results["cost_test_at_tau"] = int(test_cost_at_tau)
    results["cost_test_at_0.5"] = int(test_cost_at_05)
    results["cost_test_flag_nobody"] = int(test_cost_flag_nobody)

    # -------------------------------------------------------------------------
    # Write outputs/part2_results.txt
    # -------------------------------------------------------------------------
    results_path = os.path.join("outputs", "part2_results.txt")
    with open(results_path, "w", newline="\n", encoding="utf-8") as f:
        for key, val in results.items():
            f.write(f"{key} = {format_value(val)}\n")

    print(f"Generated {results_path} with {len(results)} keys.")
    return results


if __name__ == "__main__":
    run_part2()

