"""Part 4: Model choice, boosting, and architecture sizing.

Flipkart E-commerce Decision Intelligence Capstone.
This script performs:
1. Decision tree split evaluation (S1..S4) via entropy, Gini, and information gains.
   Outputs -> outputs/part4_splits.csv.
2. Decision stump & ensemble voting comparison (logistic, knn, stump, hard_vote, soft_vote).
   Outputs -> outputs/part4_voting.csv.
3. Two manual rounds of AdaBoost on the 8-point customer toy table.
   Outputs -> outputs/part4_adaboost.csv.
4. Out-of-bag (OOB) sampling probability calculation.
5. Neural network architecture parameter sizing for Model A (MLP vs CNN) and Model B (RNN).
   Outputs -> outputs/part4_architecture.csv.
6. Summary output generation -> outputs/part4_results.txt.
"""

import csv
import math
import os
import numpy as np
import part2_models


def format_value(val):
    """Format output value per assignment specifications."""
    if isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    elif isinstance(val, bool):
        return "yes" if val else "no"
    return str(val)


def run_part4():
    # -------------------------------------------------------------------------
    # Reusing Part 2 stratified 70/30 split and models via part2_models
    # -------------------------------------------------------------------------
    split_data = part2_models.get_stratified_split()
    X_train = split_data["X_train"]          # 560 training rows in original units
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]            # 240 test rows in original units
    y_test = split_data["y_test"]
    X_train_std = split_data["X_train_std"]  # Standardized features
    X_test_std = split_data["X_test_std"]

    n_train = len(y_train)
    p_pos = float(np.sum(y_train == 1) / n_train)
    p_neg = 1.0 - p_pos

    results = {}

    # =========================================================================
    # Task 4.1: Which split is purest?
    # =========================================================================
    # Root entropy and Gini
    root_entropy = float(-(p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg)))
    root_gini = float(1.0 - (p_pos ** 2 + p_neg ** 2))

    results["root_entropy"] = root_entropy
    results["root_gini"] = root_gini

    # Candidate splits on features in original units:
    # S1: X6 == 1 (cod_flag)
    # S2: X7 == 1 (plus_member_flag)
    # S3: X3 > 0.5 (discount_order_share)
    # S4: X1 <= 24 (customer_tenure_months)
    def apply_split(split_name, X):
        if split_name == "S1":
            return X[:, 5] == 1
        elif split_name == "S2":
            return X[:, 6] == 1
        elif split_name == "S3":
            return X[:, 2] > 0.5
        elif split_name == "S4":
            return X[:, 0] <= 24
        else:
            raise ValueError(f"Unknown split: {split_name}")

    splits_dict = {s_id: apply_split(s_id, X_train) for s_id in ["S1", "S2", "S3", "S4"]}

    def evaluate_split(yes_mask, y):
        n = len(y)
        n_yes = int(np.sum(yes_mask))
        pos_yes = int(np.sum(y[yes_mask] == 1))
        n_no = n - n_yes
        pos_no = int(np.sum(y[~yes_mask] == 1))

        def node_h(pos, tot):
            if tot == 0 or pos == 0 or pos == tot:
                return 0.0
            p1 = pos / tot
            p0 = 1.0 - p1
            return -(p1 * np.log2(p1) + p0 * np.log2(p0))

        def node_g(pos, tot):
            if tot == 0:
                return 0.0
            p1 = pos / tot
            p0 = 1.0 - p1
            return 1.0 - (p1 ** 2 + p0 ** 2)

        h_yes = node_h(pos_yes, n_yes)
        h_no = node_h(pos_no, n_no)
        w_entropy = (n_yes / n) * h_yes + (n_no / n) * h_no
        info_gain = root_entropy - w_entropy

        g_yes = node_g(pos_yes, n_yes)
        g_no = node_g(pos_no, n_no)
        w_gini = (n_yes / n) * g_yes + (n_no / n) * g_no
        gini_gain = root_gini - w_gini

        return n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain

    splits_rows = []
    best_ig = (-1.0, "")
    best_gg = (-1.0, "")

    for s_id in ["S1", "S2", "S3", "S4"]:
        ny, py, nn, pn, ig, gg = evaluate_split(splits_dict[s_id], y_train)
        splits_rows.append([s_id, ny, py, nn, pn, f"{ig:.4f}", f"{gg:.4f}"])
        if ig > best_ig[0]:
            best_ig = (ig, s_id)
        if gg > best_gg[0]:
            best_gg = (gg, s_id)

    results["best_split_by_info_gain"] = best_ig[1]
    results["best_split_by_gini_gain"] = best_gg[1]

    os.makedirs("outputs", exist_ok=True)
    splits_csv_path = os.path.join("outputs", "part4_splits.csv")
    with open(splits_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"])
        w.writerows(splits_rows)
    print(f"Generated {splits_csv_path} with {len(splits_rows)} rows.")

    # =========================================================================
    # Task 4.2: Decision stump & voting combinations
    # =========================================================================
    best_split_name = best_ig[1]
    stump_yes_train = apply_split(best_split_name, X_train)
    stump_rate_yes = float(np.sum(y_train[stump_yes_train] == 1) / np.sum(stump_yes_train))
    stump_rate_no = float(np.sum(y_train[~stump_yes_train] == 1) / np.sum(~stump_yes_train))

    results["stump_rate_yes"] = stump_rate_yes
    results["stump_rate_no"] = stump_rate_no

    # 1. Logistic regression (Task 2.2 model)
    w_log, b_log, _ = part2_models.train_logistic_regression(X_train_std, y_train, eta=0.1, epochs=1000)
    prob_logistic = part2_models.predict_logistic_prob(X_test_std, w_log, b_log)
    pred_logistic = (prob_logistic >= 0.3).astype(int)

    # 2. KNN model (reusing best K selected in Part 2)
    best_k, _ = part2_models.find_best_knn_k(X_train_std, y_train)
    pred_knn, prob_knn = part2_models.predict_knn(X_train_std, y_train, X_test_std, best_k)

    # 3. Decision stump on test set
    stump_yes_test = apply_split(best_split_name, X_test)
    prob_stump = np.where(stump_yes_test, stump_rate_yes, stump_rate_no)
    pred_stump = (prob_stump >= 0.3).astype(int)

    # 4. Hard vote: positive when at least 2 of 3 predict positive
    pred_hard = ((pred_logistic + pred_knn + pred_stump) >= 2).astype(int)

    # 5. Soft vote: mean of 3 probabilities >= 0.3
    prob_soft = (prob_logistic + prob_knn + prob_stump) / 3.0
    pred_soft = (prob_soft >= 0.3).astype(int)

    models_list = [
        ("logistic", pred_logistic),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard),
        ("soft_vote", pred_soft)
    ]

    voting_rows = []
    for m_name, p_vec in models_list:
        m = part2_models.evaluate_confusion_metrics(y_test, p_vec)
        voting_rows.append([
            m_name,
            m["tp"],
            m["fp"],
            m["tn"],
            m["fn"],
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}"
        ])

    voting_csv_path = os.path.join("outputs", "part4_voting.csv")
    with open(voting_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"])
        w.writerows(voting_rows)
    print(f"Generated {voting_csv_path} with {len(voting_rows)} rows.")

    # =========================================================================
    # Task 4.3: Two rounds of AdaBoost by hand
    # =========================================================================
    y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    # Round 1
    w0 = np.full(8, 1.0 / 8.0)
    err1 = float(np.sum(w0[h1 != y_ada]))
    alpha1 = float(0.5 * np.log((1.0 - err1) / err1))
    w1_unnorm = w0 * np.exp(-alpha1 * y_ada * h1)
    w1_norm = w1_unnorm / np.sum(w1_unnorm)

    # Round 2
    err2 = float(np.sum(w1_norm[h2 != y_ada]))
    alpha2 = float(0.5 * np.log((1.0 - err2) / err2))
    w2_unnorm = w1_norm * np.exp(-alpha2 * y_ada * h2)
    w2_norm = w2_unnorm / np.sum(w2_unnorm)

    ensemble_pred = np.sign(alpha1 * h1 + alpha2 * h2)
    ensemble_acc = float(np.mean(ensemble_pred == y_ada))

    results["round1_error"] = err1
    results["round1_alpha"] = alpha1
    results["round2_error"] = err2
    results["round2_alpha"] = alpha2
    results["ensemble_accuracy"] = ensemble_acc

    adaboost_rows = []
    for p in range(8):
        adaboost_rows.append([
            1, p + 1, f"{w0[p]:.4f}", f"{w1_unnorm[p]:.4f}", f"{w1_norm[p]:.4f}"
        ])
    for p in range(8):
        adaboost_rows.append([
            2, p + 1, f"{w1_norm[p]:.4f}", f"{w2_unnorm[p]:.4f}", f"{w2_norm[p]:.4f}"
        ])

    adaboost_csv_path = os.path.join("outputs", "part4_adaboost.csv")
    with open(adaboost_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "point", "weight_before", "weight_unnormalised", "weight_after"])
        w.writerows(adaboost_rows)
    print(f"Generated {adaboost_csv_path} with {len(adaboost_rows)} rows.")

    # =========================================================================
    # Task 4.4: Out-of-bag share
    # =========================================================================
    oob_fraction = float((1.0 - 1.0 / 560.0) ** 560)
    results["oob_fraction"] = oob_fraction

    # =========================================================================
    # Task 4.5: Size neural networks (counting parameters only)
    # =========================================================================
    # Model A Option 1: MLP 784 -> 16 -> 16 -> 10
    mlp_hidden1 = 784 * 16 + 16
    mlp_hidden2 = 16 * 16 + 16
    mlp_output = 16 * 10 + 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    # Model A Option 2: CNN
    # cnn_conv1: 5x5 filters, 6 filters, 1 input channel, stride 1, padding 2
    # weights: 5 * 5 * 1 * 6 = 150, biases: 6 -> 156
    cnn_conv1 = 5 * 5 * 1 * 6 + 6

    # cnn_conv2: 5x5 filters, 16 filters, 6 input channels, stride 1, padding 0
    # weights: 5 * 5 * 6 * 16 = 2400, biases: 16 -> 2416
    cnn_conv2 = 5 * 5 * 6 * 16 + 16

    # Spatial after pool2: (10 - 2)/2 + 1 = 5 -> flatten: 5 * 5 * 16 = 400
    # cnn_dense1: 400 inputs, 32 neurons -> 400 * 32 + 32 = 12832
    cnn_dense1 = 400 * 32 + 32

    # cnn_dense2: 32 inputs, 10 neurons -> 32 * 10 + 10 = 330
    cnn_dense2 = 32 * 10 + 10

    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    architecture_rows = [
        ["mlp_hidden1", mlp_hidden1],
        ["mlp_hidden2", mlp_hidden2],
        ["mlp_output", mlp_output],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2]
    ]

    arch_csv_path = os.path.join("outputs", "part4_architecture.csv")
    with open(arch_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "parameters"])
        w.writerows(architecture_rows)
    print(f"Generated {arch_csv_path} with {len(architecture_rows)} rows.")

    results["mlp_total"] = mlp_total
    results["cnn_total"] = cnn_total

    # Scaling comparison: FC vs Conv on 128x128x3 input
    # FC: 128 * 128 * 3 = 49152 inputs -> 1024 neurons
    fc_layer_params = 49152 * 1024 + 1024
    # Conv: 64 filters of 3x3x3 + 64 biases
    conv_layer_params = (3 * 3 * 3) * 64 + 64

    results["fc_layer_params"] = fc_layer_params
    results["conv_layer_params"] = conv_layer_params

    # Model B: Recurrent network (X_t: 64, H_t: 128, Y: 16)
    # W_xh: 128 x 64 = 8192
    # W_hh: 128 x 128 = 16384
    # b_h: 128
    # W_hy: 16 x 128 = 2048
    # b_y: 16
    rnn_total = (128 * 64) + (128 * 128) + 128 + (16 * 128) + 16
    results["rnn_total"] = rnn_total

    # =========================================================================
    # Write outputs/part4_results.txt
    # =========================================================================
    results_path = os.path.join("outputs", "part4_results.txt")
    with open(results_path, "w", newline="\n", encoding="utf-8") as f:
        for key, val in results.items():
            f.write(f"{key} = {format_value(val)}\n")

    print(f"Generated {results_path} with {len(results)} keys.")
    return results


if __name__ == "__main__":
    run_part4()

