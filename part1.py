"""Part 1: Estimation, inference, and a regression baseline.

Flipkart E-commerce Decision Intelligence Capstone.
This script performs:
1. Estimation & 95% Confidence Interval for order value.
2. Hypothesis testing against the INR 800 planning assumption.
3. Sample size determination for a target margin of error.
4. Simple linear regression via closed-form equations.
5. Batch Gradient Descent optimization and convergence comparison.
6. Heteroskedasticity (funnel) diagnostic and log-log transformation.
7. Multiple linear regression via the normal equation.

Outputs written to outputs/part1_results.txt.
"""

import math
import os
import numpy as np
import data_loader


def format_value(val):
    """Format a value according to the capstone specification:
    - 4 decimal places for floating-point numbers
    - integers as plain integers
    - booleans / strings as given
    """
    if isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    elif isinstance(val, bool):
        return "yes" if val else "no"
    return str(val)


def run_part1():
    # Load dataset using our lightweight helper
    data = data_loader.load_dataset()
    val = data["value"]  # order_value_inr_hundreds (target for Part 1)
    n = len(val)         # Total records = 800
    x4 = data["avg_order_value_inr_hundreds"]
    X_features = data["X"]

    results = {}

    # =========================================================================
    # Task 1.1: Estimate with an interval
    # =========================================================================
    # Sample mean and sample standard deviation (divisor n - 1)
    mean_value = float(np.mean(val))
    sd_value = float(np.std(val, ddof=1))
    se_value = sd_value / math.sqrt(n)

    # 95% Confidence interval using critical value 1.96 (large sample n >> 30)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    results["n"] = n
    results["mean_value"] = mean_value
    results["sd_value"] = sd_value
    results["se_value"] = se_value
    results["ci95_low"] = ci95_low
    results["ci95_high"] = ci95_high

    # =========================================================================
    # Task 1.2: Test a planning claim
    # =========================================================================
    # Test H0: mu = 8.0 vs H1: mu != 8.0 at alpha = 0.05
    mu_0 = 8.0
    z_stat = (mean_value - mu_0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    results["z_stat"] = z_stat
    results["reject_h0"] = reject_h0

    # =========================================================================
    # Task 1.3: Size a future sample
    # =========================================================================
    # Smallest integer n such that n >= (1.96 * sd_value / E)^2 with E = 0.5
    margin_error = 0.5
    n_required = math.ceil((1.96 * sd_value / margin_error) ** 2)

    results["n_required"] = n_required

    # =========================================================================
    # Task 1.4: Simple regression by closed form
    # =========================================================================
    # Regress value on X4 using manual closed-form formulas (sample divisor n - 1)
    mean_x4 = float(np.mean(x4))
    cov_x4_val = float(np.sum((x4 - mean_x4) * (val - mean_value)) / (n - 1))
    var_x4 = float(np.sum((x4 - mean_x4) ** 2) / (n - 1))

    reg_slope = cov_x4_val / var_x4
    reg_intercept = mean_value - reg_slope * mean_x4
    fitted_raw = reg_intercept + reg_slope * x4

    # Sum of squares decomposition
    tss = float(np.sum((val - mean_value) ** 2))
    ess = float(np.sum((fitted_raw - mean_value) ** 2))
    rss = float(np.sum((val - fitted_raw) ** 2))
    tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"

    r2 = 1.0 - (rss / tss)
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(val - fitted_raw)))

    results["reg_slope"] = reg_slope
    results["reg_intercept"] = reg_intercept
    results["tss"] = tss
    results["ess"] = ess
    results["rss"] = rss
    results["tss_equals_ess_plus_rss"] = tss_equals_ess_plus_rss
    results["r2"] = r2
    results["rmse"] = rmse
    results["mae"] = mae

    # =========================================================================
    # Task 1.5: Fit the same line by gradient descent
    # =========================================================================
    # Standardise X4 using training sample SD
    sd_x4 = float(np.std(x4, ddof=1))
    z = (x4 - mean_x4) / sd_x4

    # Model: y_hat = m * z + c, initialised to m = 0.0, c = 0.0
    m, c = 0.0, 0.0
    eta = 0.1
    gd_checkpoints = {}

    # Cost J before any update (iteration 0)
    e_0 = val - (m * z + c)
    gd_checkpoints[0] = float(np.mean(e_0 ** 2))

    for update in range(1, 501):
        e = val - (m * z + c)
        # Analytical gradients of MSE cost
        dj_dm = -(2.0 / n) * float(np.sum(z * e))
        dj_dc = -(2.0 / n) * float(np.sum(e))

        m -= eta * dj_dm
        c -= eta * dj_dc

        if update in [1, 10, 100, 500]:
            e_check = val - (m * z + c)
            gd_checkpoints[update] = float(np.mean(e_check ** 2))

    # Convert back to unstandardised scale: y = (m / sd_x4) * X4 + (c - m * mean_x4 / sd_x4)
    gd_slope = m / sd_x4
    gd_intercept = c - m * (mean_x4 / sd_x4)
    gd_slope_gap = abs(gd_slope - reg_slope)

    results["gd_J0"] = gd_checkpoints[0]
    results["gd_J1"] = gd_checkpoints[1]
    results["gd_J10"] = gd_checkpoints[10]
    results["gd_J100"] = gd_checkpoints[100]
    results["gd_J500"] = gd_checkpoints[500]
    results["gd_slope"] = gd_slope
    results["gd_intercept"] = gd_intercept
    results["gd_slope_gap"] = gd_slope_gap

    # =========================================================================
    # Task 1.6: Check for a funnel, then apply the log remedy
    # =========================================================================
    residuals_raw = val - fitted_raw

    # Sort rows by raw fitted value ascending using stable sort for tie-breaking
    sort_order_raw = np.argsort(fitted_raw, kind="stable")
    lower_raw_resids = residuals_raw[sort_order_raw[:400]]
    upper_raw_resids = residuals_raw[sort_order_raw[400:]]

    resid_sd_lower_raw = float(np.std(lower_raw_resids, ddof=1))
    resid_sd_upper_raw = float(np.std(upper_raw_resids, ddof=1))
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    # Log-log model: ln(value) on ln(X4)
    y_log = np.log(val)
    x_log = np.log(x4)
    mean_x_log = float(np.mean(x_log))
    mean_y_log = float(np.mean(y_log))

    cov_log = float(np.sum((x_log - mean_x_log) * (y_log - mean_y_log)) / (n - 1))
    var_log = float(np.sum((x_log - mean_x_log) ** 2) / (n - 1))

    loglog_slope = cov_log / var_log
    loglog_intercept = mean_y_log - loglog_slope * mean_x_log
    fitted_log = loglog_intercept + loglog_slope * x_log

    residuals_log = y_log - fitted_log
    log_rss = float(np.sum(residuals_log ** 2))
    log_tss = float(np.sum((y_log - mean_y_log) ** 2))
    loglog_r2 = 1.0 - (log_rss / log_tss)
    loglog_rmse = math.sqrt(log_rss / n)

    # Split rows by log model fitted values ascending
    sort_order_log = np.argsort(fitted_log, kind="stable")
    lower_log_resids = residuals_log[sort_order_log[:400]]
    upper_log_resids = residuals_log[sort_order_log[400:]]

    resid_sd_lower_log = float(np.std(lower_log_resids, ddof=1))
    resid_sd_upper_log = float(np.std(upper_log_resids, ddof=1))
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    results["resid_sd_lower_raw"] = resid_sd_lower_raw
    results["resid_sd_upper_raw"] = resid_sd_upper_raw
    results["resid_sd_ratio_raw"] = resid_sd_ratio_raw
    results["loglog_intercept"] = loglog_intercept
    results["loglog_slope"] = loglog_slope
    results["loglog_r2"] = loglog_r2
    results["loglog_rmse"] = loglog_rmse
    results["resid_sd_lower_log"] = resid_sd_lower_log
    results["resid_sd_upper_log"] = resid_sd_upper_log
    results["resid_sd_ratio_log"] = resid_sd_ratio_log

    # =========================================================================
    # Task 1.7: Multiple regression by the normal equation
    # =========================================================================
    # Exact predictor sequence:
    # 0: constant 1s, 1: ln(X4), 2: X1, 3: X2, 4: X3, 5: X5, 6: X6, 7: X7
    ones_col = np.ones((n, 1))
    ln_x4_col = np.log(X_features[:, 3:4])
    x1_col = X_features[:, 0:1]
    x2_col = X_features[:, 1:2]
    x3_col = X_features[:, 2:3]
    x5_col = X_features[:, 4:5]
    x6_col = X_features[:, 5:6]
    x7_col = X_features[:, 6:7]

    X_design = np.hstack([ones_col, ln_x4_col, x1_col, x2_col, x3_col, x5_col, x6_col, x7_col])
    Y_target = y_log

    # Solve normal equations: (X^T X) beta = X^T Y
    beta_hat = np.linalg.solve(X_design.T @ X_design, X_design.T @ Y_target)

    for i in range(8):
        results[f"mr_beta_{i}"] = float(beta_hat[i])

    fitted_mr = X_design @ beta_hat
    mr_rss = float(np.sum((Y_target - fitted_mr) ** 2))
    mr_tss = float(np.sum((Y_target - mean_y_log) ** 2))
    mr_r2 = 1.0 - (mr_rss / mr_tss)

    # Adjusted R2 for multiple regression (p = 7 non-constant predictors)
    p_mr = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1)

    # Adjusted R2 for bivariate log-log model (p = 1 predictor)
    p_log = 1
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - p_log - 1)

    results["mr_r2"] = mr_r2
    results["mr_adj_r2"] = mr_adj_r2
    results["loglog_adj_r2"] = loglog_adj_r2

    # =========================================================================
    # Write outputs/part1_results.txt
    # =========================================================================
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "part1_results.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        for key, val in results.items():
            f.write(f"{key} = {format_value(val)}\n")

    print(f"Successfully generated {output_path} with {len(results)} keys.")
    return results


if __name__ == "__main__":
    run_part1()

