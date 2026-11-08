Domain: E-commerce

# Flipkart E-commerce Decision Intelligence

Capstone Project for IIT Mandi × Masai Minor in Artificial Intelligence & Data Science.

## How to run

To run this project from a clean clone, execute the following commands in order:

```bash
# 1. Install pinned dependencies
pip install -r requirements.txt

# 2. Generate the dataset (creates data/records.csv and prints verification hash)
python generate_data.py

# 3. Optional: Run dataset integrity check
python verify_dataset.py

# 4. Run Part 1: Estimation, inference, and regression baseline
python part1.py

# 5. Run Part 2: Classification, threshold analysis, and KNN
python part2.py

# (Parts 3 to 4 will be run as implemented)
# python part3.py
# python part4.py
```

## Part 1 findings

- **R1.1 (Hypothesis Testing & Practical Size):** The null hypothesis $H_0: \mu = 8.0$ (representing the INR 800 planning assumption) was **rejected** (`reject_h0 = yes`). The computed test statistic is `z_stat = 2.8930`, which exceeds the two-tailed critical value of $1.96$ at $\alpha = 0.05$. In domain terms, a Type I error would occur if Flipkart's category planning team falsely concludes that average order values have shifted away from INR 800 (triggering costly, unnecessary revisions to category margin targets, promotional minimums, and inventory allocations) when customer order values were actually still averaging INR 800. The observed difference is `mean_value - 8.0 = 8.9382 - 8.0 = 0.9382` column units, representing an increase of **INR 93.82 per order** (an 11.73% lift). At Flipkart's operational scale of hundreds of thousands of daily transactions, an extra ~INR 94 per basket is practically substantial: it expands aggregate gross merchandise value (GMV), enhances unit-level fulfillment margins, and demonstrates that the baseline planning assumption meaningfully underestimates current order baskets.

- **R1.2 (Sample Sizing for Planning):** Sizing a future study to estimate the mean order value within a margin of error $E = 0.5$ column units (INR 50) at 95% confidence requires `n_required = 1293` records, calculated from the sample standard deviation `sd_value = 9.1723`. Compared to the 800 records currently available, our dataset is undersized for that degree of precision (yielding an actual margin of error of $\pm 0.6356$ units or $\pm \text{INR } 63.56$). For the planning team, this means that before making high-stakes budgetary or category decisions requiring $\pm \text{INR } 50$ precision, they must either pool an additional 493 customer transactions or accept a wider uncertainty band ($\pm \text{INR } 64$) when relying on the current sample.

- **R1.3 (Convergence of Gradient Descent):** Batch gradient descent converged exactly to the analytical least-squares regression line, yielding `gd_slope_gap = 0.0000` (far below the 0.001 threshold) with final `gd_slope = 0.6701` and `gd_intercept = -3.6598`. Gradient descent reached the closed-form line because the mean squared error loss surface of linear regression is strictly convex with a single global optimum, and standardizing $X_4$ ensured well-conditioned, spherical contours that permitted the fixed learning rate $\eta = 0.1$ to progress directly to the analytical minimum without oscillation.

- **R1.4 (Residual Funnel & Log Transformation):** The raw linear regression model displays a severe funnel effect (heteroskedasticity), with `resid_sd_ratio_raw = 3.7525`. The residual standard deviation in the upper half of fitted values (`resid_sd_upper_raw = 8.6891`) is nearly 3.75 times higher than in the lower half (`resid_sd_lower_raw = 2.3155`), violating the constant-variance assumption of ordinary least squares. In contrast, the log-log regression model $\ln(\text{value})$ on $\ln(X_4)$ yields `resid_sd_ratio_log = 1.0396`, completely removing the funnel effect and restoring homoskedastic error dispersion across the full range of predicted order values.

- **R1.5 (Multiple Regression Predictors & Model Comparison):** In the multiple regression of $\ln(\text{value})$ on eight design columns, the predictors whose coefficients have an absolute value strictly above $0.05$ (excluding the intercept `mr_beta_0 = -1.8613`) are:
  1. $\ln(X_4)$ (`avg_order_value_inr_hundreds` in log space): `mr_beta_1 = 1.1576`
  2. $X_2$ (`complaint_count_90d`): `mr_beta_3 = 0.0940`
  3. $X_3$ (`discount_order_share`): `mr_beta_4 = 0.1067`
  4. $X_6$ (`cod_flag`): `mr_beta_6 = 0.1518`
  *(The remaining predictors had $|\hat{\beta}_j| \le 0.05$: $X_1$ tenure with $-0.0074$, $X_5$ loyalty with $0.0028$, and $X_7$ plus member flag with $0.0172$).*
  Comparing the multiple regression adjusted $R^2$ of `mr_adj_r2 = 0.7403` against the bivariate log-log adjusted $R^2$ of `loglog_adj_r2 = 0.6561` reveals an increase of **0.0842 (8.42 percentage points)**. Because adjusted $R^2$ penalizes additional degrees of freedom, this substantial improvement confirms that the additional predictors genuinely earn their place in explaining customer order values.

## Part 2 findings

- **R2.1 (Learning-Rate Sensitivity):** Ordering the three Task 2.3 training losses after 200 epochs from lowest to highest yields:
  $$\text{loss\_epoch\_200\_eta\_1.0 } (0.3703) < \text{loss\_epoch\_200\_eta\_0.1 } (0.3776) < \text{loss\_epoch\_200\_eta\_0.01 } (0.5091)$$
  This demonstrates that an overly small learning rate ($\eta = 0.01$) progresses sluggishly and remains underfitted far from the minimum after 200 iterations, whereas on standardized features with a smooth convex loss surface, a large learning rate ($\eta = 1.0$) rapidly descends the gradient directly to the optimal cost basin without instability or divergence.

- **R2.2 (Predictor Weights & Odds Ratio Interpretation):** The three predictors with the largest absolute weights in the logistic regression model are:
  1. $X_3$ (`discount_order_share`): `weight_X3 = 0.9013` ($|w| = 0.9013$), `odds_ratio_X3 = 2.4629`
  2. $X_1$ (`customer_tenure_months`): `weight_X1 = -0.5554` ($|w| = 0.5554$), `odds_ratio_X1 = 0.5739`
  3. $X_2$ (`complaint_count_90d`): `weight_X2 = 0.5505` ($|w| = 0.5505$), `odds_ratio_X2 = 1.7341`
  *Domain interpretation of the largest odds ratio ($X_3$):* Holding all other variables constant, each one standard deviation increase in a customer's discount order share (~22.8 percentage points) multiplies the odds of churn by **2.4629** (a ~146% increase in churn odds), identifying deal-sensitive discount seekers as the highest-risk segment for defection once promotional incentives subside. Conversely, higher customer tenure protects against churn (`odds_ratio_X1 = 0.5739`, reducing odds by ~42.6%), while complaints significantly accelerate churn (`odds_ratio_X2 = 1.7341`, increasing odds by ~73.4%).

- **R2.3 (Accuracy Trap & Threshold Adjustment):** A naive baseline that blindly predicts zero churn for all customers achieves `blind_accuracy = 0.7875` (78.75%), which sits remarkably close to the default logistic regression accuracy `tau_0.5_accuracy = 0.8042` (80.42%). Accuracy alone is deeply misleading because active customers dominate the base (~78.75%); a model that simply ignores all churners scores nearly 79% accuracy while delivering zero business utility, and the $\tau = 0.5$ model misses more than half of all churners (`recall = 0.4706`, leaving 27 of 51 churners undetected). Lowering the decision threshold to $\tau = 0.3$ substantially elevates recall from **47.06% to 74.51%** (catching 38 of 51 churners), while precision adjusts moderately from **54.55% to 46.34%** as false alarms increase from 20 to 44.

- **R2.4 (Logistic Regression vs. KNN Benchmark):** Comparing the logistic regression model at $\tau = 0.3$ against the cross-validated $K$-Nearest Neighbours classifier ($K = 5$):
  - **Logistic Regression ($\tau = 0.3$):** `f1 = 0.5714`, `recall = 0.7451`
  - **KNN ($K = 5$):** `knn_f1 = 0.3696`, `knn_recall = 0.3333`
  The logistic model vastly surpasses KNN, delivering more than double the recall (74.51% vs 33.33%) and a markedly superior F1 score (0.5714 vs 0.3696). Because standard Euclidean distance weights all seven dimensions equally without recognizing directional feature relationships, majority voting in KNN breaks down under class imbalance and misses two-thirds of at-risk customers.

- **R2.5 (Financial Cost Optimization & Deployment Decision):** We recommend deploying the **logistic regression model calibrated to the financially optimal threshold $\tau = 0.15$**. On the unseen test cohort, this operating point yields `cost_test_at_tau = 6050` INR, saving nearly 50% compared to the standard threshold cost `cost_test_at_0.5 = 11800` INR and slashing losses by over 70% compared to flagging nobody `cost_test_flag_nobody = 20400` INR. The alert threshold was selected strictly on training data to prevent optimistic leakage and guarantee an honest, unbiased estimation of marketing ROI on future customer behavior.

## Part 3 findings

*To be completed in Phase 4 (Part 3).*

## Part 4 findings

*To be completed in Phase 5 (Part 4).*

## Model-choice memo

*To be completed in Phase 5 (Part 4).*

## Limitations

*To be completed upon concluding the evaluation.*

