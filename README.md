Domain: E-commerce

# Flipkart E-commerce Decision Intelligence

Capstone project for the IIT Mandi × Masai Minor in Artificial Intelligence & Data Science.

All records come from the provided synthetic generator (`generate_data.py`). They are not real Flipkart data, so the findings below show how the methods work on this exercise. They do not describe Flipkart's actual customers or performance.

## How to run

**Prerequisites:** Python 3.12 or newer (`numpy==2.5.4` requires Python 3.12+; tested on Python 3.12.10 with NumPy 2.5.4).

Run these commands in order from the root of a clean clone:

```bash
# 1. Install the pinned dependency
pip install -r requirements.txt

# 2. Generate the dataset (writes data/records.csv and prints its MD5)
python generate_data.py

# 3. Optional: check row count, column count, churners and MD5
python verify_dataset.py

# 4. Part 1: estimation, inference and regression baseline
python part1.py

# 5. Part 2: logistic regression, threshold analysis and KNN
python part2.py

# 6. Part 3: K-means, hierarchical clustering and PCA
python part3.py

# 7. Part 4: tree splits, voting, AdaBoost and network sizing
python part4.py
```

Each script writes its results to `outputs/`.

## Part 1 findings

- **R1.1 (Hypothesis test and practical size):** We reject $H_0: \mu = 8.0$ (the INR 800 planning assumption): `z_stat = 2.8930` is above the two-tailed critical value 1.96 at $\alpha = 0.05$, so `reject_h0 = yes`. A Type I error here would mean concluding that the average order value has moved away from INR 800 when it is in fact still INR 800, which would lead the category team to revise its plans for no reason. The gap is `mean_value - 8.0 = 8.9382 - 8.0 = 0.9382` column units, about **INR 94 per order** (11.73% above the assumption). We think a gap of this size matters for planning, because it applies to every order in the category plan. Since the data is synthetic, this illustrates the test rather than a real Flipkart figure.

- **R1.2 (Sample size for planning):** To estimate the mean within $E = 0.5$ column units (INR 50) at 95% confidence, we need `n_required = 1293` records (using `sd_value = 9.1723`), compared with the **800 records** available. The current sample is too small for that precision: with 800 rows, the 95% margin of error is about $\pm 0.6356$ units ($\pm$ INR 64). The planning team should either collect about 493 more records or accept the wider $\pm$ INR 64 band when using this estimate.

- **R1.3 (Gradient descent convergence):** `gd_slope_gap = 0.0000` (well below the 0.001 limit). Gradient descent finished at `gd_slope = 0.6701` and `gd_intercept = -3.6598`, the same line as the closed form. It reached that line because the mean-squared-error cost of linear regression is convex with a single minimum, and after standardising $X_4$ the fixed step size $\eta = 0.1$ moves steadily towards it (the cost is already flat between `gd_J100 = 40.3698` and `gd_J500 = 40.3698`).

- **R1.4 (Residual funnel and log transform):** The raw model shows a clear funnel. `resid_sd_ratio_raw = 3.7525`: the residual SD in the upper half of fitted values (`resid_sd_upper_raw = 8.6891`) is about 3.75 times that of the lower half (`resid_sd_lower_raw = 2.3155`), which breaks the constant-variance assumption of ordinary least squares. In the log-log model of $\ln(\text{value})$ on $\ln(X_4)$, the ratio drops to `resid_sd_ratio_log = 1.0396` (`resid_sd_upper_log = 0.5369` vs `resid_sd_lower_log = 0.5165`), so by this median-split check the log transform removes the funnel.

- **R1.5 (Multiple regression):** In the regression of $\ln(\text{value})$ on the eight design columns, the predictors with $|\hat{\beta}_j| > 0.05$ (excluding the intercept `mr_beta_0 = -1.8613`) are:
  1. $\ln(X_4)$ (log of `avg_order_value_inr_hundreds`): `mr_beta_1 = 1.1576`
  2. $X_2$ (`complaint_count_90d`): `mr_beta_3 = 0.0940`
  3. $X_3$ (`discount_order_share`): `mr_beta_4 = 0.1067`
  4. $X_6$ (`cod_flag`): `mr_beta_6 = 0.1518`

  The other predictors fall below the cut-off: $X_1$ tenure ($-0.0074$), $X_5$ loyalty ($0.0028$) and $X_7$ Plus membership ($0.0172$). The predictors are in different units, so the 0.05 rule is applied mechanically and is not a ranking of importance. Adjusted $R^2$ rises from `loglog_adj_r2 = 0.6561` to `mr_adj_r2 = 0.7403`, a gain of **0.0842**. Adjusted $R^2$ already penalises extra predictors, so this gain shows that the additional predictors do earn their place.

## Part 2 findings

- **R2.1 (Learning rate):** After 200 updates the training losses, from lowest to highest, are:
  `loss_epoch_200_eta_1.0 = 0.3703` < `loss_epoch_200_eta_0.1 = 0.3776` < `loss_epoch_200_eta_0.01 = 0.5091`.
  A learning rate that is too small ($\eta = 0.01$) is still far from the minimum after 200 updates, while on this standardised, convex problem the large rate ($\eta = 1.0$) did not diverge and reached the lowest loss fastest.

- **R2.2 (Largest weights and odds ratio):** The three predictors with the largest absolute weights are:
  1. $X_3$ (`discount_order_share`): `weight_X3 = 0.9013`, `odds_ratio_X3 = 2.4629`
  2. $X_1$ (`customer_tenure_months`): `weight_X1 = -0.5554`, `odds_ratio_X1 = 0.5739`
  3. $X_2$ (`complaint_count_90d`): `weight_X2 = 0.5505`, `odds_ratio_X2 = 1.7341`

  Interpretation of the largest odds ratio: holding the other predictors fixed, a one-SD increase in a customer's share of discounted orders (about 22.8 percentage points in the training rows) multiplies their odds of churning by **2.4629**. In the same way, longer tenure lowers the odds (×0.5739 per SD) and more complaints raise them (×1.7341 per SD). These are associations in synthetic data, not proven causes.

- **R2.3 (Accuracy and threshold):** `blind_accuracy = 0.7875`, compared with `tau_0.5_accuracy = 0.8042`. Accuracy is a weak score here because 78.75% of test customers did not churn: a "model" that never flags anyone already scores 78.75% while catching no churners. The $\tau = 0.5$ model is only 1.7 points better and still misses 27 of the 51 churners (`tau_0.5_recall = 0.4706`). Lowering $\tau$ from 0.5 to 0.3:
  - **Recall** rose from **0.4706 to 0.7451** (churners caught: 24 → 38 of 51).
  - **Precision** fell from **0.5455 to 0.4634** (false alarms: 20 → 44).
  - F1 rose from 0.5053 to `tau_0.3_f1 = 0.5714`.

  $\tau = 0.3$ is the classification benchmark used in Tasks 2.5 and 4.2. It is separate from the cost-based threshold $\tau = 0.15$ chosen in Task 2.8.

- **R2.4 (Logistic regression vs KNN):**
  - Logistic regression ($\tau = 0.3$): `tau_0.3_f1 = 0.5714`, `tau_0.3_recall = 0.7451`
  - KNN (`knn_best_k = 5`): `knn_f1 = 0.3696`, `knn_recall = 0.3333`

  Logistic regression is better on both measures. It catches 38 of 51 test churners, compared with 17 for KNN. One likely reason (not tested separately) is that KNN's distance weights all seven standardised features equally, while logistic regression learns a separate weight for each feature.

- **R2.5 (Deployment choice and cost):** We would deploy **logistic regression at $\tau = 0.15$** (`cost_tau = 0.1500`), the threshold with the lowest cost on the 560 training rows (`cost_train_at_tau = 12150`). On the 240 test rows:
  - `cost_test_at_tau = 6050` INR
  - `cost_test_at_0.5 = 11800` INR (about 48.7% more)
  - `cost_test_flag_nobody = 20400` INR (400 × 51; about 70.3% more)

  For context, at $\tau = 0.15$ the test confusion counts are TP = 45, FP = 73, TN = 116 and FN = 6 (recall 0.8824, precision 0.3814). These counts are not a required results key. They were recomputed with the same Part 2 code and agree with the stored cost: 50 × 73 + 400 × 6 = 6050. The threshold was chosen on training rows so that the test rows stay unseen and the test cost is an honest estimate of performance on new customers.

## Part 3 findings

- **R3.1 (WCSS drops and elbow):** The four successive WCSS drops are:
  - WCSS(2) − WCSS(3) = 2262.6753 − 1602.8471 = **659.8282**
  - WCSS(3) − WCSS(4) = 1602.8471 − 1494.7732 = **108.0739**
  - WCSS(4) − WCSS(5) = 1494.7732 − 1323.7157 = **171.0575**
  - WCSS(5) − WCSS(6) = 1323.7157 − 1278.1691 = **45.5466**

  The elbow rule gives **`elbow_k = 3`**. K = 3 is the smallest K in {3, 4, 5} whose next drop is less than half of the previous one: 108.0739 < 0.5 × 659.8282 = 329.9141. The drops do not shrink steadily (171.0575 for K = 4 → 5 is larger than 108.0739 for K = 3 → 4). This is because the pinned starting rows give one reproducible solution, not necessarily the global optimum for each K. The drop from K = 2 to K = 3 is by far the largest, which supports K = 3.

- **R3.2 (Segment profiles and actions):** From `part3_segments.csv` (segments ordered by mean discount share $X_3$):
  1. **Segment A (322 customers): Loyal low-discount regulars**
     - *Profile:* longest tenure (`mean_X1 = 47.5028` months), fewest complaints (`mean_X2 = 1.9503`), lowest discount share (`mean_X3 = 0.2250`), highest loyalty score (`mean_X5 = 62.1584`), smaller baskets (`mean_X4 = 12.5301`, `mean_value = 3.5976`, about INR 360) and the lowest churn rate (`outcome_rate = 0.0186`).
     - *Action:* use re-order reminders and subscription-style refills instead of deep discounts, because this group already uses discounts the least.
  2. **Segment B (255 customers): High-value, high-complaint shoppers**
     - *Profile:* middle tenure (`mean_X1 = 28.6071` months), the most complaints (`mean_X2 = 6.0706`), moderate discount share (`mean_X3 = 0.4534`), the highest historical and current order values (`mean_X4 = 28.7192`, `mean_value = 17.1248`, about INR 1,712), loyalty score `mean_X5 = 55.0082` and churn rate `outcome_rate = 0.2353`.
     - *Action:* route these customers to priority customer care and faster complaint resolution, since they combine the highest spend with the most complaints.
  3. **Segment C (223 customers): Bargain-focused deal seekers**
     - *Profile:* shortest tenure (`mean_X1 = 14.3892` months), the second-highest complaint count (`mean_X2 = 4.2287`), the highest discount share (`mean_X3 = 0.7118`), the lowest loyalty score (`mean_X5 = 38.5215`), `mean_X4 = 16.5099` and `mean_value = 7.2883` (about INR 729).
     - *Highest churn:* **Segment C has the highest churn rate, `outcome_rate = 0.4664` (46.64%).**
     - *Action:* test minimum-spend coupons (for example, INR 150 off orders above INR 1,200) that reward a repeat purchase without offering unconditional deep discounts.

  The segment profiles show which features go together. They do not show why these customers churn.

- **R3.3 (Single vs complete linkage):**
  - **Single linkage (`single_clusters_at_cut = 4`):** the largest gap is between merge 2 (1.4000) and merge 3 (2.8000). The cut gives **4 clusters**: `{E1}`, `{E4}`, `{E5}` and `{E2, E3, E6}`.
  - **Complete linkage (`complete_clusters_at_cut = 2`):** the largest gap is between merge 4 (2.9000) and merge 5 (7.9000). The cut gives **2 clusters**: `{E1, E4, E5}` and `{E2, E3, E6}`.
  - *Why they disagree:* Single linkage measures the closest pair of members, so the two groups join at only 3.4 through the single E5–E6 link; all merge heights stay between 1.3 and 3.4, and the largest gap falls early, between the tight E2/E3/E6 merges and the E4–E5 merge. Complete linkage measures the farthest pair, so the final merge costs 7.9 (the E1–E2 distance), which creates one dominant gap and a clean two-group cut.

  (Average linkage also cuts at 2 clusters: `average_clusters_at_cut = 2`.)

- **R3.4 (PCA):**
  - *Mean vector:* after standardising $X_1$–$X_5$ over all 800 rows, the mean vector is zero, $[0, 0, 0, 0, 0]$, up to floating-point error (largest absolute entry about $2.16 \times 10^{-15}$). This is Step 1 of Task 3.5.
  - *What drives PC1:* PC1 explains `explained_ratio_1 = 0.5227` of the variance. It loads positively on tenure (`pc1_loading_X1 = 0.5317`) and loyalty score (`pc1_loading_X5 = 0.4418`) and negatively on discount share (`pc1_loading_X3 = -0.5187`) and complaints (`pc1_loading_X2 = -0.4100`). So PC1 contrasts long-tenure, loyal customers with discount-heavy, complaint-heavy customers.
  - *Variance kept:* **3 components** (`components_for_80pct = 3`) are needed to keep at least 80% of the variance (`cumulative_ratio_3 = 0.8616`).
  - *Segments on PC1:* `mean_pc1_segment_A = 1.7773`, `mean_pc1_segment_B = -0.7831`, `mean_pc1_segment_C = -1.6709`. The segments are clearly separated along PC1: A sits on the loyal, long-tenure side and C on the discount-heavy side, with B in between. This is the same order as their churn rates, from lowest (A) to highest (C).

## Part 4 findings

- **R4.1 (Best split and stopping rule):** **S4 ($X_1 \le 24$ months tenure)** is the best split by both information gain (`info_gain = 0.1252`) and Gini gain (`gini_gain = 0.0606`), because it creates the most unequal children: 45.74% churn among the 188 newer customers (86 churners) against 8.87% among the 372 longer-tenured customers (33 churners). A tree grown until every leaf is pure keeps splitting until its leaves describe a handful of individual training customers, so it learns noise that does not carry over to new data. Setting a maximum depth (or a minimum number of rows per leaf) stops the tree from growing that far.

- **R4.2 (Model to ship and voting):** On the 240 test rows at $\tau = 0.3$ (`part4_voting.csv`), we would ship **standalone logistic regression**: `f1 = 0.5714`, `recall = 0.7451` (38 of 51 churners). Combining models did not help. Hard voting fell to `f1 = 0.5344`, `recall = 0.6863`, and soft voting reached `f1 = 0.5507` with the same `recall = 0.7451`. The stump alone had slightly higher recall (0.7647) but a lower F1 (0.5532) and more false alarms (51 vs 44). A vote only helps when the members are reasonably accurate and make different mistakes. Here KNN is much weaker than the other two (`f1 = 0.3696`, `recall = 0.3333`), which likely explains why the combinations fell short. For the live alert, the cost analysis in R2.5 supports running this model at $\tau = 0.15$.

- **R4.3 (AdaBoost):** After round 1 normalisation, the two misclassified points (3 and 7) each have weight 0.2500, for a **total of `0.5000`**, half of all the weight. Round 2 trusts $h_2$ more (`round2_alpha = 0.8047` vs `round1_alpha = 0.5493`) because $h_2$ gets right the two points that now carry the most weight (3 and 7), so its weighted error is lower (`round2_error = 0.1667` vs `round1_error = 0.2500`) and $\alpha$ grows as the error falls.

- **R4.4 (Network sizing and deep-learning concepts):**
  - **Model A, Option 2 (CNN), layer by layer:**

    | Layer | Operation | Output shape (H × W × depth) | Parameters | Calculation |
    |---|---|---|---|---|
    | Input | 28 × 28 grayscale image | 28 × 28 × 1 | 0 | – |
    | `cnn_conv1` | Conv 5 × 5, 6 filters, stride 1, padding 2, then ReLU | 28 × 28 × 6 | 156 | (5 × 5 × 1 + 1) × 6 |
    | `cnn_pool1` | Max-pool 2 × 2, stride 2 | 14 × 14 × 6 | 0 | – |
    | `cnn_conv2` | Conv 5 × 5, 16 filters, stride 1, padding 0, then ReLU | 10 × 10 × 16 | 2,416 | (5 × 5 × 6 + 1) × 16 |
    | `cnn_pool2` | Max-pool 2 × 2, stride 2 | 5 × 5 × 16 | 0 | – |
    | `cnn_flatten` | Flatten | 400 | 0 | 5 × 5 × 16 |
    | `cnn_dense1` | Fully connected, 32 units, ReLU | 32 | 12,832 | 400 × 32 + 32 |
    | `cnn_dense2` | Fully connected output, 10 classes | 10 | 330 | 32 × 10 + 10 |
    | **Total** | `cnn_total` | – | **15,734** | 156 + 2,416 + 12,832 + 330 |

    For comparison, the fully connected Option 1 has `mlp_total = 13002` parameters. On a 128 × 128 × 3 input, a 1,024-neuron dense layer needs `fc_layer_params = 50332672` parameters, compared with `conv_layer_params = 1792` for 64 filters of size 3 × 3, because a convolution filter reuses the same weights at every position.
  - **Model B (RNN, review text into 16 topics):** the weight matrices have these shapes:
    - $W_{xh}$: 128 × 64 (8,192 parameters)
    - $W_{hh}$: 128 × 128 (16,384 parameters)
    - $W_{hy}$: 16 × 128 (2,048 parameters)
    - With biases $b_h$ (128) and $b_y$ (16), `rnn_total = 26768`.

    `rnn_total` does not grow for longer reviews because the same $W_{xh}$, $W_{hh}$ and $W_{hy}$ are reused at every time step. A longer sequence means more steps, not more parameters.
  - **Activations and loss for 10 classes:** **ReLU** in the hidden layers, and a **softmax** output with **categorical cross-entropy** loss, so the 10 outputs form a probability distribution over the classes.
  - **Vanishing gradients and skip connections:** In a very deep plain network, backpropagation multiplies the gradient by one layer's weights and activation derivatives after another, and when these factors are mostly below 1 the gradient shrinks towards zero before it reaches the early layers, so those layers barely update. A skip connection computes $x + F(x)$, which gives the gradient a direct identity path back to earlier layers, so it no longer has to pass through every multiplication.

## Model-choice memo

To: Head of Customer Retention, Flipkart\
From: Customer Analytics (capstone project)\
Subject: Which churn model to put into production

We recommend the standalone logistic regression model. On 240 held-out test customers at the benchmark threshold of 0.30, it had the best F1 of the five models compared (0.5714) and a recall of 0.7451, which means it flagged 38 of the 51 customers who churned (outputs/part4_voting.csv). Combining models did not help: hard voting fell to an F1 of 0.5344 and a recall of 0.6863. The KNN model alone reached only an F1 of 0.3696 and a recall of 0.3333.

For the live alert we recommend a lower threshold of 0.15. It was chosen on training customers only, by minimising INR 50 per unnecessary coupon plus INR 400 per missed churner (outputs/part2_results.txt). At this threshold the logistic model costs INR 6,050 on the test customers, compared with INR 11,800 at a threshold of 0.50 and INR 20,400 if nobody is flagged. That is about 49% and 70% less. The trade-off is that more coupons go to customers who would have stayed anyway.

The main risk is that the evidence is limited. The data is synthetic, and every test figure comes from a single 70/30 split with only 51 churners. The model should be checked on real historical data, across more than one period, before rollout.

Neither Model A (product-image sorting) nor Model B (review-topic classification) is worth building before this structured model is in production, because neither directly predicts churn and both would need new data and infrastructure.

## Limitations

1. **Synthetic data:** The records come from a simple parametric generator (`generate_data.py`). Real Flipkart data would contain missing values, seasonal effects, catalogue changes and other patterns that this data does not have, so none of these numbers should be read as real business results.
2. **One split and few churners:** All test results come from a single fixed 70/30 split with only 51 churners in the test set. A few customers changing class would move recall, precision and cost noticeably, and we did not measure this uncertainty (for example with repeated splits or confidence intervals).
3. **Fixed cost assumptions:** The threshold analysis assumes every false alarm costs INR 50 and every missed churner costs INR 400. In practice these costs differ between customers, for example between a low-spend Segment A customer and a high-spend Segment B customer, so the best threshold could differ by segment.
