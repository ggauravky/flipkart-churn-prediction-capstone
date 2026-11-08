"""Data loading helper for Flipkart E-commerce Decision Intelligence.

This module provides a lightweight, reusable utility to load 'data/records.csv'
into clean NumPy arrays while preserving exact dataset row order.
"""

import csv
import os
import numpy as np

# Feature names corresponding to predictors X1 through X7
FEATURE_NAMES = [
    "customer_tenure_months",       # X1: Customer tenure in months
    "complaint_count_90d",          # X2: Complaints count in last 90 days
    "discount_order_share",         # X3: Share of discounted orders (0 to 1)
    "avg_order_value_inr_hundreds", # X4: Historical average order value (INR hundreds)
    "loyalty_score",                # X5: Loyalty program score (0 to 100)
    "cod_flag",                     # X6: Cash on Delivery preference (0 or 1)
    "plus_member_flag"              # X7: Flipkart Plus loyalty tier membership (0 or 1)
]

TARGET_NAME = "churned"                      # Outcome: 1 if churned, 0 otherwise
VALUE_NAME = "order_value_inr_hundreds"      # Value: current order value (INR hundreds)


def load_dataset(filepath=os.path.join("data", "records.csv")):
    """Load records.csv into structured NumPy arrays and dictionaries.

    Preserves the original row order without shuffling.
    Returns:
        dict containing:
            - 'record_ids': list of string IDs (e.g. ['E0001', ...])
            - 'X': 2D float64 NumPy array of shape (N, 7) for features X1 to X7
            - 'y': 1D int64 NumPy array of shape (N,) for target 'churned'
            - 'value': 1D float64 NumPy array of shape (N,) for 'order_value_inr_hundreds'
            - individual predictor arrays for direct access (e.g. data['X4'])
            - 'feature_names': list of predictor column names
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}. Run generate_data.py first.")

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    record_ids = [r[0] for r in rows]

    # Convert numeric columns to NumPy arrays
    # Columns:
    # 0: record_id
    # 1..7: X1..X7
    # 8: churned (outcome)
    # 9: order_value_inr_hundreds (value)
    raw_data = np.array([[float(val) for val in r[1:]] for r in rows], dtype=np.float64)

    X = raw_data[:, 0:7]                     # Features X1 to X7
    y = raw_data[:, 7].astype(np.int64)      # Outcome / target (churned)
    value = raw_data[:, 8]                   # Continuous value column

    data = {
        "record_ids": record_ids,
        "X": X,
        "y": y,
        "value": value,
        "feature_names": FEATURE_NAMES,
        "header": header,
        # Direct convenience accessors for individual features
        "X1": X[:, 0],
        "X2": X[:, 1],
        "X3": X[:, 2],
        "X4": X[:, 3],
        "X5": X[:, 4],
        "X6": X[:, 5],
        "X7": X[:, 6],
        "customer_tenure_months": X[:, 0],
        "complaint_count_90d": X[:, 1],
        "discount_order_share": X[:, 2],
        "avg_order_value_inr_hundreds": X[:, 3],
        "loyalty_score": X[:, 4],
        "cod_flag": X[:, 5],
        "plus_member_flag": X[:, 6],
        "churned": y,
        "order_value_inr_hundreds": value
    }

    return data


if __name__ == "__main__":
    # Quick self-test of the data loader
    dataset = load_dataset()
    print(f"Loaded successfully!")
    print(f"Total records: {len(dataset['record_ids'])}")
    print(f"X shape: {dataset['X'].shape}")
    print(f"y shape: {dataset['y'].shape}, positives: {np.sum(dataset['y'])}")
    print(f"value shape: {dataset['value'].shape}, mean: {np.mean(dataset['value']):.4f}")

