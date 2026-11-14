"""Check that data/records.csv matches the brief: 800 rows, 10 columns,
170 churners and MD5 c6f249ba693f1ebeed318d6d5e18de23.

Exits with status 1 if any check fails.
"""

import csv
import hashlib
import os
import sys

DATA_FILE = os.path.join("data", "records.csv")
EXPECTED_ROWS = 800
EXPECTED_COLS = 10
EXPECTED_POSITIVES = 170
EXPECTED_MD5 = "c6f249ba693f1ebeed318d6d5e18de23"


def verify():
    if not os.path.exists(DATA_FILE):
        print(f"[FAIL] Dataset file does not exist at '{DATA_FILE}'.")
        return False

    with open(DATA_FILE, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    header = reader[0]
    data_rows = reader[1:]

    actual_rows = len(data_rows)
    actual_cols = len(header)
    churn_idx = header.index("churned")
    actual_positives = sum(1 for r in data_rows if r[churn_idx] == "1")

    with open(DATA_FILE, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()

    all_passed = True

    print("=" * 60)
    print("DATASET VERIFICATION REPORT")
    print("=" * 60)

    c1 = actual_rows == EXPECTED_ROWS
    print(f"Check 1 - Data Rows Count:       {actual_rows} (Expected: {EXPECTED_ROWS}) -> {'PASS' if c1 else 'FAIL'}")
    all_passed = all_passed and c1

    c2 = actual_cols == EXPECTED_COLS
    print(f"Check 2 - Columns Count:         {actual_cols} (Expected: {EXPECTED_COLS}) -> {'PASS' if c2 else 'FAIL'}")
    all_passed = all_passed and c2

    c3 = actual_positives == EXPECTED_POSITIVES
    print(f"Check 3 - Positive Churn Count:  {actual_positives} (Expected: {EXPECTED_POSITIVES}) -> {'PASS' if c3 else 'FAIL'}")
    all_passed = all_passed and c3

    # The MD5 is taken over raw bytes, so any line-ending change would fail here.
    c4 = actual_md5 == EXPECTED_MD5
    print(f"Check 4 - File MD5 Checksum:     {actual_md5}")
    print(f"                          Target: {EXPECTED_MD5} -> {'PASS' if c4 else 'FAIL'}")
    all_passed = all_passed and c4

    print("=" * 60)
    if all_passed:
        print("RESULT: ALL 4 VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    else:
        print("RESULT: VERIFICATION FAILED.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)

