
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Dataset paths
ORIGINAL_PATH = Path(
    r"E:\UPI_Fraud_Detection\data\upi_transactions_synthetic.csv"
)
ENHANCED_PATH = Path(
    r"E:\UPI_Fraud_Detection\data\upi_transactions_synthetic_enhanced_final.csv"
)

# 2. Load both datasets
original = pd.read_csv(ORIGINAL_PATH)
enhanced = pd.read_csv(ENHANCED_PATH)

print("=" * 60)
print("ENHANCED DATASET INVESTIGATION")
print("=" * 60)

# 3. Check transaction IDs
print("\n1. TRANSACTION ID CHECK")

print("Original rows:", len(original))
print("Enhanced rows:", len(enhanced))
print(
    "Same transaction ID sets:",
    set(original["transaction_id"]) == set(enhanced["transaction_id"])
)

# Align both datasets by transaction ID
original = original.set_index("transaction_id").sort_index()
enhanced = enhanced.set_index("transaction_id").sort_index()

shared_ids = original.index.intersection(enhanced.index)
original = original.loc[shared_ids]
enhanced = enhanced.loc[shared_ids]

# 4. Check changes in original features
print("\n2. CHANGES IN ORIGINAL FEATURES")

columns_to_check = [
    "amount_inr",
    "transaction_latitude",
    "transaction_longitude",
    "distance_from_home_km",
    "amount_to_avg_ratio",
    "balance_drain_ratio"
]

for column in columns_to_check:
    if column in original.columns and column in enhanced.columns:
        old_values = original[column]
        new_values = enhanced[column]

        changed = ~(
            old_values.eq(new_values)
            | (old_values.isna() & new_values.isna())
        )

        print(f"{column}: {int(changed.sum()):,} changed rows")

# 5. Check amount-to-average ratio
print("\n3. AMOUNT RATIO CONSISTENCY")

required = ["amount_inr", "historical_avg_amount", "amount_to_avg_ratio"]

if all(c in enhanced.columns for c in required):
    amount = enhanced["amount_inr"]
    average = enhanced["historical_avg_amount"]
    recorded_ratio = enhanced["amount_to_avg_ratio"]

    # Avoid division by zero
    valid = average.notna() & average.ne(0) & recorded_ratio.notna()

    calculated_ratio = amount[valid] / average[valid]

    difference = (calculated_ratio - recorded_ratio[valid]).abs()
    inconsistent = difference > 0.02

    print("Rows checked:", int(valid.sum()))
    print("Potentially inconsistent ratios:", int(inconsistent.sum()))
    print("Maximum absolute difference:", round(float(difference.max()), 4)
          if len(difference) else "N/A")
else:
    print("Required amount-ratio columns are missing.")

# 6. Inspect balance-drain ratio
print("\n4. BALANCE DRAIN RATIO")

if "balance_drain_ratio" in enhanced.columns:
    print(enhanced["balance_drain_ratio"].describe().round(4))
    print(
        "Values below 0:",
        int((enhanced["balance_drain_ratio"] < 0).sum())
    )
    print(
        "Values above 1:",
        int((enhanced["balance_drain_ratio"] > 1).sum())
    )
else:
    print("balance_drain_ratio column is missing.")

# 7. Inspect changed fraud labels
print("\n5. CHANGED FRAUD LABELS")

old_label = original["is_fraud"]
new_label = enhanced["is_fraud"]

labels_changed = old_label.ne(new_label)
changed_ids = enhanced.index[labels_changed]

print("Changed labels:", int(labels_changed.sum()))

if len(changed_ids):
    label_review_columns = [
        c for c in [
            "is_fraud",
            "is_festival_period",
            "festival_name",
            "is_sale_day",
            "sale_campaign",
            "is_offer_exposed",
            "offer_type",
            "is_power_outage",
            "is_network_disruption",
            "disruption_type",
            "is_traveling",
            "location_context"
        ]
        if c in enhanced.columns
    ]

    review = enhanced.loc[changed_ids, label_review_columns].copy()
    review.insert(0, "original_is_fraud", old_label.loc[changed_ids])

    print("\nFirst 20 changed transactions:")
    print(review.head(20).to_string())

    print("\nChanged-label counts by original and enhanced labels:")
    print(
        pd.crosstab(
            old_label.loc[changed_ids],
            new_label.loc[changed_ids],
            rownames=["Original label"],
            colnames=["Enhanced label"]
        )
    )

# 8. Check event features against fraud rates
print("\n6. EVENT FEATURES AND FRAUD RATES")

event_columns = [
    "is_festival_period",
    "is_sale_day",
    "is_payday_period",
    "is_offer_exposed",
    "is_power_outage",
    "is_network_disruption",
    "is_payment_retry",
    "is_traveling"
]

for column in event_columns:
    if column in enhanced.columns:
        summary = enhanced.groupby(column, dropna=False)["is_fraud"].agg(
            transactions="size",
            fraud_cases="sum",
            fraud_rate=lambda s: s.mean() * 100
        )

        print(f"\n{column}:")
        print(summary.round(4).to_string())

print("\n" + "=" * 60)
print("INVESTIGATION COMPLETE")
print("Neither CSV file was modified.")
print("=" * 60)
