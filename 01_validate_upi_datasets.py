"""
Phase 1 — Dataset Validation
Reads original and enhanced CSV files without modifying them.
Update paths in CONFIGURATION before running.
"""
from pathlib import Path
import pandas as pd

# CONFIGURATION — update these paths if needed
ORIGINAL_CSV = Path(r"E:\UPI_Fraud_Detection\data\upi_transactions_synthetic.csv")
ENHANCED_CSV = Path(r"E:\UPI_Fraud_Detection\data\upi_transactions_synthetic_enhanced_final.csv")
TARGET = "is_fraud"
TRANSACTION_ID = "transaction_id"


def load_csv(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path}\n"
            "Edit the path at the top of this script and run again."
        )
    print(f"\nLoading {label}: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    return df


def basic_report(df, label):
    print("\n" + "=" * 68)
    print(f"{label.upper()} — BASIC DATA QUALITY")
    print("=" * 68)
    print("Shape:", df.shape)
    print("Missing cells:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Duplicate column names:", int(df.columns.duplicated().sum()))

    if TRANSACTION_ID in df:
        print("Missing transaction IDs:", int(df[TRANSACTION_ID].isna().sum()))
        print("Duplicate transaction IDs:", int(df[TRANSACTION_ID].duplicated().sum()))
    else:
        print(f"WARNING: '{TRANSACTION_ID}' column is missing.")

    if TARGET not in df:
        print(f"WARNING: target '{TARGET}' is missing.")
    else:
        print("\nTarget counts:")
        print(df[TARGET].value_counts(dropna=False).to_string())
        valid = df[TARGET].dropna()
        if len(valid):
            print("\nTarget percentages:")
            print((valid.value_counts(normalize=True) * 100).round(4).to_string())
            print(f"Fraud rate (target == 1): {(valid.eq(1).mean() * 100):.4f}%")

    print("\nData types:")
    print(df.dtypes.astype(str).to_string())


def compare_datasets(original, enhanced):
    print("\n" + "=" * 68)
    print("ORIGINAL VS ENHANCED")
    print("=" * 68)
    print("Original shape:", original.shape)
    print("Enhanced shape:", enhanced.shape)
    print("Added columns:", sorted(set(enhanced.columns) - set(original.columns)))
    print("Removed columns:", sorted(set(original.columns) - set(enhanced.columns)))

    if TRANSACTION_ID not in original or TRANSACTION_ID not in enhanced:
        print("Cannot compare rows: transaction_id missing in one or both files.")
        return
    if (original[TRANSACTION_ID].isna().any() or enhanced[TRANSACTION_ID].isna().any()
            or original[TRANSACTION_ID].duplicated().any()
            or enhanced[TRANSACTION_ID].duplicated().any()):
        print("Cannot safely compare rows: missing or duplicate transaction IDs found.")
        return

    o = original.set_index(TRANSACTION_ID, drop=False)
    e = enhanced.set_index(TRANSACTION_ID, drop=False)
    shared = o.index.intersection(e.index)
    print("IDs only in original:", len(o.index.difference(e.index)))
    print("IDs only in enhanced:", len(e.index.difference(o.index)))
    print("Shared transaction IDs:", len(shared))

    if TARGET in o and TARGET in e:
        old = o.loc[shared, TARGET]
        new = e.loc[shared, TARGET]
        changed = ~(old.eq(new) | (old.isna() & new.isna()))
        print("Target labels changed:", int(changed.sum()))
        if changed.any():
            changes = pd.crosstab(
                old[changed], new[changed],
                rownames=["Original target"], colnames=["Enhanced target"],
                dropna=False
            )
            print("Label transition table:")
            print(changes.to_string())
            print("Example changed IDs:", list(shared[changed][:20]))
        else:
            print("No target-label changes found.")

        festival_col = next(
            (c for c in ["is_festival_period", "festival_period", "is_festival"]
             if c in e.columns), None
        )
        print("\nFestival fraud-rate comparison:")
        if festival_col:
            summary = e.groupby(festival_col, dropna=False)[TARGET].agg(
                transactions="size",
                fraud_cases=lambda s: int(s.eq(1).sum()),
                fraud_rate_percent=lambda s: float(s.eq(1).mean() * 100)
            )
            print("Festival column:", festival_col)
            print(summary.round(4).to_string())
        else:
            print("No recognized festival flag found. Inspect the column list.")

    print("\nChanged values in shared original columns (excluding target and ID):")
    changes_found = []
    for col in original.columns:
        if col not in enhanced.columns or col in [TARGET, TRANSACTION_ID]:
            continue
        a, b = o.loc[shared, col], e.loc[shared, col]
        equal = a.eq(b) | (a.isna() & b.isna())
        count = int((~equal).sum())
        if count:
            changes_found.append((col, count))
    if changes_found:
        for col, count in changes_found:
            print(f"  {col}: {count:,} rows differ")
    else:
        print("No differences found in shared original features (excluding target/ID).")


def leakage_screen(df):
    print("\n" + "=" * 68)
    print("PRELIMINARY LEAKAGE REVIEW")
    print("=" * 68)
    print("Columns needing manual review:")
    terms = ("fraud", "label", "target", "scenario", "risk_score")
    flagged = [c for c in df.columns if any(t in c.lower() for t in terms)]
    print(flagged if flagged else "No obvious column-name matches.")
    print("\nLikely identifiers to exclude from model features:")
    ids = [c for c in df.columns if any(t in c.lower()
           for t in ("transaction_id", "user_id", "device_id", "payee_id"))]
    print(ids if ids else "No obvious identifier names found.")
    print("\nThis is only a screening list; review feature meaning before training.")


def main():
    original = load_csv(ORIGINAL_CSV, "original dataset")
    enhanced = load_csv(ENHANCED_CSV, "enhanced dataset")
    basic_report(original, "Original dataset")
    basic_report(enhanced, "Enhanced dataset")
    compare_datasets(original, enhanced)
    leakage_screen(enhanced)
    print("\nValidation finished. No files were changed or overwritten.")


if __name__ == "__main__":
    main()
