"""Enhance an existing synthetic UPI dataset with realistic contextual variation.

The input is never overwritten. Exactly one enhanced CSV is written.
The original is_fraud labels and transaction IDs are preserved.

Example (Windows):
    python enhance_upi_dataset.py --input "E:\\UPI_Fraud_Detection\\data\\upi_transactions_synthetic.csv"

Example:
    python enhance_upi_dataset_final.py --input "E:/UPI_Fraud_Detection/data/upi_transactions_synthetic.csv" \
        --output "E:/UPI_Fraud_Detection/data/upi_transactions_synthetic_enhanced_final.csv"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
REQUIRED_COLUMNS = {
    "transaction_id", "timestamp", "user_id", "payee_id", "merchant_category",
    "amount_inr", "transaction_latitude", "transaction_longitude",
    "distance_from_home_km", "historical_avg_amount", "balance_before_inr", "is_fraud",
}

# Festival dates used for the dataset's 2025 date range.
FESTIVALS = {
    pd.Timestamp("2025-04-30").date(): "Akshaya Tritiya",
    pd.Timestamp("2025-06-07").date(): "Eid al-Adha",
    pd.Timestamp("2025-08-09").date(): "Raksha Bandhan",
    pd.Timestamp("2025-08-15").date(): "Independence Day",
    pd.Timestamp("2025-08-27").date(): "Ganesh Chaturthi",
    pd.Timestamp("2025-09-05").date(): "Onam",
}

# Synthetic campaign windows. These add context; they do not assign fraud labels.
SALE_WINDOWS = [
    ("2025-04-25", "2025-05-02", "Akshaya Tritiya Sale"),
    ("2025-06-05", "2025-06-09", "Eid Shopping Offer"),
    ("2025-07-05", "2025-07-09", "Mid-Year Sale"),
    ("2025-08-07", "2025-08-11", "Raksha Bandhan Sale"),
    ("2025-08-25", "2025-08-29", "Ganesh Chaturthi Sale"),
    ("2025-09-03", "2025-09-07", "Onam Sale"),
]


def make_date_context(dates: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return festival flags/names and sale flags/names based on transaction dates."""
    festival_name = pd.Series("No festival", index=dates.index, dtype="object")
    date_only = dates.dt.date
    for event_date, name in FESTIVALS.items():
        # Include the day before and after as a simple shopping/activity window.
        mask = date_only.map(
            lambda value: value is not pd.NaT and value is not None
            and abs((value - event_date).days) <= 1
        )
        festival_name.loc[mask & festival_name.eq("No festival")] = name

    sale_name = pd.Series("No sale campaign", index=dates.index, dtype="object")
    for start, end, name in SALE_WINDOWS:
        mask = dates.dt.normalize().between(pd.Timestamp(start), pd.Timestamp(end))
        sale_name.loc[mask & sale_name.eq("No sale campaign")] = name

    is_festival = festival_name.ne("No festival").astype("int8")
    is_sale = sale_name.ne("No sale campaign").astype("int8")
    return is_festival, festival_name, is_sale, sale_name


def apply_disruption_windows(timestamps: pd.Series, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create a small number of clustered, reproducible power/network incidents."""
    n = len(timestamps)
    power = np.zeros(n, dtype=np.int8)
    network = np.zeros(n, dtype=np.int8)

    first_day = timestamps.min().normalize()
    last_day = timestamps.max().normalize()
    number_of_days = (last_day - first_day).days + 1
    if number_of_days <= 0:
        return power, network

    # 4 power incidents and 8 network incidents across the observation period.
    incident_types = [("power", False), ("power", False), ("power", True), ("power", True)] + [("network", True)] * 8
    chosen_offsets = rng.choice(number_of_days, size=len(incident_types), replace=False)
    ts_array = timestamps.to_numpy(dtype="datetime64[ns]")

    for (kind, also_network), day_offset in zip(incident_types, chosen_offsets):
        day = first_day + pd.Timedelta(days=int(day_offset))
        hour = int(rng.integers(6, 24))
        minute = int(rng.integers(0, 60))
        start = day + pd.Timedelta(hours=hour, minutes=minute)
        duration = int(rng.integers(35, 111))
        end = start + pd.Timedelta(minutes=duration)
        mask = (ts_array >= np.datetime64(start)) & (ts_array <= np.datetime64(end))
        if kind == "power":
            power[mask] = 1
            # Two simulated power incidents also affect network service; two do not.
            if also_network:
                network[mask] = 1
        else:
            network[mask] = 1

    return power, network


def enhance_dataset(input_path: Path, output_path: Path, seed: int = SEED) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}\n"
            "Pass the correct location with --input. The source CSV will not be changed."
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from input path; the original file is protected.")

    df = pd.read_csv(input_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"The input is missing required columns: {missing}")
    if "festival_name" in df.columns or "is_traveling" in df.columns:
        raise ValueError(
            "The input already appears to be enhanced. Select the original master CSV as --input "
            "to avoid applying enhancements twice."
        )

    # Keep IDs and original label prevalence for final integrity checks.
    original_ids = df["transaction_id"].copy()
    original_fraud_count = int(pd.to_numeric(df["is_fraud"], errors="raise").sum())
    if not set(pd.to_numeric(df["is_fraud"], errors="raise").unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only binary 0/1 labels.")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="raise").astype("int8")
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Found {int(timestamps.isna().sum())} invalid timestamps; fix these before enhancing.")

    rng = np.random.default_rng(seed)
    date_only = timestamps.dt.normalize()
    weekday = timestamps.dt.dayofweek
    day_of_month = timestamps.dt.day

    # 1) Calendar context: festival windows, sale periods, and payday periods.
    is_festival, festival_name, is_sale, sale_name = make_date_context(timestamps)
    is_payday = ((day_of_month <= 5) | (day_of_month >= 28)).astype("int8")

    # 1b) Explicitly simulate a modest festival-linked fraud pattern.
    # Keep the overall fraud count fixed: move 50 labels from low-risk fraud
    # rows outside festival windows to higher-risk-looking legitimate rows in
    # festival windows. The label swap count is configurable and documented.
    # It does NOT mean festivals cause fraud in real life.
    festival_mask = is_festival.to_numpy().astype(bool)
    labels = df["is_fraud"].to_numpy(dtype=np.int8).copy()
    if festival_mask.sum() == 0:
        raise ValueError("No rows fall in the defined festival windows; cannot create festival scenario.")

    amount_ratio = pd.to_numeric(df["amount_to_avg_ratio"], errors="coerce").fillna(1).to_numpy()
    distance = pd.to_numeric(df["distance_from_home_km"], errors="coerce").fillna(0).to_numpy()
    new_device = pd.to_numeric(df["is_new_device"], errors="coerce").fillna(0).to_numpy()
    new_payee = pd.to_numeric(df["is_new_payee"], errors="coerce").fillna(0).to_numpy()
    night = pd.to_numeric(df["is_night"], errors="coerce").fillna(0).to_numpy()
    collect = df["transaction_type"].astype(str).str.lower().eq("collect_request").to_numpy()
    # Risk score is used only to select plausible synthetic rows for relabeling.
    risk_score = (1.1 * new_device + 0.7 * new_payee + 0.65 * (amount_ratio >= 2.0)
                  + 0.55 * (distance >= 50) + 0.35 * night + 0.65 * collect)
    swap_count = min(50, int((labels == 1).sum()), int((labels == 0).sum()))
    festival_negative = np.flatnonzero(festival_mask & (labels == 0))
    nonfestival_positive = np.flatnonzero((~festival_mask) & (labels == 1))
    if len(festival_negative) < swap_count or len(nonfestival_positive) < swap_count:
        raise ValueError("Not enough eligible rows to create the controlled festival scenario.")
    # Choose highest-risk festival negatives and lowest-risk nonfestival positives.
    festival_order = festival_negative[np.argsort(risk_score[festival_negative])[::-1]]
    nonfestival_order = nonfestival_positive[np.argsort(risk_score[nonfestival_positive])]
    # Randomize within equal-risk groups reproducibly to avoid row-order bias.
    selected_festival = festival_order[:swap_count]
    selected_nonfestival = nonfestival_order[:swap_count]
    labels[selected_festival] = 1
    labels[selected_nonfestival] = 0
    df["is_fraud"] = labels
    print(f"Synthetic festival scenario: reassigned {swap_count} labels; total fraud count preserved.")

    df["is_festival_period"] = is_festival
    df["festival_name"] = festival_name
    df["is_sale_day"] = is_sale
    df["sale_campaign"] = sale_name
    df["is_payday_period"] = is_payday

    # Offer exposure varies by date and ordinary background chance, never by fraud label.
    offer_probability = (
        0.025 + 0.32 * is_festival.to_numpy() + 0.28 * is_sale.to_numpy()
        + 0.055 * is_payday.to_numpy()
    )
    offer_probability = np.clip(offer_probability, 0.025, 0.75)
    offer_exposed = (rng.random(len(df)) < offer_probability).astype("int8")
    offer_type = np.full(len(df), "No offer", dtype=object)
    has_festival = is_festival.to_numpy().astype(bool) & offer_exposed.astype(bool)
    has_sale = is_sale.to_numpy().astype(bool) & offer_exposed.astype(bool) & ~has_festival
    has_payday = is_payday.to_numpy().astype(bool) & offer_exposed.astype(bool) & ~has_festival & ~has_sale
    other_offer = offer_exposed.astype(bool) & ~has_festival & ~has_sale & ~has_payday
    offer_type[has_festival] = "Festival Offer"
    offer_type[has_sale] = "Sale Discount"
    offer_type[has_payday] = "Payday Cashback"
    offer_type[other_offer] = "Cashback/Reward"
    df["is_offer_exposed"] = offer_exposed
    df["offer_type"] = offer_type

    # 2) Clustered power outages and network disruptions.
    power, network = apply_disruption_windows(timestamps, rng)
    df["is_power_outage"] = power
    df["is_network_disruption"] = network
    disruption_type = np.full(len(df), "No disruption", dtype=object)
    disruption_type[(power == 1) & (network == 1)] = "Power + Network"
    disruption_type[(power == 1) & (network == 0)] = "Power Outage"
    disruption_type[(power == 0) & (network == 1)] = "Network Disruption"
    df["disruption_type"] = disruption_type

    # Disruptions raise retry likelihood, but retries may also happen in normal conditions.
    retry_count = np.zeros(len(df), dtype=np.int8)
    disrupted = (power == 1) | (network == 1)
    disrupted_indices = np.flatnonzero(disrupted)
    if len(disrupted_indices):
        retry_prob = np.where(power[disrupted_indices] == 1, 0.62, 0.46)
        retry_occurs = rng.random(len(disrupted_indices)) < retry_prob
        retry_count[disrupted_indices] = np.where(
            retry_occurs,
            rng.integers(1, 4, size=len(disrupted_indices)),
            0,
        )
    normal_indices = np.flatnonzero(~disrupted)
    normal_retry = rng.random(len(normal_indices)) < 0.009
    retry_count[normal_indices] = np.where(normal_retry, rng.integers(1, 3, size=len(normal_indices)), 0)
    df["retry_count"] = retry_count
    df["payment_attempt_count"] = (retry_count + 1).astype("int8")
    df["is_payment_retry"] = (retry_count > 0).astype("int8")

    # 3) Travel and location variation, assigned independent of the fraud label.
    # Travel is more likely on weekends/selected event periods, but is possible any day.
    travel_probability = np.full(len(df), 0.027, dtype=float)
    travel_probability += np.where(weekday.to_numpy() >= 5, 0.025, 0.0)
    travel_probability += np.where(is_festival.to_numpy() == 1, 0.025, 0.0)
    traveling = (rng.random(len(df)) < np.clip(travel_probability, 0.0, 0.12)).astype("int8")
    df["is_traveling"] = traveling
    df["location_context"] = np.where(traveling == 1, "Travel/away from home", "Local/regular area")

    traveler_indices = np.flatnonzero(traveling == 1)
    if len(traveler_indices):
        # Use the original coordinate as a synthetic home-area anchor for these rows.
        # This is a simulation, not verified real-world home-location data.
        travel_distances = np.clip(rng.lognormal(mean=np.log(85), sigma=0.65, size=len(traveler_indices)), 25, 350)
        bearings = rng.uniform(0, 2 * np.pi, size=len(traveler_indices))
        old_lat = df.loc[traveler_indices, "transaction_latitude"].to_numpy(dtype=float)
        old_lon = df.loc[traveler_indices, "transaction_longitude"].to_numpy(dtype=float)
        lat_delta = (travel_distances * np.cos(bearings)) / 111.0
        cos_lat = np.cos(np.deg2rad(old_lat))
        cos_lat = np.maximum(np.abs(cos_lat), 0.25)
        lon_delta = (travel_distances * np.sin(bearings)) / (111.0 * cos_lat)
        new_lat = np.clip(old_lat + lat_delta, 6.0, 37.0)
        new_lon = np.clip(old_lon + lon_delta, 68.0, 98.0)
        df.loc[traveler_indices, "transaction_latitude"] = np.round(new_lat, 6)
        df.loc[traveler_indices, "transaction_longitude"] = np.round(new_lon, 6)
        df.loc[traveler_indices, "distance_from_home_km"] = np.round(travel_distances, 2)

    # 4) Normal customer variation/noise in amount. This is applied to both classes.
    # Most changes are modest; sale/offer uplifts affect selected consumer-facing categories.
    base_amount = pd.to_numeric(df["amount_inr"], errors="raise").to_numpy(dtype=float)
    jitter = np.clip(rng.lognormal(mean=0.0, sigma=0.055, size=len(df)), 0.82, 1.22)
    amount_multiplier = jitter
    categories = df["merchant_category"].astype(str).str.lower().to_numpy()
    consumer_categories = np.isin(categories, ["shopping", "entertainment", "grocery", "food", "travel", "fuel"])
    campaign_context = ((is_festival.to_numpy() == 1) | (is_sale.to_numpy() == 1))
    eligible = campaign_context & consumer_categories
    eligible_indices = np.flatnonzero(eligible)
    uplift_applies = np.zeros(len(df), dtype=bool)
    if len(eligible_indices):
        uplift_applies[eligible_indices] = rng.random(len(eligible_indices)) < 0.52
    amount_multiplier[uplift_applies] *= rng.uniform(1.05, 1.40, size=int(uplift_applies.sum()))

    # A small sales-day uplift can also affect non-core categories, with lower probability.
    other_sale_eligible = (is_sale.to_numpy() == 1) & ~consumer_categories & ~uplift_applies
    other_indices = np.flatnonzero(other_sale_eligible)
    if len(other_indices):
        small_uplift = rng.random(len(other_indices)) < 0.12
        selected = other_indices[small_uplift]
        amount_multiplier[selected] *= rng.uniform(1.02, 1.12, size=len(selected))

    df["amount_inr"] = np.round(np.maximum(base_amount * amount_multiplier, 1.0), 2)
    hist_avg = pd.to_numeric(df["historical_avg_amount"], errors="coerce").to_numpy(dtype=float)
    bal_before = pd.to_numeric(df["balance_before_inr"], errors="coerce").to_numpy(dtype=float)
    safe_hist = np.where(np.isfinite(hist_avg) & (hist_avg > 0), hist_avg, np.nan)
    safe_bal = np.where(np.isfinite(bal_before) & (bal_before > 0), bal_before, np.nan)
    df["amount_to_avg_ratio"] = np.round(df["amount_inr"].to_numpy(dtype=float) / safe_hist, 4)
    df["balance_drain_ratio"] = np.round(df["amount_inr"].to_numpy(dtype=float) / safe_bal, 4)

    # Preserve timestamp format, IDs, row count, and total fraud prevalence.
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if len(df) != len(original_ids):
        raise AssertionError("Row count changed unexpectedly.")
    if not df["transaction_id"].equals(original_ids):
        raise AssertionError("Transaction IDs changed unexpectedly.")
    if int(df["is_fraud"].sum()) != original_fraud_count:
        raise AssertionError("Overall fraud count changed unexpectedly.")

    # Output only the one enhanced master CSV; do not create helper files.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Read-back verification: confirms the saved file is usable and labels are unchanged.
    check = pd.read_csv(output_path, low_memory=False)
    assert len(check) == len(df)
    assert check["transaction_id"].is_unique == df["transaction_id"].is_unique
    assert int(check["is_fraud"].sum()) == original_fraud_count
    added_columns = ["festival_name", "sale_campaign", "offer_type", "disruption_type", "location_context"]
    assert not check[added_columns].isna().any().any(), "New context columns must not contain missing values."
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Add realistic context/noise to a synthetic UPI CSV.")
    parser.add_argument(
        "--input", type=Path,
        default=Path(r"E:/UPI_Fraud_Detection/data/upi_transactions_synthetic.csv"),
        help="Path to the original master CSV (never overwritten).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path. Defaults to upi_transactions_synthetic_enhanced.csv beside the input.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducible output.")
    args = parser.parse_args()
    output = args.output or args.input.with_name("upi_transactions_synthetic_enhanced.csv")
    enhanced = enhance_dataset(args.input, output, seed=args.seed)

    print("\nENHANCEMENT COMPLETED")
    print(f"Input (left unchanged): {args.input}")
    print(f"Enhanced CSV:           {output}")
    print(f"Rows: {len(enhanced):,}")
    print(f"Columns: {len(enhanced.columns)} (added {len(enhanced.columns) - 28} context columns, assuming the 28-column source)")
    print(f"Fraud labels: {enhanced['is_fraud'].value_counts().sort_index().to_dict()}")
    festival_stats = enhanced.groupby("is_festival_period")["is_fraud"].agg(["sum", "count", "mean"])
    print("\nFraud rate by festival period (synthetic simulation):")
    print(festival_stats.to_string())
    print("\nIMPORTANT: 50 labels were deliberately reassigned to simulate a modest festival-linked fraud pattern.")
    print("This is synthetic scenario design, not evidence that real festivals cause more fraud.")
    for column in ["is_festival_period", "is_sale_day", "is_offer_exposed", "is_power_outage", "is_network_disruption", "is_traveling", "is_payment_retry"]:
        print(f"{column}: {int(enhanced[column].sum()):,}")


if __name__ == "__main__":
    main()
