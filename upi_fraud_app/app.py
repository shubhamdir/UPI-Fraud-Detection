from pathlib import Path
from datetime import datetime
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

# ============================================================
# UPI SecureCheck: one scrolling page, made for everyday users.
# Same model, same 37 features, same checks as before.
# New: big clickable tiles, plain-language help, an analysis
# animation, and automatic scrolling down to the result.
# ============================================================

st.set_page_config(
    page_title="UPI SecureCheck",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "xgboost_preprocessor.pkl"
FEATURES_PATH = MODEL_DIR / "xgboost_feature_columns.pkl"
THRESHOLD_PATH = MODEL_DIR / "xgboost_threshold.pkl"

# How long the "analyzing" animation stays on screen, in seconds.
ANALYSIS_SECONDS = 1.2

FEATURE_COLUMNS = [
    "payee_type", "merchant_category", "transaction_type", "amount_inr",
    "transaction_latitude", "transaction_longitude", "distance_from_home_km",
    "hour_of_day", "day_of_week", "is_weekend", "is_night", "is_new_device",
    "is_new_payee", "payee_account_age_days", "txns_last_1h", "txns_last_24h",
    "distinct_payees_last_24h", "minutes_since_last_txn", "historical_avg_amount",
    "amount_to_avg_ratio", "balance_before_inr", "balance_drain_ratio",
    "is_festival_period", "festival_name", "is_sale_day", "sale_campaign",
    "is_payday_period", "is_offer_exposed", "offer_type", "is_power_outage",
    "is_network_disruption", "disruption_type", "retry_count",
    "payment_attempt_count", "is_payment_retry", "is_traveling",
    "location_context",
]

CATEGORICAL_FIELDS = {
    "payee_type", "merchant_category", "transaction_type", "day_of_week",
    "festival_name", "sale_campaign", "offer_type", "disruption_type",
    "location_context",
}
BOOLEAN_FIELDS = {
    "is_weekend", "is_night", "is_new_device", "is_new_payee",
    "is_festival_period", "is_sale_day", "is_payday_period",
    "is_offer_exposed", "is_power_outage", "is_network_disruption",
    "is_payment_retry", "is_traveling",
}
FIELD_LABELS = {
    "payee_type": "Payee type",
    "merchant_category": "Merchant category",
    "transaction_type": "Transaction type",
    "amount_inr": "Transaction amount (₹)",
    "transaction_latitude": "Transaction latitude",
    "transaction_longitude": "Transaction longitude",
    "distance_from_home_km": "Distance from home (km)",
    "hour_of_day": "Hour of day (0–23)",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend?",
    "is_night": "Night-time?",
    "is_new_device": "New device?",
    "is_new_payee": "New payee?",
    "payee_account_age_days": "Payee account age (days)",
    "txns_last_1h": "Transactions in last hour",
    "txns_last_24h": "Transactions in last 24 hours",
    "distinct_payees_last_24h": "Different payees in last 24 hours",
    "minutes_since_last_txn": "Minutes since previous transaction",
    "historical_avg_amount": "Historical average amount (₹)",
    "amount_to_avg_ratio": "Amount / usual average ratio",
    "balance_before_inr": "Balance before transaction (₹)",
    "balance_drain_ratio": "Balance drain ratio",
    "is_festival_period": "Festival period?",
    "festival_name": "Festival name",
    "is_sale_day": "Sale day?",
    "sale_campaign": "Sale campaign",
    "is_payday_period": "Payday period?",
    "is_offer_exposed": "Offer shown?",
    "offer_type": "Offer type",
    "is_power_outage": "Power outage?",
    "is_network_disruption": "Network disruption?",
    "disruption_type": "Disruption type",
    "retry_count": "Retry count",
    "payment_attempt_count": "Payment attempt count",
    "is_payment_retry": "Payment retry?",
    "is_traveling": "User traveling?",
    "location_context": "Location context",
}

# Plain-language hints shown with a ❔ icon next to each box.
FIELD_HELP = {
    "payee_type": "Who is receiving the money: a person, a shop, or a business?",
    "merchant_category": "What kind of place or service is this payment for?",
    "transaction_type": "P2P = sending money to a person. P2M = paying a shop or merchant.",
    "amount_inr": "How much money is being sent, in rupees (₹).",
    "transaction_latitude": "Where the payment was made. Keep the default if you do not know.",
    "transaction_longitude": "Where the payment was made. Keep the default if you do not know.",
    "distance_from_home_km": "How far the payment place is from home, in kilometres.",
    "hour_of_day": "Time of the payment from 0 to 23. For example, 14 means 2 PM.",
    "day_of_week": "Which day of the week the payment was made.",
    "is_weekend": "Was the payment made on a Saturday or Sunday?",
    "is_night": "Was the payment made late at night?",
    "is_new_device": "Was it made from a phone or device this user has never used before?",
    "is_traveling": "Is the user away from home or travelling right now?",
    "location_context": "Where is the user right now: home, work, or somewhere else?",
    "is_new_payee": "Has this person or shop never been paid before?",
    "payee_account_age_days": "How many days old is the receiver's account?",
    "txns_last_1h": "How many payments were made in the last 1 hour?",
    "txns_last_24h": "How many payments were made in the last 24 hours?",
    "distinct_payees_last_24h": "How many different people or shops were paid in the last 24 hours?",
    "minutes_since_last_txn": "Minutes since the previous payment.",
    "historical_avg_amount": "The amount this user usually pays, in rupees (₹).",
    "amount_to_avg_ratio": "This payment divided by the usual amount. 1 is normal; 3 or more is very high.",
    "balance_before_inr": "Money in the account just before this payment, in rupees (₹).",
    "balance_drain_ratio": "Share of the balance used in this payment. 0.8 means 80%.",
    "is_festival_period": "Was this payment made during a festival season?",
    "festival_name": "Name of the festival, if any. Choose None if not applicable.",
    "is_sale_day": "Was it a big shopping sale day?",
    "sale_campaign": "Name of the sale campaign, if any. Choose None if not applicable.",
    "is_payday_period": "Was it close to salary day?",
    "is_offer_exposed": "Did the user see a cashback or discount offer?",
    "offer_type": "Type of offer the user saw, if any.",
    "is_power_outage": "Was there a power cut around the time of payment?",
    "is_network_disruption": "Was there an internet or network problem?",
    "disruption_type": "Type of network or power problem, if any.",
    "retry_count": "How many times this payment was retried.",
    "payment_attempt_count": "Total attempts made, including the first one.",
    "is_payment_retry": "Was this a retry of an earlier failed payment?",
}

# Starting values. Category names must match the training data exactly.
DEFAULTS = {
    "payee_type": "Individual", "merchant_category": "Other",
    "transaction_type": "Pay", "amount_inr": 500.0,
    "transaction_latitude": 19.0760, "transaction_longitude": 72.8777,
    "distance_from_home_km": 2.0, "hour_of_day": 14,
    "day_of_week": "Monday", "is_weekend": 0, "is_night": 0,
    "is_new_device": 0, "is_new_payee": 0, "payee_account_age_days": 365,
    "txns_last_1h": 1, "txns_last_24h": 3, "distinct_payees_last_24h": 2,
    "minutes_since_last_txn": 120.0, "historical_avg_amount": 600.0,
    "amount_to_avg_ratio": 0.83, "balance_before_inr": 10000.0,
    "balance_drain_ratio": 0.05, "is_festival_period": 0,
    "festival_name": "No festival", "is_sale_day": 0, "sale_campaign": "No sale campaign",
    "is_payday_period": 0, "is_offer_exposed": 0, "offer_type": "No offer",
    "is_power_outage": 0, "is_network_disruption": 0,
    "disruption_type": "No disruption", "retry_count": 0, "payment_attempt_count": 1,
    "is_payment_retry": 0, "is_traveling": 0, "location_context": "Local/regular area",
}

# The model was trained with day_of_week as a number, not a name.
# ASSUMED: Monday = 0 ... Sunday = 6 (the pandas / Python convention).
# If your training data used a different code, change the numbers here.
DAY_CODES = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

# Example payment used by the "Load a sample payment" button.
# It is an illustration only; the result comes from the model.
SAMPLE_PAYMENT = {
    "payee_type": "Individual", "merchant_category": "Travel",
    "transaction_type": "Collect_Request", "amount_inr": 4999.0,
    "transaction_latitude": 19.0760, "transaction_longitude": 72.8777,
    "distance_from_home_km": 300.0, "hour_of_day": 2,
    "day_of_week": "Sunday", "is_weekend": 1, "is_night": 1,
    "is_new_device": 1, "is_new_payee": 1, "payee_account_age_days": 12,
    "txns_last_1h": 5, "txns_last_24h": 9, "distinct_payees_last_24h": 7,
    "minutes_since_last_txn": 3.0, "historical_avg_amount": 800.0,
    "amount_to_avg_ratio": 6.2, "balance_before_inr": 5200.0,
    "balance_drain_ratio": 0.96, "is_festival_period": 0,
    "festival_name": "No festival", "is_sale_day": 0, "sale_campaign": "No sale campaign",
    "is_payday_period": 0, "is_offer_exposed": 0, "offer_type": "No offer",
    "is_power_outage": 0, "is_network_disruption": 0,
    "disruption_type": "No disruption", "retry_count": 0, "payment_attempt_count": 1,
    "is_payment_retry": 0, "is_traveling": 1, "location_context": "Travel/away from home",
}

# Dropdown choices for the five category boxes, copied from the training
# notebook. The model's own list is used first; these are a backup.
TEXT_CHOICES = {
    "festival_name": ["Akshaya Tritiya", "Eid al-Adha", "Ganesh Chaturthi", "Independence Day", "No festival", "Onam", "Raksha Bandhan"],
    "sale_campaign": ["Akshaya Tritiya Sale", "Eid Shopping Offer", "Ganesh Chaturthi Sale", "Mid-Year Sale", "No sale campaign", "Onam Sale", "Raksha Bandhan Sale"],
    "offer_type": ["Cashback/Reward", "Festival Offer", "No offer", "Payday Cashback", "Sale Discount"],
    "disruption_type": ["Network Disruption", "No disruption", "Power + Network", "Power Outage"],
    "location_context": ["Local/regular area", "Travel/away from home"],
}

def widget_value(field, raw):
    """Turn a plain value into what its form box stores."""
    if field in BOOLEAN_FIELDS:
        return "Yes" if int(raw) == 1 else "No"
    if field == "day_of_week":
        return str(raw) if str(raw) in DAY_CODES else "Monday"
    if field in CATEGORICAL_FIELDS:
        options = MODEL_CATEGORIES.get(field, []) or TEXT_CHOICES.get(field, [])
        if options:
            return str(raw) if str(raw) in options else options[0]
    return raw

def set_form_values(source):
    for field in FEATURE_COLUMNS:
        st.session_state[f"manual_{field}"] = widget_value(field, source[field])

def load_sample_payment():
    set_form_values(SAMPLE_PAYMENT)

def clear_manual_form():
    set_form_values(DEFAULTS)

# Theme and layout styling: same colours, fonts and look as before.
# The tile and animation rules at the bottom are the only additions.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#f8f6ff; --muted:#c9c3e7; --purple:#a855f7; --violet:#7138e8; --navy:#090b2c; }
html { scroll-behavior:smooth; }
.stApp { background: radial-gradient(ellipse at 75% 0%, #24205d 0%, #101238 38%, #090b25 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(8,10,36,.92); }
[data-testid="stAppViewContainer"] > .main { background:transparent; }
.block-container { padding-top:1.2rem; padding-bottom:3rem; max-width:1600px; }
h1,h2,h3,h4,p,label,span,div { font-family:'DM Sans',sans-serif; }
h1,h2,h3,h4 { color:#fff !important; font-family:'Manrope',sans-serif; }
p, label, .stMarkdown, .stCaption { color:#e9e5ff; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101033 0%,#21134b 55%,#100d35 100%); border-right:1px solid #453080; }
[data-testid="stSidebar"] * { color:#f7f2ff !important; }
[data-testid="stSidebar"] .stButton button { text-align:left; background:transparent; border:1px solid #453080; }
.hero { background:linear-gradient(105deg,rgba(13,13,54,.98),rgba(42,20,93,.96) 60%,rgba(84,30,145,.92)); border:1px solid #6844c9; border-radius:22px; padding:26px 30px; box-shadow:0 0 28px rgba(132,68,255,.25), inset 0 0 35px rgba(116,57,230,.12); margin:6px 0 18px 0; }
.hero-kicker { color:#d7b7ff; font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:.78rem; }
.hero-title { font-family:'Manrope',sans-serif; font-size:clamp(2rem,3.3vw,3.15rem); line-height:1.08; font-weight:800; color:white; margin:.35rem 0; }
.hero-title span { color:#c47bff; }
.hero-sub { color:#e5ddff; font-size:1.05rem; max-width:760px; }
.hero-pills { display:flex; gap:9px; flex-wrap:wrap; margin-top:16px; }
.pill { border:1px solid #8553db; color:#f4eaff; background:rgba(112,65,190,.22); padding:7px 13px; border-radius:30px; font-size:.82rem; }
.section-card { background:linear-gradient(145deg,rgba(31,30,83,.96),rgba(19,19,57,.96)); border:1px solid #6849b7; border-radius:19px; padding:17px 19px; box-shadow:0 7px 24px rgba(0,0,0,.18),0 0 14px rgba(117,69,220,.12); margin-bottom:14px; }
.section-title { font-family:'Manrope',sans-serif; font-size:1.25rem; font-weight:800; color:#fff; margin-bottom:12px; }
.metric { border-radius:17px; padding:16px 17px; min-height:104px; border:1px solid rgba(255,255,255,.22); box-shadow:0 0 20px rgba(127,72,230,.16); }
.metric.blue { background:linear-gradient(125deg,#123b91,#174d9f 65%,#312b83); }
.metric.red { background:linear-gradient(125deg,#76234f,#a32b65 65%,#642d91); }
.metric.green { background:linear-gradient(125deg,#075b50,#087f65 65%,#145e81); }
.metric.purple { background:linear-gradient(125deg,#5422a2,#7a28c2 65%,#402b91); }
.metric-label { color:#f2eaff; font-size:.88rem; }
.metric-value { color:white; font-family:'Manrope',sans-serif; font-size:1.8rem; font-weight:800; line-height:1.3; }
.action-card { border-radius:18px; padding:19px; min-height:110px; border:1px solid; margin-bottom:10px; }
.action-green { background:linear-gradient(110deg,#064d43,#087d5b); border-color:#32efb0; box-shadow:0 0 18px rgba(21,235,163,.32); }
.action-purple { background:linear-gradient(110deg,#482087,#7829c8); border-color:#d18bff; box-shadow:0 0 18px rgba(180,91,255,.3); }
.result-box { background:linear-gradient(100deg,#fff0f5,#f5eaff); color:#1b1641; border-radius:16px; padding:18px; }
.result-box h3,.result-box p,.result-box strong { color:#24133d !important; }
.good-text { color:#087b4b !important; font-weight:800; }
.bad-text { color:#c51f3a !important; font-weight:800; }
div[data-testid="stForm"] { background:rgba(28,26,76,.9); border:1px solid #6044a6; border-radius:16px; padding:18px; }
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"] > div,.stFileUploader section { background:#171943 !important; color:white !important; border-color:#6d55ad !important; }
.stSelectbox [data-baseweb="select"] * { color:white !important; }
.stButton button,.stDownloadButton button { border-radius:11px; border:1px solid #9465ef; background:linear-gradient(100deg,#7138e8,#a33be8); color:white; font-weight:700; }
.stButton button:hover,.stDownloadButton button:hover { border-color:#d8b5ff; color:white; box-shadow:0 0 16px rgba(176,99,255,.35); }
div[data-testid="stExpander"] { background:rgba(30,27,78,.72); border:1px solid #56418f; border-radius:13px; }
[data-testid="stDataFrame"] { border:1px solid #57438e; border-radius:12px; }
hr { border-color:#49377c; }
.small-muted { color:#bdb5df; font-size:.83rem; }
a { color:#d8a9ff !important; }

/* ---- Added for easier use ---- */
[id] { scroll-margin-top:18px; }
.stButton button, .stDownloadButton button { font-size:1.02rem; padding:.55rem 1rem; }

/* Big clickable entry tiles (Manual Entry and Upload CSV) */
.st-key-tile_manual button, .st-key-tile_upload button {
  min-height:150px !important; width:100%; border-radius:18px !important;
  text-align:left !important; white-space:normal !important; padding:18px 20px !important;
  justify-content:flex-start !important; align-items:flex-start !important;
  transition:transform .15s ease, box-shadow .15s ease;
}
.st-key-tile_manual button { background:linear-gradient(110deg,#064d43,#087d5b) !important; border:1px solid #32efb0 !important; box-shadow:0 0 18px rgba(21,235,163,.32) !important; }
.st-key-tile_upload button { background:linear-gradient(110deg,#482087,#7829c8) !important; border:1px solid #d18bff !important; box-shadow:0 0 18px rgba(180,91,255,.3) !important; }
.st-key-tile_manual button:hover, .st-key-tile_upload button:hover { transform:translateY(-3px); color:#fff !important; }
.st-key-tile_manual button p, .st-key-tile_upload button p { color:#eafff6 !important; font-size:1rem !important; line-height:1.5 !important; text-align:left; }
.st-key-tile_upload button p { color:#f6eeff !important; }
.st-key-tile_manual button p:first-child, .st-key-tile_upload button p:first-child { font-family:'Manrope',sans-serif; font-size:1.45rem !important; font-weight:800 !important; color:#fff !important; }

/* Clickable stat tiles */
.st-key-metric_blue button, .st-key-metric_red button, .st-key-metric_green button, .st-key-metric_purple button {
  min-height:104px !important; width:100%; border-radius:17px !important; text-align:left !important;
  border:1px solid rgba(255,255,255,.22) !important; box-shadow:0 0 20px rgba(127,72,230,.16) !important;
  white-space:normal !important; align-items:flex-start !important; justify-content:flex-start !important;
  transition:transform .15s ease;
}
.st-key-metric_blue button:hover, .st-key-metric_red button:hover, .st-key-metric_green button:hover, .st-key-metric_purple button:hover { transform:translateY(-2px); }
.st-key-metric_blue button { background:linear-gradient(125deg,#123b91,#174d9f 65%,#312b83) !important; }
.st-key-metric_red button { background:linear-gradient(125deg,#76234f,#a32b65 65%,#642d91) !important; }
.st-key-metric_green button { background:linear-gradient(125deg,#075b50,#087f65 65%,#145e81) !important; }
.st-key-metric_purple button { background:linear-gradient(125deg,#5422a2,#7a28c2 65%,#402b91) !important; }
.st-key-metric_blue button p, .st-key-metric_red button p, .st-key-metric_green button p, .st-key-metric_purple button p { color:#f2eaff !important; font-size:.92rem !important; text-align:left; }
.st-key-metric_blue button p:last-child, .st-key-metric_red button p:last-child, .st-key-metric_green button p:last-child, .st-key-metric_purple button p:last-child { color:#fff !important; font-family:'Manrope',sans-serif; font-size:1.8rem !important; font-weight:800 !important; line-height:1.3 !important; }

/* How-it-works steps */
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:4px 0 18px 0; }
.step-card { background:linear-gradient(145deg,rgba(31,30,83,.96),rgba(19,19,57,.96)); border:1px solid #6849b7; border-radius:17px; padding:16px 18px; color:#e9e5ff; line-height:1.5; }
.step-card b { color:#fff; font-family:'Manrope',sans-serif; font-size:1.05rem; }
.step-num { width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:linear-gradient(100deg,#7138e8,#a33be8); color:#fff; font-weight:800; margin-bottom:8px; }

/* Red warning note for the 37-column CSV rule */
.red-note { background:rgba(197,31,58,.18); border:1px solid #ff4d6d; border-left:6px solid #ff3b5c; color:#ff5c75 !important; border-radius:14px; padding:14px 16px; font-weight:600; line-height:1.55; margin:10px 0 14px 0; }
.red-note b { color:#ff7a8f !important; }

/* Simple "analyzing" animation */
.analyzing { text-align:center; padding:26px 20px; margin:14px 0; background:linear-gradient(145deg,rgba(31,30,83,.96),rgba(19,19,57,.96)); border:1px solid #6849b7; border-radius:19px; box-shadow:0 0 20px rgba(117,69,220,.2); }
.analyzing .shield { font-size:3rem; display:inline-block; animation:su-pulse 1.4s ease-in-out infinite; }
.analyzing-title { font-family:'Manrope',sans-serif; font-size:1.2rem; font-weight:800; color:#fff; margin-top:6px; }
.analyzing .bar { height:8px; width:100%; max-width:420px; margin:14px auto 0 auto; background:#2a2766; border-radius:8px; overflow:hidden; }
.analyzing .bar span { display:block; height:100%; width:0; background:linear-gradient(90deg,#32efb0,#a33be8); border-radius:8px; animation:su-bar 1.2s ease-out forwards; }
@keyframes su-pulse { 0%,100% { transform:scale(1); opacity:.85; } 50% { transform:scale(1.08); opacity:1; } }
@keyframes su-bar { from { width:0; } to { width:100%; } }

@media (max-width: 900px) { .steps { grid-template-columns:1fr; } }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Model loading (same as before)
# ============================================================
@st.cache_resource
def load_artifacts():
    paths = [MODEL_PATH, PREPROCESSOR_PATH, FEATURES_PATH, THRESHOLD_PATH]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifact(s): " + ", ".join(missing))
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_columns = list(joblib.load(FEATURES_PATH))
    threshold = joblib.load(THRESHOLD_PATH)
    if hasattr(threshold, "item"):
        threshold = threshold.item()
    return model, preprocessor, feature_columns, float(threshold)

try:
    model, preprocessor, saved_features, FRAUD_THRESHOLD = load_artifacts()
    if saved_features != FEATURE_COLUMNS:
        st.error("The saved feature list does not match this app's expected feature names/order.")
        st.write("Saved features:", saved_features)
        st.write("App features:", FEATURE_COLUMNS)
        st.stop()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.info("Make sure the four model files (.pkl) are inside the 'models' folder next to app.py, then refresh this page.")
    st.stop()

def get_encoder_categories(prep):
    found = {}
    try:
        for _, transformer, columns in prep.transformers_:
            if transformer in ("drop", "passthrough"):
                continue
            candidates = [transformer]
            if hasattr(transformer, "get_params"):
                candidates.extend(transformer.get_params(deep=True).values())
            encoder = next((obj for obj in candidates if hasattr(obj, "categories_")), None)
            if encoder is not None:
                for col, vals in zip(columns, encoder.categories_):
                    found[str(col)] = [str(v) for v in vals]
    except Exception:
        pass
    return found

MODEL_CATEGORIES = get_encoder_categories(preprocessor)

def make_prediction(frame):
    frame = frame[FEATURE_COLUMNS].copy()
    transformed = preprocessor.transform(frame)
    probabilities = model.predict_proba(transformed)[:, 1]
    predictions = (probabilities >= FRAUD_THRESHOLD).astype(int)
    return probabilities, predictions

def risk_level(probability):
    if probability < 0.30:
        return "Low"
    if probability < 0.70:
        return "Medium"
    return "High"

def explain_indicators(row):
    """Simple input checks for context; these are not SHAP feature attributions."""
    checks = [
        ("is_new_device", "Transaction used a new device."),
        ("is_new_payee", "The payee is new."),
        ("is_night", "Transaction occurred at night."),
        ("is_traveling", "The user is marked as traveling."),
        ("is_payment_retry", "The payment was retried."),
        ("is_network_disruption", "A network disruption was recorded."),
        ("is_power_outage", "A power outage was recorded."),
    ]
    notes = [message for key, message in checks
             if key in row and str(row[key]).strip().lower() in ("1", "1.0", "true", "yes")]
    try:
        if float(row["amount_to_avg_ratio"]) >= 3:
            notes.append("The amount is much higher than the historical average.")
    except Exception:
        pass
    try:
        if float(row["balance_drain_ratio"]) >= 0.8:
            notes.append("The transaction uses a large share of the available balance.")
    except Exception:
        pass
    try:
        if float(row["txns_last_1h"]) >= 5:
            notes.append("Several transactions occurred within the last hour.")
    except Exception:
        pass
    return notes[:5] or ["No simple input indicator crossed the display checks."]

def result_table(input_frame, probabilities, predictions):
    out = input_frame.copy()
    out["fraud_risk_percentage"] = np.round(probabilities * 100, 2)
    out["risk_level"] = [risk_level(float(p)) for p in probabilities]
    out["prediction"] = np.where(predictions == 1, "Suspicious", "Likely Legitimate")
    out["decision_threshold"] = FRAUD_THRESHOLD
    out["checked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out

# ============================================================
# Session state and helpers
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "latest_results" not in st.session_state:
    st.session_state.latest_results = None
if "result_source" not in st.session_state:
    st.session_state.result_source = None   # "manual" or "csv"
if "scroll_to" not in st.session_state:
    st.session_state.scroll_to = None

def add_results_to_history(results, source):
    st.session_state.history.extend(results.to_dict(orient="records"))
    st.session_state.latest_results = results.copy()
    st.session_state.result_source = source

def request_scroll(anchor_id):
    """Ask the page to glide down to an anchor once this run finishes."""
    st.session_state.scroll_to = anchor_id

def analyzing_html(message):
    return (
        '<div class="analyzing">'
        '<div class="shield">🛡️</div>'
        f'<div class="analyzing-title">{message}</div>'
        '<div class="small-muted">Please wait a moment. The XGBoost model is checking the details.</div>'
        '<div class="bar"><span></span></div>'
        '</div>'
    )

def run_analysis(input_df, slot, message):
    """Show a simple analyzing animation while the model runs."""
    slot.markdown(analyzing_html(message), unsafe_allow_html=True)
    try:
        probabilities, predictions = make_prediction(input_df)
        time.sleep(ANALYSIS_SECONDS)
    finally:
        slot.empty()
    return probabilities, predictions

def render_results(source):
    """Show the latest result only inside the section that produced it."""
    latest = st.session_state.latest_results
    if latest is None or latest.empty or st.session_state.result_source != source:
        return
    st.markdown(f'<div id="result-{source}"></div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Your result")
    if len(latest) == 1:
        row = latest.iloc[0]
        suspicious = row["prediction"] == "Suspicious"
        risk_pct = float(row["fraud_risk_percentage"])
        a, b = st.columns([1.25, 1])
        with a:
            result_class = "bad-text" if suspicious else "good-text"
            status = "⚠ Suspicious Transaction" if suspicious else "✓ Likely Legitimate"
            plain = ("This payment looks unusual. Please check with the receiver before you pay."
                     if suspicious else
                     "This payment looks normal based on the details you entered.")
            st.markdown(f'<div class="result-box"><div style="font-weight:700;color:#5d4c78">TRANSACTION STATUS</div><h2 class="{result_class}" style="font-size:2rem;margin:.25rem 0">{status}</h2><h3 style="color:#29194b!important;font-size:2.1rem">{risk_pct:.2f}% <span style="font-size:1rem">Fraud Risk Score</span></h3><p>Risk Level: <b>{row["risk_level"]}</b></p><p>{plain}</p><p style="font-size:.83rem">Decision threshold: {FRAUD_THRESHOLD:.7f}. The model decision uses this saved threshold.</p></div>', unsafe_allow_html=True)
            st.markdown('<p class="small-muted">Risk guide: Low = below 30% · Medium = 30% to 70% · High = above 70%.</p>', unsafe_allow_html=True)
        with b:
            st.markdown('<div class="section-card"><div class="section-title">💡 Why Was It Flagged?</div>', unsafe_allow_html=True)
            for reason in explain_indicators(row):
                st.markdown(f"- {reason}")
            st.caption("These are simple input-based indicators, not SHAP explanations or proof of the model's internal reasoning.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        flagged_latest = int((latest["prediction"] == "Suspicious").sum())
        st.info(f"{len(latest):,} transactions analyzed; {flagged_latest:,} look suspicious. See the 'prediction' column below.")
        st.dataframe(latest[["prediction", "fraud_risk_percentage", "risk_level"] + (["amount_inr"] if "amount_inr" in latest else [])], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download these results", latest.to_csv(index=False).encode("utf-8"), "upi_securecheck_results.csv", "text/csv")
    if st.button("↑ Check another", key=f"again_{source}"):
        request_scroll("manual-section" if source == "manual" else "upload-section")

SCROLL_JS = """
<script>
(function () {
  var target = "__TARGET__";
  var tries = 0;
  function go() {
    try {
      var el = window.parent.document.getElementById(target);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
    } catch (e) {}
    tries += 1;
    if (tries < 40) { setTimeout(go, 150); }
  }
  setTimeout(go, 200);
})();
</script>
<!-- __STAMP__ -->
"""

def run_pending_scroll():
    """Glide the page down to the requested section (runs at the end of the page)."""
    target = st.session_state.get("scroll_to")
    if target:
        st.session_state.scroll_to = None
        components.html(
            SCROLL_JS.replace("__TARGET__", target).replace("__STAMP__", str(time.time())),
            height=0,
        )

# ============================================================
# Sidebar, header and quick numbers
# ============================================================
with st.sidebar:
    st.markdown("## 🛡️ UPI SecureCheck")
    st.caption("Smarter Detection. Safer Digital Payments.")
    st.markdown("---")
    st.markdown("**QUICK NAVIGATION**")
    st.markdown("[⌂  Home](#dashboard)")
    st.markdown("[✍️  Manual Entry](#manual-section)")
    st.markdown("[☁️  Upload CSV](#upload-section)")
    st.markdown("[▤  Fraud statistics](#fraud-statistics)")
    st.markdown("[▥  Model performance](#model-performance)")
    st.markdown("[◷  Transaction history](#transaction-history)")
    st.markdown("---")
    st.caption("Academic demonstration using synthetic transaction data.")
    st.caption("Not connected to a live UPI or banking network.")

st.markdown("""
<div id="dashboard" class="hero">
  <div class="hero-kicker">🛡️ Powered by XGBoost</div>
  <div class="hero-title">UPI Transaction <span>Fraud Detection</span></div>
  <div class="hero-sub">UPI SecureCheck keeps your digital payments safer. Type in a payment, or upload a file, and we will tell you whether it looks normal or suspicious. No technical knowledge needed.</div>
  <div class="hero-pills"><span class="pill">✦ AI Powered</span><span class="pill">⚡ Fast Analysis</span><span class="pill">🔒 Model-based Screening</span></div>
</div>
<div class="steps">
  <div class="step-card"><div class="step-num">1</div><b>Pick a tile</b><br>Choose Manual Entry for one payment, or Upload CSV for many.</div>
  <div class="step-card"><div class="step-num">2</div><b>Add the details</b><br>Fill in the boxes, or upload your file.</div>
  <div class="step-card"><div class="step-num">3</div><b>Read the result</b><br>See whether each payment looks normal or suspicious.</div>
</div>
""", unsafe_allow_html=True)

# Session metrics: never display invented transaction counts.
history_df = pd.DataFrame(st.session_state.history)
total_checked = len(history_df)
flagged_count = int((history_df["prediction"] == "Suspicious").sum()) if total_checked else 0
legit_count = total_checked - flagged_count
fraud_rate = (flagged_count / total_checked * 100) if total_checked else 0.0

# Entry tiles: big and clickable. Each one scrolls down to its section.
st.markdown('<div id="check-transaction"></div>', unsafe_allow_html=True)
st.markdown("## 👇 Start here: pick how you want to check")
st.caption("Click a big tile. The page will glide down to the right section for you.")
left, right = st.columns(2)
with left:
    if st.button("**✍️  Manual Entry**\n\nType in one transaction and check it. Best for a single payment.", key="tile_manual", use_container_width=True):
        request_scroll("manual-section")
with right:
    if st.button("**☁️  Upload Dataset (CSV)**\n\nUpload a file with many transactions and check them all together.", key="tile_upload", use_container_width=True):
        request_scroll("upload-section")

# Quick-number tiles: clickable, each one jumps to its details.
st.caption("Quick numbers for this session. Click any tile to jump to its details.")
m1, m2, m3, m4 = st.columns(4)
metric_tiles = [
    (m1, "metric_blue", "▰", f"{total_checked:,}", "Transactions Checked", "transaction-history"),
    (m2, "metric_red", "⚠", f"{flagged_count:,}", "Flagged as Suspicious", "transaction-history"),
    (m3, "metric_green", "✓", f"{legit_count:,}", "Likely Legitimate", "transaction-history"),
    (m4, "metric_purple", "%", f"{fraud_rate:.2f}%", "Fraud Rate in This Session", "fraud-statistics"),
]
for col, key, icon, value, label, target in metric_tiles:
    with col:
        if st.button(f"**{icon}  {label}**\n\n{value}", key=key, use_container_width=True):
            request_scroll(target)

# ============================================================
# Section 1: Manual Entry (all boxes on one scrolling page)
# ============================================================
st.markdown("---")
st.markdown('<div id="manual-section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card"><div class="section-title">✍️ Manual Entry: check one transaction</div><div class="small-muted">Fill in the boxes one step at a time. Not sure what a box means? Hover over the ❔ icon beside it. When you are done, press <b>Analyze Transaction</b> at the bottom.</div></div>', unsafe_allow_html=True)

sections = [
    ("1. Basic transaction details", ["payee_type", "merchant_category", "transaction_type", "amount_inr"]),
    ("2. Location, time and device", ["transaction_latitude", "transaction_longitude", "distance_from_home_km", "hour_of_day", "day_of_week", "is_weekend", "is_night", "is_new_device", "is_traveling", "location_context"]),
    ("3. Payee and transaction activity", ["is_new_payee", "payee_account_age_days", "txns_last_1h", "txns_last_24h", "distinct_payees_last_24h", "minutes_since_last_txn", "historical_avg_amount", "amount_to_avg_ratio"]),
    ("4. Balance, offers and other conditions", ["balance_before_inr", "balance_drain_ratio", "is_festival_period", "festival_name", "is_sale_day", "sale_campaign", "is_payday_period", "is_offer_exposed", "offer_type", "is_power_outage", "is_network_disruption", "disruption_type", "retry_count", "payment_attempt_count", "is_payment_retry"]),
]

# Give every manual box a starting value (the sample and clear buttons update these later).
for _field in FEATURE_COLUMNS:
    if f"manual_{_field}" not in st.session_state:
        st.session_state[f"manual_{_field}"] = widget_value(_field, DEFAULTS[_field])

# Two helper buttons above the form.
btn_sample_col, btn_clear_col = st.columns(2)
with btn_sample_col:
    st.button("🧪 Load a sample payment", key="btn_sample", on_click=load_sample_payment,
              help="Fills every box with an example payment, so you can see how the check works.",
              use_container_width=True)
with btn_clear_col:
    st.button("🧹 Clear form", key="btn_clear", on_click=clear_manual_form,
              help="Resets every box to its starting value.", use_container_width=True)

with st.form("manual_transaction_form", clear_on_submit=False):
    values = {}
    for section_title, fields in sections:
        st.markdown(f"#### {section_title}")
        cols = st.columns(2)
        for i, field in enumerate(fields):
            with cols[i % 2]:
                label = FIELD_LABELS[field]
                help_text = FIELD_HELP.get(field)
                key = f"manual_{field}"
                if field in BOOLEAN_FIELDS:
                    choice = st.selectbox(label, ["No", "Yes"], key=key, help=help_text)
                    values[field] = int(choice == "Yes")
                elif field == "day_of_week":
                    day_name = st.selectbox(label, list(DAY_CODES.keys()), key=key, help=help_text)
                    values[field] = DAY_CODES[day_name]
                elif field in CATEGORICAL_FIELDS:
                    options = MODEL_CATEGORIES.get(field, []) or TEXT_CHOICES.get(field, [])
                    if options:
                        values[field] = st.selectbox(label, options, key=key, help=help_text)
                    else:
                        values[field] = st.text_input(label, key=key, help=help_text)
                elif field == "hour_of_day":
                    values[field] = st.number_input(label, min_value=0, max_value=23, step=1, key=key, help=help_text)
                elif field == "transaction_latitude":
                    values[field] = st.number_input(label, min_value=-90.0, max_value=90.0, format="%.6f", key=key, help=help_text)
                elif field == "transaction_longitude":
                    values[field] = st.number_input(label, min_value=-180.0, max_value=180.0, format="%.6f", key=key, help=help_text)
                elif field in {"payee_account_age_days", "txns_last_1h", "txns_last_24h", "distinct_payees_last_24h", "retry_count", "payment_attempt_count"}:
                    values[field] = st.number_input(label, min_value=0, step=1, key=key, help=help_text)
                else:
                    values[field] = st.number_input(label, min_value=0.0, step=1.0, key=key, help=help_text)
    submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary", use_container_width=True)

# The "analyzing" animation appears right under the button.
manual_anim = st.empty()
if submitted:
    input_df = pd.DataFrame([{col: values[col] for col in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    try:
        probabilities, predictions = run_analysis(input_df, manual_anim, "Checking this transaction...")
        results = result_table(input_df, probabilities, predictions)
        add_results_to_history(results, "manual")
        request_scroll("result-manual")
    except Exception as exc:
        st.error(f"Could not analyze this transaction: {exc}")
        st.info("Please check that every box is filled in correctly, then try again.")

render_results("manual")

# ============================================================
# Section 2: Upload Dataset (CSV)
# ============================================================
st.markdown("---")
st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-card"><div class="section-title">☁️ Upload Dataset (CSV): check many transactions</div><div class="small-muted">Your file should have one transaction per row. Follow the three steps below.</div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="red-note">⚠️ <b>Important:</b> The CSV file you upload should have all <b>37 feature columns</b> to get good accuracy. '
    'If any column is missing, the check will not start, and the missing column names will be shown to you.</div>',
    unsafe_allow_html=True,
)
st.markdown("**Step 1:** Download the sample template. It already has the 37 correct column names.")
template = pd.DataFrame([{c: DEFAULTS[c] for c in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
st.download_button("⬇️ Download sample CSV template", template.to_csv(index=False).encode("utf-8"), "upi_securecheck_template.csv", "text/csv")
st.markdown("**Step 2:** Fill in your transactions using the same column names.")
st.markdown("**Step 3:** Upload your file below, then press Analyze.")
uploaded = st.file_uploader("Drag your CSV file here, or click to choose a file", type=["csv"], key="transaction_csv")

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLUMNS if c not in raw_df.columns]
        st.write(f"Columns found: **{len(FEATURE_COLUMNS) - len(missing)} of {len(FEATURE_COLUMNS)}**")
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif raw_df.empty:
            st.warning("The uploaded CSV has no transaction rows.")
        else:
            input_df = raw_df[FEATURE_COLUMNS].copy()
            # Accept day names ("Sunday") in the CSV too; numbers pass through unchanged.
            input_df["day_of_week"] = input_df["day_of_week"].map(
                lambda v: DAY_CODES.get(str(v).strip().title(), v)
            )
            st.success(f"Loaded {len(input_df):,} transaction(s). Here is a preview of the first 5 rows:")
            st.dataframe(input_df.head(5), use_container_width=True, hide_index=True)
            if st.button("🔍 Analyze Uploaded Transactions", type="primary", key="analyze_csv", use_container_width=True):
                csv_anim = st.empty()
                try:
                    probabilities, predictions = run_analysis(input_df, csv_anim, "Checking your transactions...")
                    results = result_table(input_df, probabilities, predictions)
                    add_results_to_history(results, "csv")
                    request_scroll("result-csv")
                except Exception as exc:
                    st.error(f"Could not analyze this file: {exc}")
    except Exception as exc:
        st.error(f"Could not read or process this CSV: {exc}")

render_results("csv")

# ============================================================
# Fraud statistics
# ============================================================
st.markdown("---")
st.markdown('<div id="fraud-statistics"></div>', unsafe_allow_html=True)
st.markdown("## 📊 Fraud statistics")
if history_df.empty:
    st.info("Statistics will appear here after you analyze at least one transaction.")
else:
    hist = pd.DataFrame(st.session_state.history)
    n_total = len(hist)
    n_fraud = int((hist["prediction"] == "Suspicious").sum())
    n_legit = n_total - n_fraud
    c1, c2 = st.columns([1, 1.2])
    with c1:
        donut = pd.DataFrame({"Status": ["Likely Legitimate", "Suspicious"], "Transactions": [n_legit, n_fraud]})
        fig = px.pie(donut, names="Status", values="Transactions", hole=0.62,
                     color="Status", color_discrete_map={"Likely Legitimate": "#20d58a", "Suspicious": "#ff4c78"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", legend_title_text="", margin=dict(t=15,b=15,l=5,r=5))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "transaction_type" in hist.columns:
            by_type = hist.groupby(["transaction_type", "prediction"]).size().reset_index(name="Transactions")
            fig2 = px.bar(by_type, x="transaction_type", y="Transactions", color="prediction", barmode="group",
                          color_discrete_map={"Suspicious": "#ff4c78", "Likely Legitimate": "#8d59ef"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white", xaxis_title="Transaction type", yaxis_title="Transactions",
                               legend_title_text="", margin=dict(t=15,b=15,l=5,r=5))
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# Model performance
# ============================================================
st.markdown("---")
st.markdown('<div id="model-performance"></div>', unsafe_allow_html=True)
st.markdown("## 🏆 Model performance — XGBoost")
st.caption("These are the project's held-out test-set metrics, not live metrics calculated from the current session.")
perf = [
    ("Accuracy", "99.46%", "Overall correct predictions"),
    ("Precision", "86.99%", "Share of flagged cases that were fraud"),
    ("Recall", "53.50%", "Share of fraud cases detected"),
    ("F1-score", "66.25%", "Balance of precision and recall"),
    ("ROC-AUC", "96.14%", "Class separation across thresholds"),
    ("PR-AUC", "68.84%", "Precision-recall performance"),
]
perf_cols = st.columns(3)
for i, (name, value, note) in enumerate(perf):
    with perf_cols[i % 3]:
        st.markdown(f'<div class="metric purple"><div class="metric-label">{name}</div><div class="metric-value">{value}</div><div class="small-muted">{note}</div></div>', unsafe_allow_html=True)
cm = pd.DataFrame([[19784, 16], [93, 107]],
                  index=["Actual legitimate", "Actual fraud"],
                  columns=["Predicted legitimate", "Predicted fraud"])
st.markdown("#### Confusion matrix")
st.dataframe(cm, use_container_width=True)
st.caption(f"Saved fraud decision threshold: {FRAUD_THRESHOLD:.7f}")
st.warning("This is an academic demonstration trained on synthetic transaction data. It is not connected to a bank or live UPI network, and these metrics do not establish real-world banking performance.")

# ============================================================
# Transaction history
# ============================================================
st.markdown("---")
st.markdown('<div id="transaction-history"></div>', unsafe_allow_html=True)
st.markdown("## 🕘 Transaction history")
st.caption("Every payment you checked in this session is listed here.")
if not st.session_state.history:
    st.info("No transactions have been checked in this browser session yet.")
else:
    history_view = pd.DataFrame(st.session_state.history)
    preferred = ["checked_at", "transaction_type", "amount_inr", "prediction", "fraud_risk_percentage", "risk_level"]
    visible = [c for c in preferred if c in history_view.columns]
    st.dataframe(history_view[visible], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download transaction history", history_view.to_csv(index=False).encode("utf-8"), "upi_securecheck_history.csv", "text/csv")
    if st.button("Clear session history", key="clear_history"):
        st.session_state.history = []
        st.session_state.latest_results = None
        st.session_state.result_source = None
        st.rerun()

st.markdown("---")
st.markdown('<div style="text-align:center;color:#bdb5df;padding:12px">🛡️ UPI SecureCheck · Academic Machine-Learning Project · Synthetic-data demonstration only</div>', unsafe_allow_html=True)

# Glide to the requested section, if a tile or button asked for it.
run_pending_scroll()