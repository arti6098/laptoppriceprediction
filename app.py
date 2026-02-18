import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
)

# ----------------------------
# Load model + dataframe
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

with open(BASE_DIR / "df.pkl", "rb") as f:
    df = pickle.load(f)

# ----------------------------
# Styling (card UI + clean inputs)
# ----------------------------
st.markdown(
    """
<style>
/* clean inputs */
div[data-testid="stTextInput"],
div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"] {
  border: none !important;
  box-shadow: none !important;
}

/* simple UI */
.big-title { font-size: 40px; font-weight: 800; margin: 8px 0 0 0; }
.subtitle  { font-size: 16px; opacity: 0.8; margin-bottom: 18px; }

.card {
  padding: 18px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
}

.kpi { font-size: 14px; opacity: 0.8; margin-top: 8px; }
.price { font-size: 44px; font-weight: 900; margin-top: 6px; }
.small { font-size: 12px; opacity: 0.75; margin-top: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="big-title">💻 Laptop Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter configuration and get an estimated price (₹ INR).</div>', unsafe_allow_html=True)
st.write("")

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("⚙️ Laptop Configuration")

Company = st.sidebar.selectbox("Brand", sorted(df["Company"].dropna().unique()))
TypeName = st.sidebar.selectbox("Type", sorted(df["TypeName"].dropna().unique()))

col1, col2 = st.sidebar.columns(2)
Ram = col1.selectbox("RAM (GB)", [2, 4, 6, 8, 10, 12, 14, 16, 18], index=3)
Weight = col2.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1, value=1.8)

Touchscreen = st.sidebar.radio("Touchscreen", ["No", "Yes"], horizontal=True)
Ips = st.sidebar.radio("IPS Display", ["No", "Yes"], horizontal=True)

Inches = st.sidebar.slider("Screen Size (inches)", 10.0, 18.0, 15.6)

Resolution = st.sidebar.selectbox(
    "Resolution",
    [
        "1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800",
        "2880x1800", "2560x1600", "2560x1440", "2304x1440"
    ],
    index=0,
)

Cpu_brand = st.sidebar.selectbox("CPU Brand", sorted(df["Cpu brand"].dropna().unique()))
HDD = st.sidebar.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048], index=0)
SSD = st.sidebar.selectbox("SSD (GB)", [0, 8, 128, 256, 512, 1024], index=4)
Gpu_brand = st.sidebar.selectbox("GPU Brand", sorted(df["Gpu brand"].dropna().unique()))
os_name = st.sidebar.selectbox("OS", sorted(df["os"].dropna().unique()))

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 Predict Price", use_container_width=True)

# ----------------------------
# Feature engineering
# ----------------------------
touchscreen_val = 1 if Touchscreen == "Yes" else 0
ips_val = 1 if Ips == "Yes" else 0

X_res = int(Resolution.split("x")[0])
Y_res = int(Resolution.split("x")[1])
ppi = ((X_res**2 + Y_res**2) ** 0.5) / Inches

# ----------------------------
# Build model input
#   - Add ONLY columns the model expects (prevents feature mismatch errors)
# ----------------------------
base_features = {
    "Company": Company,
    "TypeName": TypeName,
    "Inches": Inches,
    "Ram": Ram,
    "Weight": Weight,
    "Touchscreen": touchscreen_val,
    "Ips": ips_val,
    "ppi": ppi,
    "Cpu brand": Cpu_brand,
    "HDD": HDD,
    "SSD": SSD,
    "Gpu brand": Gpu_brand,
    "os": os_name,
    "Unnamed: 0": 0,  # will be removed if not expected
}

input_df = pd.DataFrame([base_features])

expected_cols = list(getattr(pipe, "feature_names_in_", []))
if expected_cols:
    # Keep ONLY expected columns and add missing ones as NaN (rare but safe)
    keep = [c for c in expected_cols if c in input_df.columns]
    input_df = input_df[keep].copy()
    for c in expected_cols:
        if c not in input_df.columns:
            input_df[c] = np.nan
    input_df = input_df[expected_cols]
else:
    # If pipeline doesn't expose feature_names_in_, at least drop Unnamed: 0 if df doesn't have it
    if "Unnamed: 0" not in df.columns and "Unnamed: 0" in input_df.columns:
        input_df = input_df.drop(columns=["Unnamed: 0"])

# ----------------------------
# Helper: decide if output is log(price)
# ----------------------------
def to_price(model_output: float) -> float:
    # If output looks like a log-price (~6 to 13), use exp; otherwise treat as price.
    if 3 < model_output < 20:
        return float(np.exp(model_output))
    return float(model_output)

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1.25, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Selected Configuration")
    st.write(
        f"**{Company}** · {TypeName} · **{Ram}GB RAM** · **{Inches:.1f}\"** · {Resolution} · "
        f"{'IPS' if ips_val else 'Non-IPS'} · {'Touch' if touchscreen_val else 'Non-touch'}"
    )
    st.write(f"**CPU:** {Cpu_brand}")
    st.write(f"**GPU:** {Gpu_brand}")
    st.write(f"**Storage:** SSD {SSD}GB + HDD {HDD}GB")
    st.write(f"**OS:** {os_name}")
    st.write(f"**Weight:** {Weight:.1f} kg · **PPI:** {ppi:.1f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    with st.expander("🧪 Show model input (debug)"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 Price Prediction")

    if predict_btn:
        try:
            with st.spinner("Predicting..."):
                raw_pred = float(pipe.predict(input_df)[0])
                pred_price = to_price(raw_pred)

            st.write('<div class="kpi">Estimated Price</div>', unsafe_allow_html=True)
            st.write(f'<div class="price">₹ {int(pred_price):,}</div>', unsafe_allow_html=True)
            st.write('<div class="small">Note: This is an estimate based on training data.</div>', unsafe_allow_html=True)

            # optional debug
            with st.expander("🔎 Prediction debug"):
                st.write("Raw model output:", raw_pred)
                st.write("Interpreted as price:", pred_price)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Tip: Most common reason is column/feature mismatch between training and this input.")
    else:
        st.info("Set laptop configuration in the sidebar, then click **🚀 Predict Price**.")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Built with Streamlit · Laptop Price Prediction ML Project")
