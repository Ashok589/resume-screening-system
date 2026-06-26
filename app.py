import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zepto Delivery Partner Earnings Predictor",
    page_icon="",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%);
    }
    .result-card {
        background: linear-gradient(135deg, #14532d, #15803d);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(21,128,61,0.3);
    }
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        margin: 0.3rem;
    }
    .section-header {
        color: #7dd3fc;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.5rem 0;
        padding-left: 0.3rem;
        border-left: 3px solid #7c3aed;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f2e 0%, #0d1b2a 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(124,58,237,0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 🖼️ Banner Poster
# ─────────────────────────────────────────────
from PIL import Image
import os

if os.path.exists("zepto_poster.png"):
    poster = Image.open("zepto_poster.png")
    st.image(poster, use_container_width=True)


# ─────────────────────────────────────────────
# Load Model & Scaler
# ─────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    model = load_model("zepto_ann_model.keras")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()
# ─────────────────────────────────────────────
# LabelEncoder mappings (alphabetical — matches sklearn LabelEncoder)
# ─────────────────────────────────────────────
label_maps = {
    "store_name": {
        "Zepto SS Store — Medipally": 0,
    },
    "delivery_area": {
        "Medipally": 0, "Uppal": 1, "Boduppal": 2,
        "Pocharam": 3, "Ghatkesar": 4, "Nagaram": 5,
        "Peerzadiguda": 6, "Keesara": 7, "Kondapur": 8,
        "Other": 9,
    },
    "grocery_category": {
        "Fruits & Vegetables": 0, "Dairy & Breakfast": 1,
        "Snacks & Beverages": 2, "Bakery & Biscuits": 3,
        "Personal Care": 4, "Household Essentials": 5,
        "Meats & Seafood": 6, "Baby Care": 7,
        "Frozen Food": 8, "Other": 9,
    },
    "product_item": {
        "Mixed": 0, "Single": 1, "Other": 2,
    },
    "vehicle_type": {
        "Bicycle": 0, "Bike": 1, "E-Bike": 2, "E-Scooter": 3,
    },
    "day_of_week": {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
    },
    "weather": {
        "Clear": 0, "Cloudy": 1, "Foggy": 2,
        "Heavy Rain": 3, "Light Rain": 4, "Windy": 5,
    },
}

def encode(col, val):
    return label_maps.get(col, {}).get(val, 0)

# ─────────────────────────────────────────────
# Sidebar Inputs
# ─────────────────────────────────────────────
sb = st.sidebar
sb.markdown("## ⚙️ Enter Partner Details")
sb.markdown("---")

sb.markdown('<p class="section-header" style="color:#7dd3fc">📍 Location & Route</p>', unsafe_allow_html=True)
delivery_area      = sb.selectbox("Delivery Area",      list(label_maps["delivery_area"].keys()))
distance_km        = sb.slider("Distance (km)",          0.5, 15.0, 3.5, 0.1)
store_latitude     = 17.4399
store_longitude    = 78.5667
delivery_latitude  = 17.4400
delivery_longitude = 78.5666

sb.markdown('<p class="section-header" style="color:#7dd3fc">📦 Order Details</p>', unsafe_allow_html=True)
grocery_category   = sb.selectbox("Grocery Category",   list(label_maps["grocery_category"].keys()))
product_item       = sb.selectbox("Product Item Type",  list(label_maps["product_item"].keys()))
num_items_in_order = sb.slider("Items in Order",         1, 20, 5)
vehicle_type       = sb.selectbox("Vehicle Type",        list(label_maps["vehicle_type"].keys()))

sb.markdown('<p class="section-header" style="color:#7dd3fc">📅 Shift & Conditions</p>', unsafe_allow_html=True)
day_of_week        = sb.selectbox("Day of Week",         list(label_maps["day_of_week"].keys()))
is_weekend         = sb.selectbox("Is Weekend?",         ["No", "Yes"])
is_festival_day    = sb.selectbox("Is Festival Day?",    ["No", "Yes"])
weather            = sb.selectbox("Weather",             list(label_maps["weather"].keys()))
login_hours        = sb.slider("Login Hours",            1.0, 12.0, 6.0, 0.5)
peak_hours_worked  = sb.slider("Peak Hours Worked",      0.0, 6.0, 2.0, 0.5)
surge_multiplier   = sb.slider("Surge Multiplier",       1.0, 3.0, 1.2, 0.1)

sb.markdown('<p class="section-header" style="color:#7dd3fc">💰 Earnings Components</p>', unsafe_allow_html=True)
orders_delivered        = sb.slider("Orders Delivered",       1, 30, 8)
avg_distance_per_order  = sb.slider("Avg Distance/Order (km)", 0.5, 10.0, 2.5, 0.1)
total_distance_km       = sb.slider("Total Distance (km)",    1.0, 100.0, float(orders_delivered * avg_distance_per_order), 0.5)
base_earning_per_order  = sb.number_input("Base Earning/Order (₹)", 10, 80, 25)
surge_earning           = sb.number_input("Surge Earning (₹)",       0.0, 500.0, 40.0)
peak_bonus              = sb.number_input("Peak Bonus (₹)",           0.0, 300.0, 30.0)
weekend_bonus           = sb.number_input("Weekend Bonus (₹)",        0, 200, 0 if is_weekend == "No" else 50)
festival_bonus          = sb.number_input("Festival Bonus (₹)",       0, 200, 0 if is_festival_day == "No" else 100)
fuel_deduction          = sb.number_input("Fuel Deduction (₹)",       0.0, 150.0, 20.0)

# Encode categoricals
order_id_enc   = 5000
partner_id_enc = 100
store_name_enc = encode("store_name", "Zepto SS Store — Medipally")

input_data = np.array([[
    order_id_enc,
    partner_id_enc,
    store_name_enc,
    store_latitude,
    store_longitude,
    encode("delivery_area", delivery_area),
    delivery_latitude,
    delivery_longitude,
    distance_km,
    encode("grocery_category", grocery_category),
    encode("product_item", product_item),
    num_items_in_order,
    encode("vehicle_type", vehicle_type),
    encode("day_of_week", day_of_week),
    1 if is_weekend == "Yes" else 0,
    1 if is_festival_day == "Yes" else 0,
    login_hours,
    peak_hours_worked,
    encode("weather", weather),
    surge_multiplier,
    orders_delivered,
    avg_distance_per_order,
    total_distance_km,
    base_earning_per_order,
    surge_earning,
    peak_bonus,
    weekend_bonus,
    festival_bonus,
    fuel_deduction,
]], dtype=float)

# ─────────────────────────────────────────────
# Main Panel — Predict Button
# ─────────────────────────────────────────────
st.markdown("### 📊 Prediction Dashboard")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮  Predict Total Earnings", use_container_width=True)

if predict_btn:
    input_scaled = scaler.transform(input_data)
    prediction   = float(model.predict(input_scaled, verbose=0)[0][0])
    prediction   = max(0.0, prediction)

    st.markdown("---")

    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        st.markdown(f"""
        <div class="result-card">
            <p style="color:rgba(255,255,255,0.75); margin:0; font-size:1rem; letter-spacing:0.08em; text-transform:uppercase;">
                💰 Predicted Total Earnings
            </p>
            <h1 style="color:white; font-size:3.5rem; margin:0.4rem 0; font-weight:900;">
                ₹ {prediction:,.2f}
            </h1>
            <p style="color:rgba(255,255,255,0.6); margin:0; font-size:0.9rem;">
                Based on {orders_delivered} orders over {login_hours:.1f} hrs
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 📈 Earnings Breakdown")
    gross = base_earning_per_order * orders_delivered
    bonuses = surge_earning + peak_bonus + weekend_bonus + festival_bonus
    per_order = prediction / orders_delivered if orders_delivered > 0 else 0
    per_hour  = prediction / login_hours if login_hours > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        (m1, "Base Earnings", f"₹ {gross:,.0f}", "#60a5fa"),
        (m2, "Total Bonuses",  f"₹ {bonuses:,.0f}", "#34d399"),
        (m3, "Fuel Deduction", f"- ₹ {fuel_deduction:,.0f}", "#f87171"),
        (m4, "Per Order",      f"₹ {per_order:,.1f}", "#a78bfa"),
        (m5, "Per Hour",       f"₹ {per_hour:,.1f}", "#fbbf24"),
    ]
    for col, label, val, color in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color:rgba(255,255,255,0.6); margin:0; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.07em;">{label}</p>
                <h3 style="color:{color}; margin:0.3rem 0; font-size:1.3rem; font-weight:700;">{val}</h3>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🔍 View Input Summary"):
        import pandas as pd
        summary = pd.DataFrame({
            "Feature": [
                "Delivery Area", "Distance (km)", "Vehicle Type", "Day of Week",
                "Weather", "Is Weekend", "Is Festival Day",
                "Orders Delivered", "Login Hours", "Peak Hours",
                "Surge Multiplier", "Grocery Category",
                "Base Earning/Order", "Surge Earning", "Peak Bonus",
                "Weekend Bonus", "Festival Bonus", "Fuel Deduction"
            ],
            "Value": [
                delivery_area, f"{distance_km} km", vehicle_type, day_of_week,
                weather, is_weekend, is_festival_day,
                orders_delivered, f"{login_hours} hrs", f"{peak_hours_worked} hrs",
                f"×{surge_multiplier}", grocery_category,
                f"₹{base_earning_per_order}", f"₹{surge_earning}", f"₹{peak_bonus}",
                f"₹{weekend_bonus}", f"₹{festival_bonus}", f"₹{fuel_deduction}"
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("#### 💡 Earnings Insight")
    if prediction >= 1000:
        st.success(f"🌟 **Excellent shift!** ₹{prediction:,.0f} is a high-earning day. "
                   f"Surge multiplier of ×{surge_multiplier} and {orders_delivered} deliveries contributed well.")
    elif prediction >= 600:
        st.info(f"👍 **Good earnings.** Consider working during more peak hours to push above ₹1,000. "
                f"Current: ₹{prediction:,.0f}")
    else:
        st.warning(f"📉 **Below average.** ₹{prediction:,.0f} — try increasing login hours, "
                   f"targeting festival days, or working peak-hour shifts for better earnings.")

else:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, label, desc in [
        (c1, "🗺️", "Location-Aware", "Accounts for delivery area, distance, and route"),
        (c2, "⏰", "Time & Shift Based", "Models surge hours, peak bonuses, weekends"),
        (c3, "💸", "Earnings Breakdown", "Separates base, bonuses, deductions"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:1.8rem;">
                <div style="font-size:2.2rem;">{icon}</div>
                <h4 style="color:#7dd3fc; margin:0.5rem 0 0.3rem;">{label}</h4>
                <p style="color:rgba(255,255,255,0.55); font-size:0.88rem; margin:0;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style="background:rgba(124,58,237,0.1); border:1px solid rgba(124,58,237,0.3);
                border-radius:12px; padding:1.2rem; text-align:center; margin-top:1rem;">
        <p style="color:rgba(255,255,255,0.6); margin:0; font-size:0.95rem;">
            👈 Fill in partner details in the sidebar, then click <strong style="color:#a78bfa">Predict Total Earnings</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:rgba(255,255,255,0.35); font-size:0.85rem;'>
    Zepto Delivery Partner Earnings Predictor &nbsp;|&nbsp; ANN Model &nbsp;|&nbsp;
    Built with Streamlit & TensorFlow &nbsp;|&nbsp; R² = 0.979
</p>
""", unsafe_allow_html=True)
