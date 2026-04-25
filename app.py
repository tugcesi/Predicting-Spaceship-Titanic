"""
🚀 Spaceship Titanic — Yolcu Transport Tahmini
Çalıştır: streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Spaceship Titanic",
    page_icon="🚀",
    layout="wide",
)

# ── Constants (identical to save_model.py) ────────────────────────────────────
HOME_PLANET_MAP = {"Earth": 0, "Europa": 1, "Mars": 2}
DESTINATION_MAP = {"55 Cancri e": 0, "PSO J318.5-22": 1, "TRAPPIST-1e": 2}
DECK_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
SIDE_MAP = {"P": 0, "S": 1}

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

MODEL_PATH = Path("model.joblib")
FEATURES_PATH = Path("feature_columns.joblib")


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


model, feature_columns = load_artifacts()

if model is None:
    st.error(
        "❌ Model dosyaları bulunamadı. Lütfen önce `python save_model.py` çalıştırın."
    )
    st.stop()


# ── Feature engineering (identical to save_model.py) ─────────────────────────
def build_features(
    home_planet: str,
    cryo_sleep: bool,
    destination: str,
    age: float,
    vip: bool,
    room_service: float,
    food_court: float,
    shopping_mall: float,
    spa: float,
    vr_deck: float,
    deck: str,
    side: str,
    is_alone: int,
) -> pd.DataFrame:
    total_spent = room_service + food_court + shopping_mall + spa + vr_deck

    row = {
        "HomePlanet": HOME_PLANET_MAP.get(home_planet, 0),
        "CryoSleep": int(cryo_sleep),
        "Destination": DESTINATION_MAP.get(destination, 2),
        "Age": float(age),
        "VIP": int(vip),
        "RoomService": float(room_service),
        "FoodCourt": float(food_court),
        "ShoppingMall": float(shopping_mall),
        "Spa": float(spa),
        "VRDeck": float(vr_deck),
        "deck": DECK_MAP.get(deck, 5),
        "side": SIDE_MAP.get(side, 0),
        "total_spent": float(total_spent),
        "is_alone": int(is_alone),
    }

    df = pd.DataFrame([row], columns=feature_columns)
    return df.astype(float)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🚀 Yolcu Bilgileri")
st.sidebar.markdown("---")

home_planet = st.sidebar.selectbox("🌍 HomePlanet", ["Europa", "Earth", "Mars"])
cryo_sleep_str = st.sidebar.selectbox("❄️ CryoSleep", ["False", "True"])
destination = st.sidebar.selectbox(
    "🎯 Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
)
age = st.sidebar.slider("🎂 Age", min_value=0, max_value=80, value=28)
vip_str = st.sidebar.selectbox("⭐ VIP", ["False", "True"])

st.sidebar.markdown("---")
deck = st.sidebar.selectbox("🛳️ Deck", ["A", "B", "C", "D", "E", "F", "G", "T"])
side = st.sidebar.selectbox("🔀 Side", ["P", "S"])

st.sidebar.markdown("---")
room_service = st.sidebar.number_input("🛎️ RoomService", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
food_court = st.sidebar.number_input("🍔 FoodCourt", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
shopping_mall = st.sidebar.number_input("🛍️ ShoppingMall", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
spa = st.sidebar.number_input("💆 Spa", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
vr_deck = st.sidebar.number_input("🎮 VRDeck", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)

st.sidebar.markdown("---")
is_alone_str = st.sidebar.selectbox("👤 Is Alone?", ["No", "Yes"])

predict_btn = st.sidebar.button("🚀 Predict", use_container_width=True)

# Convert inputs
cryo_sleep = cryo_sleep_str == "True"
vip = vip_str == "True"
is_alone = 1 if is_alone_str == "Yes" else 0
total_spent = room_service + food_court + shopping_mall + spa + vr_deck

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🚀 Spaceship Titanic — Transport Tahmin Uygulaması")
st.markdown(
    "Yolcu bilgilerini girerek **uzay-zaman anomalisine** kapılıp kapılmadığını tahmin edin!"
)
st.markdown("---")

# ── Prediction (runs when button is clicked) ──────────────────────────────────
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False

if predict_btn:
    st.session_state.show_prediction = True

if st.session_state.show_prediction:
    X_input = build_features(
        home_planet, cryo_sleep, destination, age, vip,
        room_service, food_court, shopping_mall, spa, vr_deck,
        deck, side, is_alone,
    )

    proba_raw = model.predict_proba(X_input)[0]
    prob = float(np.clip(proba_raw[1], 0, 1))
    prediction = model.predict(X_input)[0]
    is_transported = int(prediction) == 1

    pred_label = "✅ Transported" if is_transported else "❌ Not Transported"

    # ── Risk banner ───────────────────────────────────────────────────────────
    if prob >= 0.6:
        st.success(f"**Transport Olasılığı Yüksek:** {prob * 100:.1f}%  —  {pred_label}")
    elif prob >= 0.4:
        st.warning(f"**Transport Olasılığı Orta:** {prob * 100:.1f}%  —  {pred_label}")
    else:
        st.error(f"**Transport Olasılığı Düşük:** {prob * 100:.1f}%  —  {pred_label}")

    # ── Metric cards ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔮 Prediction", pred_label)
    col2.metric("📊 Probability", f"{prob * 100:.1f}%")
    col3.metric("💰 Total Spent", f"${total_spent:,.0f}")
    col4.metric("🎂 Age", f"{age}")

    st.markdown("---")

    # ── Gauge ─────────────────────────────────────────────────────────────────
    gauge_col, table_col = st.columns([1, 1])

    with gauge_col:
        st.subheader("🎯 Transport Probability Gauge")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={"text": "Transport Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "royalblue"},
                    "steps": [
                        {"range": [0, 40], "color": "lightcoral"},
                        {"range": [40, 60], "color": "lightyellow"},
                        {"range": [60, 100], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": prob * 100,
                    },
                },
                delta={"reference": 50},
            )
        )
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Input summary table ───────────────────────────────────────────────────
    with table_col:
        st.subheader("📋 Girilen Bilgiler")
        summary = pd.DataFrame(
            {
                "Özellik": [
                    "HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
                    "Deck", "Side", "RoomService", "FoodCourt", "ShoppingMall",
                    "Spa", "VRDeck", "Total Spent", "Is Alone",
                ],
                "Değer": [
                    home_planet, cryo_sleep_str, destination, age, vip_str,
                    deck, side, room_service, food_court, shopping_mall,
                    spa, vr_deck, total_spent, is_alone_str,
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")

else:
    st.info("👈 Sol panelden yolcu bilgilerini girin ve **🚀 Predict** butonuna tıklayın.")
    st.markdown("---")

# ── Feature importance ────────────────────────────────────────────────────────
st.subheader("📈 Feature Importance")

try:
    importances = model.get_feature_importance()
    feat_names = feature_columns
    fi_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    fig_fi = go.Figure(
        go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker_color="steelblue",
        )
    )
    fig_fi.update_layout(
        title="CatBoost Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=450,
        margin={"l": 120},
    )
    st.plotly_chart(fig_fi, use_container_width=True)
except Exception:
    st.info("Feature importance alınamadı.")

# ── Model parameters expander ─────────────────────────────────────────────────
with st.expander("⚙️ Model Parametreleri"):
    try:
        params = model.get_all_params()
        params_df = pd.DataFrame(
            {"Parametre": list(params.keys()), "Değer": list(params.values())}
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    except Exception:
        st.json({"model": "CatBoostClassifier", "random_state": 42, "verbose": 0})
