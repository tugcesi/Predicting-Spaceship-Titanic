"""
app.py – Spaceship Titanic Transported Prediction
Run: streamlit run app.py
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
    page_title="🚀 Spaceship Titanic – Transported Prediction",
    page_icon="🚀",
    layout="wide",
)

# ── Artifact paths ────────────────────────────────────────────────────────────
MODEL_PATH = Path("model.joblib")
SCALER_PATH = Path("scaler.joblib")
FEATURES_PATH = Path("feature_columns.joblib")
ENCODERS_PATH = Path("encoders.joblib")


# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not all(
        p.exists() for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH, ENCODERS_PATH]
    ):
        return None, None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, feature_columns, encoders


model, scaler, feature_columns, encoders = load_artifacts()

if model is None:
    st.warning(
        "Please run `python train_model.py` first to generate model files."
    )
    st.stop()

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🚀 Spaceship Titanic – Transported Prediction")
st.markdown("---")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Passenger Information")

home_planet = st.sidebar.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
cryo_sleep = st.sidebar.selectbox("CryoSleep", ["False", "True"])
destination = st.sidebar.selectbox(
    "Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
)
age = st.sidebar.slider("Age", min_value=0, max_value=79, value=27)
vip = st.sidebar.selectbox("VIP", ["False", "True"])
room_service = st.sidebar.number_input(
    "RoomService", min_value=0, max_value=15000, value=0
)
food_court = st.sidebar.number_input(
    "FoodCourt", min_value=0, max_value=30000, value=0
)
shopping_mall = st.sidebar.number_input(
    "ShoppingMall", min_value=0, max_value=25000, value=0
)
spa = st.sidebar.number_input("Spa", min_value=0, max_value=25000, value=0)
vr_deck = st.sidebar.number_input("VRDeck", min_value=0, max_value=25000, value=0)
deck = st.sidebar.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "T"])
side = st.sidebar.selectbox("Side", ["P", "S"])

predict_btn = st.sidebar.button("🔍 Predict", use_container_width=True)


# ── Build feature row ─────────────────────────────────────────────────────────
def build_input_df() -> pd.DataFrame:
    total_spend = room_service + food_court + shopping_mall + spa + vr_deck

    row: dict = {
        "HomePlanet": home_planet,
        "CryoSleep": 1 if cryo_sleep == "True" else 0,
        "Destination": destination,
        "Age": float(age),
        "VIP": 1 if vip == "True" else 0,
        "RoomService": float(room_service),
        "FoodCourt": float(food_court),
        "ShoppingMall": float(shopping_mall),
        "Spa": float(spa),
        "VRDeck": float(vr_deck),
        "Deck": deck,
        "Side": side,
        "TotalSpend": float(total_spend),
    }

    df = pd.DataFrame([row])

    # Encode categorical columns using loaded encoders
    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        le = encoders[col]
        val = str(df[col].iloc[0])
        if val not in le.classes_:
            val = le.classes_[0]
        df[col] = le.transform([val])[0]

    df = df[feature_columns].astype(float)
    df_scaled = scaler.transform(df)
    return pd.DataFrame(df_scaled, columns=feature_columns)


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    X_input = build_input_df()
    proba = model.predict_proba(X_input)[0]
    prob_transported = float(np.clip(proba[1], 0.0, 1.0))
    prediction = int(model.predict(X_input)[0])

    if prediction == 1:
        st.markdown(
            "## 🌀 This passenger was **TRANSPORTED** to another dimension!"
        )
        st.success(f"Transport Probability: {prob_transported * 100:.1f}%")
    else:
        st.markdown(
            "## ✅ This passenger was **NOT transported** and stayed on the ship."
        )
        st.info(f"Transport Probability: {prob_transported * 100:.1f}%")

    # ── Gauge chart ───────────────────────────────────────────────────────────
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_transported * 100,
            title={"text": "Transport Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "royalblue"},
                "steps": [
                    {"range": [0, 40], "color": "lightcoral"},
                    {"range": [40, 60], "color": "lightyellow"},
                    {"range": [60, 100], "color": "lightgreen"},
                ],
            },
        )
    )
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("### 📊 Feature Importance")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame(
                {"feature": feature_columns, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .head(10)
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
            title="Top 10 Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model.")

else:
    st.info(
        "👈 Enter passenger information in the sidebar and click **🔍 Predict**."
    )

# ── About expander ────────────────────────────────────────────────────────────
with st.expander("ℹ️ About"):
    st.markdown(
        """
        ### Spaceship Titanic Challenge

        Welcome to the year 2912! The **Spaceship Titanic** was an interstellar passenger
        liner on its maiden voyage when it collided with a spacetime anomaly.
        Almost half of the ~13,000 passengers were transported to an alternate dimension.

        This app uses a machine learning model trained on recovered passenger records
        to predict whether a given passenger was **transported** or **stayed on the ship**.

        **Techniques used:**
        - KNN Imputation for missing numeric values
        - Label Encoding for categorical features
        - Feature engineering (TotalSpend, Cabin splitting)
        - XGBoost / Gradient Boosting Classifier
        - StandardScaler for feature normalisation

        **Data source:** [Kaggle – Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
        """
    )
