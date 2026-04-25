"""
🚀 Spaceship Titanic — Model Eğitimi
Çalıştır: python save_model.py
Çıktı:    model.joblib + feature_columns.joblib
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ── Constants ────────────────────────────────────────────────────────────────
TARGET = "Transported"

HOME_PLANET_MAP = {"Earth": 0, "Europa": 1, "Mars": 2}
DESTINATION_MAP = {"55 Cancri e": 0, "PSO J318.5-22": 1, "TRAPPIST-1e": 2}
DECK_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
SIDE_MAP = {"P": 0, "S": 1}

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

FEATURE_COLS = [
    "HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "deck", "side", "total_spent", "is_alone",
]


# ── Feature Engineering ──────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill missing — numeric → median, categorical → mode
    for col in SPENDING_COLS:
        df[col] = df[col].fillna(0.0)
    df["Age"] = df["Age"].fillna(df["Age"].median())

    for col in ["HomePlanet", "Destination"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["CryoSleep"] = df["CryoSleep"].fillna(df["CryoSleep"].mode()[0])
    df["VIP"] = df["VIP"].fillna(df["VIP"].mode()[0])

    # CryoSleep and VIP: bool → int  (True → 1, False → 0)
    df["CryoSleep"] = (
        df["CryoSleep"]
        .map({True: 1, False: 0, "True": 1, "False": 0})
        .fillna(0)
        .astype(int)
    )
    df["VIP"] = (
        df["VIP"]
        .map({True: 1, False: 0, "True": 1, "False": 0})
        .fillna(0)
        .astype(int)
    )

    # Cabin → deck, cabin_num, side
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["deck"] = cabin_split[0].fillna("G")
    df["side"] = cabin_split[2].fillna("S")

    # total_spent
    df["total_spent"] = df[SPENDING_COLS].sum(axis=1)

    # is_alone: group size == 1 from PassengerId  (format: XXXX_YY)
    group_id = df["PassengerId"].str.split("_").str[0]
    group_size = group_id.map(group_id.value_counts())
    df["is_alone"] = (group_size == 1).astype(int)

    # Label / ordinal encode categorical columns
    df["HomePlanet"] = df["HomePlanet"].map(HOME_PLANET_MAP).fillna(0).astype(int)
    df["Destination"] = df["Destination"].map(DESTINATION_MAP).fillna(2).astype(int)
    df["deck"] = df["deck"].map(DECK_MAP).fillna(5).astype(int)
    df["side"] = df["side"].map(SIDE_MAP).fillna(0).astype(int)

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Veri yükleniyor...")
    df = pd.read_csv("train.csv")
    print(f"Veri: {df.shape}")

    df = feature_engineering(df)

    # Target: Transported True → 1, False → 0
    df[TARGET] = (
        df[TARGET]
        .map({True: 1, False: 0, "True": 1, "False": 0})
        .astype(int)
    )

    x = df[FEATURE_COLS].astype(float)
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print("CatBoost eğitiliyor...")
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Not Transported", "Transported"]))

    joblib.dump(model, "model.joblib")
    joblib.dump(FEATURE_COLS, "feature_columns.joblib")
    print("\n✅ model.joblib ve feature_columns.joblib kaydedildi.")


if __name__ == "__main__":
    main()
