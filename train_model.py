"""
train_model.py – Spaceship Titanic Model Training

Run:    python train_model.py
Output: model.joblib, scaler.joblib, feature_columns.joblib, encoders.joblib
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    _USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _USE_XGB = False

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET = "Transported"

CATEGORICAL_COLS = ["HomePlanet", "Destination", "Deck", "Side"]
NUMERIC_COLS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

FEATURE_COLS = [
    "HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Deck", "Side", "TotalSpend",
]


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()

    # Group info from PassengerId (format: XXXX_YY)
    df["Group"] = df["PassengerId"].str.split("_").str[0]

    # Split Cabin into Deck, CabinNum, Side
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    # Map CryoSleep and VIP to boolean integers (True=1, False=0)
    for col in ["CryoSleep", "VIP"]:
        df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0})

    # Fill missing categorical columns with mode
    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in ["CryoSleep", "VIP"]:
        df[col] = df[col].fillna(0).astype(int)

    # Fill missing numeric columns with KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    df[NUMERIC_COLS] = imputer.fit_transform(df[NUMERIC_COLS])

    # TotalSpend feature
    df["TotalSpend"] = df[SPENDING_COLS].sum(axis=1)

    # Label-encode categorical columns
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda v: v if v in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # Drop unused columns
    df = df.drop(
        columns=["PassengerId", "Name", "Cabin", "Group", "CabinNum"],
        errors="ignore",
    )

    return df, encoders


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data...")
    df = pd.read_csv("train.csv")
    print(f"Shape: {df.shape}")

    # Target: True → 1, False → 0
    df[TARGET] = (
        df[TARGET]
        .map({True: 1, False: 0, "True": 1, "False": 0})
        .astype(int)
    )

    # Preprocess
    encoders: dict[str, LabelEncoder] = {}
    df_processed, encoders = preprocess(df, encoders=encoders, fit=True)

    X = df_processed[FEATURE_COLS].astype(float)
    y = df_processed[TARGET]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train model
    if _USE_XGB:
        print("Training XGBClassifier...")
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
    else:
        print("Training GradientBoostingClassifier (xgboost not available)...")
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Save artifacts
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(FEATURE_COLS, "feature_columns.joblib")
    joblib.dump(encoders, "encoders.joblib")
    print(
        "\n✅ Saved: model.joblib, scaler.joblib, "
        "feature_columns.joblib, encoders.joblib"
    )


if __name__ == "__main__":
    main()
