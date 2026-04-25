# 🚀 Spaceship Titanic – Transported Prediction

A machine learning project that predicts which passengers aboard the **Spaceship Titanic** were transported to an alternate dimension after the vessel collided with a spacetime anomaly.

---

## Overview

Welcome to the year 2912! The Spaceship Titanic was an interstellar passenger liner on its maiden voyage when it collided with a spacetime anomaly hidden in a dust cloud near Alpha Centauri. Almost **half of the ~13,000 passengers** were transported to an alternate dimension.

Using records recovered from the ship's damaged computer system, this project trains a classification model to predict which passengers were transported — helping rescue crews change history!

This is based on the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition.

---

## Features

- 📊 **Exploratory Data Analysis** in `spaceship-titanic-classification.ipynb`
- 🔧 **Automated preprocessing pipeline** with KNN imputation and feature engineering
- 🤖 **XGBoost classifier** (falls back to Gradient Boosting if XGBoost is unavailable)
- 🌐 **Interactive Streamlit web app** for real-time passenger transport prediction
- 📈 **Plotly gauge chart** and feature importance visualisation

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train the model

```bash
python train_model.py
```

This reads `train.csv`, preprocesses the data, trains an XGBoost classifier, and saves:

- `model.joblib` – trained model
- `scaler.joblib` – fitted StandardScaler
- `feature_columns.joblib` – list of feature column names
- `encoders.joblib` – fitted LabelEncoders for categorical columns

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`). Enter passenger details in the sidebar and click **🔍 Predict** to see whether the passenger was transported.

---

## Model

| Component | Detail |
|-----------|--------|
| Algorithm | XGBClassifier (GradientBoostingClassifier fallback) |
| Imputation | KNNImputer (k=5) for numeric features |
| Encoding | LabelEncoder for HomePlanet, Destination, Deck, Side |
| Scaling | StandardScaler |
| Target | Transported → 1 (True), 0 (False) |

### ML Techniques

- **KNN Imputation** – missing numeric values are filled using the k=5 nearest neighbours
- **Feature Engineering** – `TotalSpend` (sum of all spending columns), `Deck` / `CabinNum` / `Side` extracted from `Cabin`
- **Label Encoding** – categorical columns encoded as ordinal integers
- **XGBoost / GradientBoosting** – ensemble gradient boosting for robust binary classification

---

## Project Structure

```
Predicting-Spaceship-Titanic/
│
├── spaceship-titanic-classification.ipynb  # EDA and model experiments
├── train_model.py                          # Training script → saves model artifacts
├── app.py                                  # Streamlit prediction app
├── train.csv                               # Training data
├── test.csv                                # Test data
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
plotly
```
