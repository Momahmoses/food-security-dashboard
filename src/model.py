"""
Food security IPC phase classification model.
Multi-class classifier predicting IPC Phase 1–5 per LGA.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

FEATURES = [
    "conflict_score", "rainfall_mm", "ndvi", "yield_kg_ha",
    "price_index", "pct_food_insecure", "stunting_pct", "wasting_pct",
    "dietary_diversity_score", "market_access_km", "displacement_pct",
    "season_enc", "state_enc",
]
TARGET = "ipc_phase"
MODEL_PATH = Path("assets/food_security_model.pkl")
META_PATH = Path("assets/food_security_meta.json")


def _encode(df: pd.DataFrame):
    df = df.copy()
    le_state = LabelEncoder()
    le_season = LabelEncoder()
    df["state_enc"] = le_state.fit_transform(df["state"])
    df["season_enc"] = le_season.fit_transform(df["season"])
    return df, le_state, le_season


def train(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    df, le_state, le_season = _encode(df)
    X = df[FEATURES].values
    y = df[TARGET].values

    clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    for train_idx, val_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        cv_accs.append(clf.score(X[val_idx], y[val_idx]))

    clf.fit(X, y)
    report = classification_report(y, clf.predict(X), output_dict=True)

    meta = {
        "cv_accuracy_mean": float(np.mean(cv_accs)),
        "cv_accuracy_std": float(np.std(cv_accs)),
        "train_accuracy": report["accuracy"],
        "feature_importances": dict(zip(FEATURES, clf.feature_importances_.tolist())),
        "state_classes": le_state.classes_.tolist(),
        "season_classes": le_season.classes_.tolist(),
    }
    return clf, meta


def save_model(clf, meta: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2))


def load_model():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())
