"""
Synthetic dataset generator for food security and nutrition monitoring.
Each record is an LGA (Local Government Area) observation with crop production,
market prices, IPC phase classification, and household-level indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_LGAS = 3000

STATES = [
    ("Borno", 11.83, 13.15), ("Yobe", 12.29, 11.44), ("Adamawa", 9.33, 12.40),
    ("Zamfara", 12.17, 6.66), ("Katsina", 12.98, 7.60), ("Sokoto", 13.06, 5.24),
    ("Kebbi", 12.45, 4.20), ("Niger", 9.93, 6.56), ("Bauchi", 10.31, 9.84),
    ("Gombe", 10.29, 11.17), ("Jigawa", 12.23, 9.56), ("Kano", 12.00, 8.52),
    ("Plateau", 9.22, 9.52), ("Nassarawa", 8.50, 8.54), ("Benue", 7.73, 8.52),
    ("Taraba", 7.87, 11.47),
]

IPC_PHASES = {1: "Minimal", 2: "Stressed", 3: "Crisis", 4: "Emergency", 5: "Famine"}
SEASONS = ["Wet Season", "Dry Season", "Off-Season"]
CONFLICT_LEVELS = ["None", "Low", "Moderate", "High", "Severe"]


def generate_food_security_dataset(n_lgas: int = N_LGAS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for lga_id in range(n_lgas):
        state, lat, lon = STATES[rng.integers(0, len(STATES))]
        season = SEASONS[rng.integers(0, len(SEASONS))]
        year = int(rng.integers(2018, 2025))
        conflict = CONFLICT_LEVELS[rng.integers(0, len(CONFLICT_LEVELS))]
        conflict_score = CONFLICT_LEVELS.index(conflict)

        # agricultural indicators
        rainfall_mm = max(0.0, float(rng.normal(800, 300)))
        ndvi = float(np.clip(rng.normal(0.45, 0.15), 0.05, 0.95))
        crop_production_mt = max(0.0, float(rng.lognormal(6.0, 0.9)) * (1 - 0.12 * conflict_score))
        area_cultivated_ha = max(0.1, float(rng.lognormal(7.0, 0.8)))
        yield_kg_ha = crop_production_mt * 1000 / area_cultivated_ha

        # market indicators
        maize_price_naira_kg = max(100.0, float(rng.normal(350, 80)) * (1 + 0.1 * conflict_score))
        sorghum_price_naira_kg = max(80.0, float(rng.normal(300, 70)))
        rice_price_naira_kg = max(200.0, float(rng.normal(600, 120)))
        wheat_price_naira_kg = max(150.0, float(rng.normal(500, 100)))
        price_index = (maize_price_naira_kg + sorghum_price_naira_kg + rice_price_naira_kg) / 3

        # household indicators
        pop_total = max(500, int(rng.lognormal(9.5, 1.1)))
        pct_food_insecure = float(np.clip(
            0.1 + 0.08 * conflict_score + rng.normal(0, 0.08) - 0.002 * ndvi * 10,
            0.02, 0.95
        ))
        stunting_pct = float(np.clip(rng.normal(30, 10) + 3 * conflict_score, 5, 70))
        wasting_pct = float(np.clip(rng.normal(8, 4) + 2 * conflict_score, 1, 40))
        global_acute_malnutrition_pct = wasting_pct * 1.15
        dietary_diversity_score = float(np.clip(rng.normal(4.5, 1.2) - 0.3 * conflict_score, 1, 9))
        market_access_km = max(0.5, float(rng.exponential(18)))
        displacement_pct = float(np.clip(rng.exponential(0.03) * conflict_score, 0, 0.6))

        # IPC phase derived from composite vulnerability
        vulnerability = (
            pct_food_insecure * 0.4 +
            (wasting_pct / 40) * 0.3 +
            (conflict_score / 4) * 0.2 +
            (1 - ndvi) * 0.1
        )
        if vulnerability < 0.15:
            ipc_phase = 1
        elif vulnerability < 0.30:
            ipc_phase = 2
        elif vulnerability < 0.50:
            ipc_phase = 3
        elif vulnerability < 0.70:
            ipc_phase = 4
        else:
            ipc_phase = 5

        records.append({
            "lga_id": lga_id, "state": state, "year": year, "season": season,
            "latitude": round(lat + rng.normal(0, 0.5), 6),
            "longitude": round(lon + rng.normal(0, 0.5), 6),
            "conflict_level": conflict, "conflict_score": conflict_score,
            "rainfall_mm": round(rainfall_mm, 1), "ndvi": round(ndvi, 4),
            "crop_production_mt": round(crop_production_mt, 1),
            "area_cultivated_ha": round(area_cultivated_ha, 1),
            "yield_kg_ha": round(yield_kg_ha, 1),
            "maize_price_naira_kg": round(maize_price_naira_kg, 2),
            "sorghum_price_naira_kg": round(sorghum_price_naira_kg, 2),
            "rice_price_naira_kg": round(rice_price_naira_kg, 2),
            "wheat_price_naira_kg": round(wheat_price_naira_kg, 2),
            "price_index": round(price_index, 2),
            "population_total": pop_total,
            "pct_food_insecure": round(pct_food_insecure, 4),
            "stunting_pct": round(stunting_pct, 2),
            "wasting_pct": round(wasting_pct, 2),
            "global_acute_malnutrition_pct": round(global_acute_malnutrition_pct, 2),
            "dietary_diversity_score": round(dietary_diversity_score, 2),
            "market_access_km": round(market_access_km, 1),
            "displacement_pct": round(displacement_pct, 4),
            "ipc_phase": ipc_phase,
        })

    return pd.DataFrame(records)


def save_dataset(output_dir: str | Path = "data/raw") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_food_security_dataset()
    path = output_dir / "food_security_data.csv"
    df.to_csv(path, index=False)
    return path
