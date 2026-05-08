
# Food Security & Nutrition Monitoring Dashboard
IPC Phase classification and early warning system for 16 Northern Nigerian states. Integrates crop production, market price data, malnutrition rates, and conflict indicators to track food insecurity across LGAs and flag emergency zones requiring humanitarian response.

---

## Features

- Random Forest IPC Phase 1–5 classifier (Minimal → Famine)
- 3,000 LGA-season records across 2018–2024
- Market price tracking: maize, sorghum, rice, wheat
- Multi-year trend lines for IPC phase, crop yield, and price index
- Conflict-food security correlation analysis

## Project Structure

```
food-security-dashboard/
├── src/
│   ├── data_generator.py   # Synthetic LGA dataset (3,000 observations)
│   └── model.py            # Random Forest IPC phase classifier
├── streamlit_app.py        # 4-page Streamlit dashboard
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dataset

Synthetic dataset of 3,000 LGA-season records covering Borno, Yobe, Adamawa, Zamfara, Katsina, Sokoto, Kebbi, Niger, Bauchi, Gombe, Jigawa, Kano, Plateau, Nassarawa, Benue, and Taraba states (2018–2024).

## Tech Stack

| Layer | Library |
|---|---|
| Dashboard | Streamlit |
| ML Model | scikit-learn RandomForestClassifier |
| Visualisation | Plotly Express + Mapbox |
| Data | NumPy / Pandas synthetic generator |

---

*Dataset is synthetic and generated for demonstration purposes.*
