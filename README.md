# 🌍 Global Market Crash Prediction

> Predicts the probability of a US stock market crash in the next 63 trading days using institutional-grade macro indicators.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Railway](https://img.shields.io/badge/Deployed-Railway-purple)](https://railway.app)
[![AUC](https://img.shields.io/badge/AUC--ROC-0.8613-orange)]()

---

## 🚀 Live API

**Base URL:** https://global-market-crash-prediction-production.up.railway.app

| Endpoint | Method | Description |
|---|---|---|
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | API health check |
| `/predict` | POST | Crash probability prediction |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | LightGBM (Tuned) |
| AUC-ROC | **0.8613** |
| Accuracy | 0.84 |
| Crash Recall | 0.64 |
| Train Period | 2006–2022 |
| Test Period | 2022–2026 (unseen) |

> Institutional quant funds consider AUC 0.60–0.75 as a strong live trading signal.
> This model achieves 0.86 — driven by institutional-grade features not typically included in academic models.

---

## 🔍 Feature Set (22 Institutional-Grade Indicators)

| Category | Features | Why It Matters |
|---|---|---|
| Market Returns | SP500_Return_5d, 21d, 63d | Momentum across timeframes |
| Volatility | VIX | Fear gauge — coincident crash signal |
| Safe Haven | Gold_Return_5d, 21d, 63d | Gold rises during market fear |
| Commodity | Oil_Return_5d, 21d, 63d | Supply/demand crash signal |
| Yield Curve | T10Y, Yield_Spread | Inversion = leading recession indicator |
| Macro | CPI, Unemployment, GDP_Growth, Gov_Expenditure | Economic regime context |
| Credit | Credit_Spread | Institutional panic before equity crashes |
| Valuation | CAPE | Shiller P/E — overvaluation signal |
| Labor | Sahm_Rule | Most accurate recession trigger |
| Regime | Oil_Price_Ratio, Gold_Price_Ratio, SP500_Price_Ratio | Price relative to 1-year history |

---

## 📈 Crash Definition

A crash is labeled when the **forward 63-day S&P 500 return drops below -10%**.

**6 Major Crash Periods Identified (2006–2026):**

| Period | Event |
|---|---|
| 2008 | Global Financial Crisis |
| 2011 | European Debt Crisis |
| 2015–2016 | China Slowdown |
| 2018 | Q4 Correction |
| 2020 | COVID Crash |
| 2022 | Rate Hike Crash |

**Class Balance:** 94% No Crash / 6% Crash
→ Handled with `scale_pos_weight=12.78` in LightGBM

---

## 🔬 Methodology & Key Decisions

### No Data Leakage
- **Chronological 80/20 train/test split** — no random shuffling
- Training: 2006–2022 | Testing: 2022–2026
- `merge_asof` with `direction="backward"` for all FRED macro data — prevents future data from leaking into past rows

### Feature Engineering
- **Price ratios** (e.g., `SP500_Price / rolling_252d_mean`) — captures regime information without the trending price level problem
- **Sahm Rule** — computed from rolling unemployment averages, triggers at +0.5% above 12-month low
- **Yield Spread** (T10Y - T2Y) — directly captures yield curve inversion signal

### Class Imbalance
- AUC-ROC used instead of accuracy (94% no-crash baseline makes accuracy meaningless)
- `scale_pos_weight = total_negatives / total_positives = 12.78`

---

## 🏗️ Project Structure

```
global-market-crash-prediction/
├── app/
│   ├── api.py            # FastAPI endpoints
│   ├── main.py           # App entry point
│   └── schemas.py        # Input/output schemas
├── config/
│   ├── config.py         # Path and config loader
│   └── config.yml        # Feature names, model config
├── processing/
│   ├── data_manager.py   # Model load/save
│   └── features.py       # Input transformation
├── trained_model/
│   ├── final_lgbm.pkl    # Trained LightGBM model
│   └── feature_names.pkl # Feature order
├── charts/               # EDA visualizations
├── research.ipynb        # Full research notebook
├── train_pipeline.py     # Model training script
├── predict.py            # Prediction script
├── Dockerfile            # Docker deployment
├── Procfile              # Railway start command
└── requirements.txt      # Dependencies
```

---

## 📡 API Usage

```python
import requests

response = requests.post(
    "https://global-market-crash-prediction-production.up.railway.app/predict",
    json={
        "SP500_Return_5d": -0.02,
        "SP500_Return_21d": -0.08,
        "SP500_Return_63d": -0.15,
        "VIX": 35.0,
        "Gold_Return_63d": 0.05,
        "Gold_Return_21d": 0.02,
        "Gold_Return_5d": 0.01,
        "Oil_Return_63d": -0.10,
        "Oil_Return_21d": -0.05,
        "Oil_Return_5d": -0.02,
        "T10Y": 3.5,
        "CPI": 280.0,
        "Unemployment": 4.5,
        "Yield_Spread": -0.5,
        "GDP_Growth": -1.5,
        "Gov_Expenditure": 6500.0,
        "Credit_Spread": 5.5,
        "CAPE": 32.0,
        "Sahm_Rule": 0.6,
        "Oil_Price_Ratio": 0.85,
        "Gold_Price_Ratio": 1.15,
        "SP500_Price_Ratio": 0.90
    }
)

print(response.json())
# {"crash_probability": 0.87, "crash_predicted": 1, "risk_level": "HIGH"}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | LightGBM, scikit-learn |
| Data | FRED API, yfinance, Shiller Yale Dataset |
| API | FastAPI, Uvicorn |
| Deployment | Railway, Docker |
| Language | Python 3.11 |

---

## 👤 About the Author

**Abdallah Hashad**
Quantitative Finance × Machine Learning

```python
class AbdallahHashad:
    role        = "Machine Learning Engineer"
    location    = "Cairo, Egypt 🇪🇬"
    edge        = "CFA-level valuation + Data Science"

    def building(self):
        return ["end-to-end ML pipelines",
                "tuned gradient boosting models",
                "production FastAPI endpoints"]

    def current_focus(self):
        return {"learning": ["Hands-On ML with Géron"],
                "goal": "models that answer real financial questions"}
```

🌐 Portfolio: [abdallah-hashad.vercel.app](https://abdallah-hashad.vercel.app)
📊 Kaggle: [abdallahhashad0](https://www.kaggle.com/abdallahhashad0)
✍️ Medium: [@abdallahhashad029](https://medium.com/@abdallahhashad029)
🐙 GitHub: [abdallahh07](https://github.com/abdallahh07)
