import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime
from pipeline import create_pipeline
from processing.data_manager import save_model
from config.config import FEATURES, TARGET, DROP_FEATURES
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path


def run_training():
    print("Starting training pipeline...")

    # ---- 1. Date range ----
    start = datetime(2006, 1, 1)
    end = datetime(2026, 7, 11)

    # ---- 2. Pull data ----
    print("Pulling data...")
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(start=start, end=end)["Close"].reset_index()
    sp500_data.columns = ["Date", "SP500_Price"]
    sp500_data["SP500_Return_5d"] = sp500_data["SP500_Price"].pct_change(5)
    sp500_data["SP500_Return_21d"] = sp500_data["SP500_Price"].pct_change(21)
    sp500_data["SP500_Return_63d"] = sp500_data["SP500_Price"].pct_change(63)

    vix = pdr.get_data_fred("VIXCLS", start, end).reset_index()
    vix.columns = ["Date", "VIX"]

    xau = yf.Ticker("GC=F")
    gold = xau.history(start=start, end=end)["Close"].reset_index()
    gold.columns = ["Date", "Gold_Price"]
    gold["Gold_Return_5d"] = gold["Gold_Price"].pct_change(5)
    gold["Gold_Return_21d"] = gold["Gold_Price"].pct_change(21)
    gold["Gold_Return_63d"] = gold["Gold_Price"].pct_change(63)

    oil = yf.Ticker("CL=F")
    oil_data = oil.history(start=start, end=end)["Close"].reset_index()
    oil_data.columns = ["Date", "Oil_Price"]
    oil_data["Oil_Return_5d"] = oil_data["Oil_Price"].pct_change(5)
    oil_data["Oil_Return_21d"] = oil_data["Oil_Price"].pct_change(21)
    oil_data["Oil_Return_63d"] = oil_data["Oil_Price"].pct_change(63)

    t10 = pdr.get_data_fred("GS10", start, end).reset_index()
    t10.columns = ["Date", "T10Y"]

    t2 = pdr.get_data_fred("GS2", start, end).reset_index()
    t2.columns = ["Date", "T2Y"]

    fed = pdr.get_data_fred("FEDFUNDS", start, end).reset_index()
    fed.columns = ["Date", "Fed_Rate"]

    cpi = pdr.get_data_fred("CPIAUCSL", start, end).reset_index()
    cpi.columns = ["Date", "CPI"]

    unemp = pdr.get_data_fred("UNRATE", start, end).reset_index()
    unemp.columns = ["Date", "Unemployment"]

    gdp_growth = pdr.get_data_fred("A191RL1Q225SBEA", start, end).reset_index()
    gdp_growth.columns = ["Date", "GDP_Growth"]
    gdp_growth["GDP_Growth"] = gdp_growth["GDP_Growth"].shift(1)

    gov_exp = pdr.get_data_fred("FGEXPND", start, end).reset_index()
    gov_exp.columns = ["Date", "Gov_Expenditure"]
    gov_exp["Gov_Expenditure"] = gov_exp["Gov_Expenditure"].shift(1)

    credit_spread = pdr.get_data_fred("TEDRATE", start, end).reset_index()
    credit_spread.columns = ["Date", "Credit_Spread"]

    # CAPE from Shiller
    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    shiller = pd.read_excel(url, sheet_name="Data", skiprows=7, engine="xlrd")
    cape = shiller[["Date", "CAPE"]].copy()
    cape = cape.dropna(subset=["Date", "CAPE"])
    cape["Date"] = pd.to_datetime(
        cape["Date"].apply(lambda x: f"{int(x)}-{round((x % 1) * 100):02d}"),
        format="%Y-%m"
    )
    cape = cape[cape["Date"] >= pd.Timestamp("2006-01-01")].copy()
    cape.sort_values("Date", inplace=True)

    # ---- 3. Clean timezones ----
    all_dfs = [sp500_data, vix, gold, oil_data, t10, t2,
               fed, cpi, unemp, gdp_growth, gov_exp, credit_spread, cape]
    for df in all_dfs:
        df["Date"] = pd.to_datetime(df["Date"])
        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
        df.sort_values("Date", inplace=True)

    # ---- 4. Merge ----
    print("Merging data...")
    master = sp500_data.copy()
    for df in [vix, gold, oil_data, t10, t2, fed, cpi, unemp,
               gdp_growth, gov_exp, credit_spread, cape]:
        master = pd.merge_asof(master, df, on="Date", direction="backward")

    master = master.dropna(subset=["CPI", "Unemployment"])
    master["Yield_Spread"] = master["T10Y"] - master["T2Y"]

    # ---- 5. Derived features ----
    master["Unemployment_3m"] = master["Unemployment"].rolling(3).mean()
    master["Unemployment_12m_low"] = master["Unemployment"].rolling(12).min()
    master["Sahm_Rule"] = master["Unemployment_3m"] - master["Unemployment_12m_low"]
    master["Oil_Price_Ratio"] = master["Oil_Price"] / master["Oil_Price"].rolling(252).mean()
    master["Gold_Price_Ratio"] = master["Gold_Price"] / master["Gold_Price"].rolling(252).mean()
    master["SP500_Price_Ratio"] = master["SP500_Price"] / master["SP500_Price"].rolling(252).mean()

    # ---- 6. Target ----
    master["future_return"] = master["SP500_Price"].pct_change(63).shift(-63)
    master["crash"] = (master["future_return"] < -0.10).astype(int)
    master = master.dropna(subset=["future_return"])
    master = master.dropna()

    # ---- 7. Features and target ----
    X = master[FEATURES]
    y = master[TARGET]

    # ---- 8. Chronological split ----
    split_idx = int(len(master) * 0.80)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # ---- 9. Train ----
    print("Training model...")
    total_negatives = (y_train == 0).sum()
    total_positives = (y_train == 1).sum()
    scale_pos_weight = total_negatives / total_positives

    model = create_pipeline(scale_pos_weight)
    model.fit(X_train, y_train)

    # ---- 10. Evaluate ----
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"Test AUC: {auc:.4f}")

    # ---- 11. Save ----
    save_model(model, "final_lgbm.pkl")
    joblib.dump(X_train.columns.tolist(), 
                Path("trained_model") / "feature_names.pkl")
    joblib.dump(scale_pos_weight, 
                Path("trained_model") / "scale_pos_weight.pkl")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    run_training()