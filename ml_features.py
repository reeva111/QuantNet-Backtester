# backtest_engine/strategies/ml_features.py

import pandas as pd
import numpy as np
import ta

def create_features(df):
    df = df.copy()
    df["Return_1D"] = df["Close"].pct_change()
    df["Return_3D"] = df["Close"].pct_change(3)
    df["Volatility"] = df["Return_1D"].rolling(5).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"].squeeze(), window=14).rsi()


    # Drop rows with NaNs from rolling calculations
    df.dropna(inplace=True)

    # Define target: 1 if next day return > 0 else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    features = ["Return_1D", "Return_3D", "Volatility", "Volume_Change", "RSI"]
    X = df[features]
    y = df["Target"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y.loc[X.index]  

    return X, y, df.loc[X.index]  # filtered full_df

