import pandas as pd

class BollingerReversionStrategy:
    def __init__(self, window=20, entry_z=1.0, exit_z=0.3):
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ✅ Extract Close as a clean Series
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        # 🧼 Force 'Close' to be 1D, float, forward-filled
        df["Close"] = pd.Series(df["Close"]).astype(float).ffill()

        # Rolling mean and std
        df["rolling_mean"] = df["Close"].rolling(window=self.window).mean()
        df["rolling_std"] = df["Close"].rolling(window=self.window).std()

        # ✅ Z-score (both numerator & denominator are Series)
        df["z_score"] = (df["Close"] - df["rolling_mean"]) / df["rolling_std"]

        # Signal logic
        df["Signal"] = 0
        df.loc[df["z_score"] < -self.entry_z, "Signal"] = 1
        df.loc[df["z_score"] > self.entry_z, "Signal"] = -1
        df.loc[abs(df["z_score"]) < self.exit_z, "Signal"] = 0
        df["Position"] = df["Signal"].diff().fillna(0)

        return df
