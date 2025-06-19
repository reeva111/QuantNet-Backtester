import pandas as pd

class MovingAverageCrossoverStrategy:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["SMA_Short"] = df["Close"].rolling(window=self.short_window).mean()
        df["SMA_Long"] = df["Close"].rolling(window=self.long_window).mean()
        df["Signal"] = 0

        crossover_range = df.index[self.long_window:]
        df.loc[crossover_range, "Signal"] = (
            df.loc[crossover_range, "SMA_Short"] > df.loc[crossover_range, "SMA_Long"]
        ).astype(int)

        df["Position"] = df["Signal"].diff()
        return df
