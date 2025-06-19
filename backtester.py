import pandas as pd

class Backtester:
    def __init__(self, df: pd.DataFrame, initial_capital=100000, units=1):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.units = units

    def run(self) -> pd.DataFrame:
        df = self.df.copy()
        df["Close"] = df["Close"].astype(float).ffill()
        df["Signal"] = df["Signal"].fillna(0)

        cash = self.initial_capital
        position = 0
        portfolio_value = []

        for i in range(len(df)):
            signal = df["Signal"].iloc[i]
            price = df["Close"].iloc[i]

            # Execute trades
            if signal == 1 and position == 0:
                position = self.units
                cash -= self.units * price
            elif signal == -1 and position > 0:
                cash += position * price
                position = 0

            total = cash + position * price
            portfolio_value.append(total)

        df["Total"] = portfolio_value
        df["Returns"] = df["Total"].pct_change().fillna(0)
        return df
