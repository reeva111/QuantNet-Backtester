# main.py

import pandas as pd
import matplotlib.pyplot as plt
from backtest_engine.strategies.ma_crossover import MovingAverageCrossoverStrategy
from backtest_engine.strategies.bollinger_reversion import BollingerReversionStrategy
from backtest_engine.strategies import MLSignalStrategy
from backtest_engine.backtester import Backtester
import yfinance as yf


df = yf.download("RELIANCE.NS", start="2010-01-01", end="2025-01-01", auto_adjust=True)

df.columns = df.columns.get_level_values(0)

if "Adj Close" in df.columns:
    df["Close"] = df["Adj Close"]
elif "Close" in df.columns:
    df["Close"] = df["Close"]
else:
    raise ValueError("No 'Close' or 'Adj Close' found.")


df["Close"] = df["Close"].astype(float).ffill()

def evaluate_strategy(df):
    df = df.copy()
    df["Daily_Return"] = df["Total"].pct_change().fillna(0)

    cum_return = df["Total"].iloc[-1] / df["Total"].iloc[0] - 1
    sharpe = (df["Daily_Return"].mean() / df["Daily_Return"].std()) * (252 ** 0.5)

    drawdown = (df["Total"] / df["Total"].cummax() - 1).min()

    return {
        "Cumulative Return": round(cum_return * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown": round(abs(drawdown) * 100, 2)
    }


strategies = [
    ("MA Crossover", MovingAverageCrossoverStrategy(short_window=5, long_window=10)),
    ("Bollinger Reversion", BollingerReversionStrategy(window=20, entry_z=2.5, exit_z=1.0)),
    ("ML Signal", MLSignalStrategy(model_path="models/xgb_model.pkl")),
]

results = []
equity_curves = {}
signals_dict = {}      
results_df_dict = {}    

for name, strategy in strategies:
    print(f"\nRunning strategy: {name}")
    signal_df = strategy.generate_signals(df)
    signals_dict[name] = signal_df.copy()       # <--- SAVE signals
    print(f"{name} signal summary:\n", signal_df["Signal"].value_counts())
    print(f"{name} position summary:\n", signal_df["Position"].value_counts())

    bt = Backtester(signal_df, initial_capital=100000, units=100)
    result_df = bt.run()
    results_df_dict[name] = result_df.copy()    # <--- SAVE result_df

    equity_curves[name] = result_df["Total"]
    print(f"{name} final portfolio value:", result_df['Total'].iloc[-1])
    print(f"{name} total trades:", signal_df['Position'].abs().sum())

    metrics = evaluate_strategy(result_df)
    metrics["name"] = name
    results.append(metrics)

results_df = pd.DataFrame(results).set_index("name")
print("\n Strategy Comparison:\n", results_df)


# Optional: Smooth curves using a rolling average
window_size = 10  # adjust for more/less smoothing

import matplotlib.dates as mdates

fig, axes = plt.subplots(len(strategies), 1, figsize=(14, 10), sharex=True)
fig.suptitle("Equity Curves with Buy/Sell Markers (Separate Subplots)", fontsize=16)

colors = {
    "MA Crossover": "blue",
    "Bollinger Reversion": "orange",
    "ML Signal": "pink"
}

for i, (name, _) in enumerate(strategies):
    ax = axes[i]
    equity = equity_curves[name]
    strategy_df = strategies[[n for n, _ in strategies].index(name)][1].generate_signals(df)

    # Plot the main equity curve
    ax.plot(equity.index, equity, label=name, color=colors[name], linewidth=1.5)

    # Extract only **entry points**
    buy_signals = strategy_df[(strategy_df["Position"] == 1.0) & (strategy_df["Position"].shift(1) != 1.0)]
    sell_signals = strategy_df[(strategy_df["Position"] == -1.0) & (strategy_df["Position"].shift(1) != -1.0)]

    # Plot Buy/Sell
    ax.scatter(buy_signals.index, equity.loc[buy_signals.index],
               marker="^", color="green", label="Buy", alpha=0.7)

    ax.scatter(sell_signals.index, equity.loc[sell_signals.index],
               marker="v", color="red", label="Sell", alpha=0.7)

    # Label & Title
    ax.set_title(f"{name} Strategy", fontsize=12)
    ax.set_ylabel("Portfolio Value")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)

# Date formatting
axes[-1].set_xlabel("Date")
plt.xticks(rotation=45)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(1))

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Room for suptitle
plt.show()
