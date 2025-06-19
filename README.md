# QuantNet-Backtester
Multi-strategy, ML-powered backtesting engine for Indian markets


QuantNet is a research-grade backtesting engine built to simulate and compare different trading strategies on Indian market data (NIFTY50). It supports technical strategies like Moving Average Crossover and Bollinger Band Reversion, as well as machine learning-based signal generation.

This project was designed to mimic the real-world trading environment with capital tracking, performance metrics, and strategy comparison visuals.

---

##  Features

- **Data Source**: Yahoo Finance (NSE Index)
- **Strategies**:
  - Moving Average Crossover
  - Bollinger Reversion
  - ML Signal (XGBoost-based momentum model)
- **Evaluation Metrics**:
  - Cumulative Return
  - Sharpe Ratio
  - Maximum Drawdown
- **Confidence Filtering**: Model trades only when probability is high
- **Trade Logging**: CSV export of every executed trade
- **Visualization**:
  - Equity curves
  - Buy/Sell markers on portfolio value chart
  - Feature importance plot for ML model

---

##  Model Overview

The ML strategy is trained using technical indicators:
- RSI
- Volatility
- Volume change
- 1-day and 3-day returns

The classifier predicts the likelihood of a price going up. Only high-confidence predictions are converted into trade signals.

---

## Screenshots

### Equity Curve with Buy/Sell Markers

![Equity Curve](images/equity_curve.png)

### Feature Importance

![Feature Importance](images/feature_importance.png)

---

##  How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train ML model
python train_model.py

# Run strategies + backtest
python main.py
