import pandas as pd
import yfinance as yf
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier, plot_importance

from backtest_engine.strategies.ml_features import create_features

df = yf.download("RELIANCE.NS", start="2010-01-01", end="2025-01-01", auto_adjust=True)


if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ✅ Ensure Close is set
if "Adj Close" in df.columns:
    df["Close"] = df["Adj Close"]
elif "Close" in df.columns:
    df["Close"] = df["Close"]
else:
    raise ValueError("No 'Close' or 'Adj Close' found.")

df["Close"] = df["Close"].astype(float).ffill()

# Extract features and labels
X, y, full_df = create_features(df)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)


model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)


print("Model Performance on Test Set:")
print(classification_report(y_test, model.predict(X_test)))


os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")

# Plot feature importance
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='gain', max_num_features=10)
plt.title("Feature Importance (by Gain)")
plt.grid(True)
plt.tight_layout()
plt.show()
