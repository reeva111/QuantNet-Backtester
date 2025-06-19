import pandas as pd
import joblib
from .ml_features import create_features

class MLSignalStrategy:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        X, y, feature_df = create_features(df)

        # Get prediction probabilities
        probs = self.model.predict_proba(X)
        buy_conf = probs[:, 1]  # Probability of class '1' (up)
        
        # Apply confidence filter
        signal = []
        for p in buy_conf:
            if p >= 0.7:
                signal.append(1)
            elif p <= 0.3:
                signal.append(0)  # Optionally use -1 for shorting
            else:
                signal.append(None)  # Skip low-confidence predictions

        feature_df["Signal"] = signal
        feature_df["Signal"] = feature_df["Signal"].fillna(method='ffill').fillna(0).astype(int)
        feature_df["Position"] = feature_df["Signal"].diff().fillna(0)

        return feature_df
