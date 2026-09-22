"""
Machine Learning Strategy - Random Forest for price direction prediction.
Trains a classifier on technical features extracted from historical candles.
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from config.settings import settings, PROJECT_ROOT
from src.indicators import technical as ta
from src.utils.logger import log
from src.utils.helpers import to_json_safe, save_json
from src.strategies.base import BaseStrategy, Signal

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "rf_direction.pkl"


class MLStrategy(BaseStrategy):
    name = "ml_random_forest"
    weight = 0.8  # lower weight - ML can be noisy

    def __init__(self, weight: float = None):
        super().__init__(weight)
        self.model = None
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                log.info("[green]ML model loaded[/] from data/models/rf_direction.pkl")
            except Exception as e:
                log.warning(f"Failed to load ML model: {e}")
                self.model = None
        else:
            log.info("[yellow]No ML model found[/]. Run `python scripts/train_ml_model.py` to train one.")

    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from OHLCV data."""
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI
        df["rsi"] = ta.rsi(close, 14)
        # MACD
        macd_df = ta.macd(close, 12, 26, 9)
        df["macd"] = macd_df["macd"]
        df["macd_signal"] = macd_df["signal"]
        df["macd_hist"] = macd_df["histogram"]
        # BB
        bb = ta.bollinger_bands(close, 20, 2)
        df["bb_pct"] = bb["percent_b"]
        df["bb_width"] = bb["bandwidth"]
        # EMAs
        df["ema_9"] = ta.ema(close, 9)
        df["ema_21"] = ta.ema(close, 21)
        df["ema_50"] = ta.ema(close, 50)
        df["ema_diff_9_21"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"]
        df["ema_diff_21_50"] = (df["ema_21"] - df["ema_50"]) / df["ema_50"]
        # ATR
        df["atr"] = ta.atr(high, low, close, 14)
        df["atr_pct"] = df["atr"] / close
        # ROC
        df["roc_5"] = ta.roc(close, 5)
        df["roc_12"] = ta.roc(close, 12)
        # Stochastic
        stoch = ta.stochastic(high, low, close, 14, 3)
        df["stoch_k"] = stoch["k"]
        df["stoch_d"] = stoch["d"]
        # ADX
        adx_df = ta.adx(high, low, close, 14)
        df["adx"] = adx_df["adx"]
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]
        # Volume features
        df["vol_sma"] = volume.rolling(20).mean()
        df["vol_ratio"] = volume / df["vol_sma"]
        # CCI
        df["cci"] = ta.cci(high, low, close, 20)
        # Returns
        df["ret_1"] = close.pct_change(1)
        df["ret_3"] = close.pct_change(3)
        df["ret_6"] = close.pct_change(6)
        df["ret_12"] = close.pct_change(12)

        return df

    @staticmethod
    def build_labels(df: pd.DataFrame, horizon: int = 6,
                     threshold: float = 0.01) -> pd.Series:
        """
        Label: 1 (bullish) if future return > +threshold,
              -1 (bearish) if < -threshold,
               0 (neutral) otherwise.
        """
        future_return = df["close"].shift(-horizon) / df["close"] - 1
        labels = pd.Series(
            np.where(future_return > threshold, 1,
                     np.where(future_return < -threshold, -1, 0)),
            index=df.index
        )
        return labels

    def train(self, df: pd.DataFrame, symbol_for_log: str = "aggregate") -> dict:
        """Train a Random Forest classifier on the given OHLCV data."""
        log.info(f"[cyan]Training ML model[/] on {symbol_for_log} ({len(df)} bars)")

        features_df = self.build_features(df)
        features_df["label"] = self.build_labels(features_df, horizon=6, threshold=0.01)
        features_df = features_df.dropna()

        feature_cols = [
            "rsi", "macd", "macd_signal", "macd_hist", "bb_pct", "bb_width",
            "ema_diff_9_21", "ema_diff_21_50", "atr_pct", "roc_5", "roc_12",
            "stoch_k", "stoch_d", "adx", "plus_di", "minus_di", "vol_ratio",
            "cci", "ret_1", "ret_3", "ret_6", "ret_12"
        ]
        X = features_df[feature_cols]
        y = features_df["label"]

        if len(X) < 100 or y.nunique() < 2:
            log.warning(f"Insufficient data for ML training (n={len(X)})")
            return {"status": "failed", "reason": "insufficient_data"}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        log.info(f"[green]ML model trained[/] - Test accuracy: {acc:.4f}")

        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        log.info(f"Model saved to {MODEL_PATH}")

        # Feature importance
        importance = dict(zip(feature_cols, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))
        log.info(f"Top features: {list(importance.items())[:5]}")

        return {
            "status": "success",
            "accuracy": float(acc),
            "samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "feature_importance": importance,
            "classification_report": report,
        }

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if self.model is None:
            return self._neutral("ML model not trained (run train_ml_model.py)")

        if len(df) < 60:
            return self._neutral("Insufficient data for ML")

        try:
            features_df = self.build_features(df)
            last_features = features_df.iloc[[-1]]
            feature_cols = [
                "rsi", "macd", "macd_signal", "macd_hist", "bb_pct", "bb_width",
                "ema_diff_9_21", "ema_diff_21_50", "atr_pct", "roc_5", "roc_12",
                "stoch_k", "stoch_d", "adx", "plus_di", "minus_di", "vol_ratio",
                "cci", "ret_1", "ret_3", "ret_6", "ret_12"
            ]
            X = last_features[feature_cols]
            if X.isna().any().any():
                return self._neutral("NaN features")
            pred = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            classes = self.model.classes_.tolist()
            proba_dict = {int(c): float(p) for c, p in zip(classes, proba)}

            bull_prob = proba_dict.get(1, 0)
            bear_prob = proba_dict.get(-1, 0)
            net = (bull_prob - bear_prob) * 100  # -100..100

            details = {
                "prediction": pred,
                "probabilities": proba_dict,
                "bull_probability": bull_prob,
                "bear_probability": bear_prob,
            }

            if pred == 1 and bull_prob > 0.5:
                return self._bull(net, f"ML predicts UP ({bull_prob*100:.1f}%)", details)
            elif pred == -1 and bear_prob > 0.5:
                return self._bear(abs(net), f"ML predicts DOWN ({bear_prob*100:.1f}%)", details)
            return self._neutral(f"ML neutral (bull={bull_prob:.2f}, bear={bear_prob:.2f})", details)
        except Exception as e:
            log.error(f"ML analyze error: {e}")
            return self._neutral(f"ML error: {e}")
