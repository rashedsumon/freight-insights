# models.py
# Light-weight implementations of ETA, Price, and Capacity models.
# Replace with heavier production pipelines (batch training, feature stores, xgboost/lightgbm, etc.)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _prepare_xy_for_price(df: pd.DataFrame):
    # Minimal feature engineer: origin, destination, product -> price
    X = df[["origin", "destination", "product"]].astype(str)
    y = df["price_usd"].astype(float)
    return X, y


class PriceEstimator:
    def __init__(self):
        self.model = None
        self.pipeline = None

    def fit(self, df: pd.DataFrame):
        if df.empty or "price_usd" not in df.columns:
            logger.info("PriceEstimator: no data to train")
            return
        X, y = _prepare_xy_for_price(df.dropna(subset=["price_usd"]))
        # simple pipeline
        self.pipeline = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ("lr", LinearRegression())
        ])
        self.pipeline.fit(X.to_dict(orient="records"), y)
        logger.info("Trained PriceEstimator")

    def predict(self, origin, destination, product):
        if self.pipeline is None:
            # fallback: global mean
            return 1000.0
        X = pd.DataFrame([{"origin": origin, "destination": destination, "product": product}])
        # scikit OneHotEncoder expects 2D array-like
        try:
            # The pipeline expects array-like dictionaries if fit that way — adapt
            preds = self.pipeline.predict(X.to_dict(orient="records"))
            return float(preds[0])
        except Exception:
            # fallback coarse rule
            return float(self.pipeline.named_steps["lr"].intercept_) if hasattr(self.pipeline.named_steps["lr"], "intercept_") else 1000.0

    def predict_rate_change_pct(self, origin, destination, product):
        # Placeholder simple heuristic: random small movement
        return round(np.random.normal(0.0, 2.0), 2)  # percent


class ETAEstimator:
    def __init__(self):
        self.model = None

    def fit(self, df: pd.DataFrame):
        # We'll train a simple regressor for transit days from departed_at to arrived_at
        if df.empty or not {"departed_at", "arrived_at"}.issubset(df.columns):
            return
        df = df.dropna(subset=["departed_at", "arrived_at"])
        df["transit_days"] = (pd.to_datetime(df["arrived_at"]) - pd.to_datetime(df["departed_at"])).dt.days
        X = df[["origin", "destination", "product"]].astype(str)
        y = df["transit_days"].astype(float)
        self.pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                                  ("rf", RandomForestRegressor(n_estimators=50, random_state=42))])
        # fit using dict records
        self.pipeline.fit(X.to_dict(orient="records"), y)
        logger.info("Trained ETAEstimator")

    def predict_delay_days(self, origin, destination, product):
        # Predict transit days and compare to nominal transit (simple)
        try:
            X = pd.DataFrame([{"origin": origin, "destination": destination, "product": product}])
            predicted_days = self.pipeline.predict(X.to_dict(orient="records"))[0]
            # baseline expected transit (heuristic)
            baseline = max(1.0, predicted_days - np.random.uniform(-1, 1))
            delay = max(0.0, predicted_days - baseline)
            return float(round(delay, 2))
        except Exception:
            return 0.0


class CapacityForecaster:
    def __init__(self):
        self.trend = None

    def fit(self, df: pd.DataFrame):
        # Simple weekly average capacity by lane/product
        if df.empty or "capacity_pct" not in df.columns:
            return
        # group by lane-week
        df = df.copy()
        if "booked_at" in df.columns:
            df["week"] = pd.to_datetime(df["booked_at"]).dt.to_period("W").apply(lambda r: r.start_time)
            grp = df.groupby(["origin", "destination", "product", "week"])["capacity_pct"].mean().reset_index()
            self.trend = grp
            logger.info("Fitted capacity trend")

    def forecast(self, origin, destination, product, periods=4):
        # naive forecast: take last known avg and repeat with random small noise
        if self.trend is None or self.trend.empty:
            # default mid-capacity
            return [60.0 + np.random.uniform(-5, 5) for _ in range(periods)]
        subset = self.trend[
            (self.trend["origin"].str.upper() == origin.upper()) &
            (self.trend["destination"].str.upper() == destination.upper()) &
            (self.trend["product"].str.lower() == product.lower())
        ]
        if subset.empty:
            # use lane-average if product exact missing
            lane = self.trend[
                (self.trend["origin"].str.upper() == origin.upper()) &
                (self.trend["destination"].str.upper() == destination.upper())
            ]
            base = lane["capacity_pct"].mean() if not lane.empty else self.trend["capacity_pct"].mean()
        else:
            base = subset.sort_values("week")["capacity_pct"].iloc[-1]
        return [float(round(base + np.random.uniform(-3, 3), 1)) for _ in range(periods)]
