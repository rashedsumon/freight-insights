# insights_engine.py
# Encapsulates logic to create "insights" given booking fields.
# Uses simple ML models (from models.py) and rules. In production, wire up real-time model serving.

import pandas as pd
import numpy as np
from models import ETAEstimator, PriceEstimator, CapacityForecaster
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InsightsEngine:
    def __init__(self, dataframes: dict):
        """
        dataframes: dict with keys - shipments, rates, carriers, capacity, external_market
        """
        self.shipments = dataframes.get("shipments", pd.DataFrame())
        self.rates = dataframes.get("rates", pd.DataFrame())
        self.carriers = dataframes.get("carriers", pd.DataFrame())
        self.capacity = dataframes.get("capacity", pd.DataFrame())
        self.external = dataframes.get("external_market", pd.DataFrame())

        # Instantiate or load trained models (could be pre-trained serialized models)
        self.eta_model = ETAEstimator()
        self.price_model = PriceEstimator()
        self.capacity_model = CapacityForecaster()

        # Fit models on available data (lightweight). In prod, load pre-trained artefacts.
        try:
            if not self.shipments.empty:
                self.eta_model.fit(self.shipments)
                self.price_model.fit(self.shipments)
                self.capacity_model.fit(self.shipments)
        except Exception as e:
            logger.warning("Model fit failed with error: %s. Models will use defaults.", e)

    def generate_insights(self, agent: str, origin: str, destination: str, product: str) -> dict:
        """
        Returns dict with keys:
          - summary
          - historical_patterns
          - price_intel
          - capacity_trends
          - risk_signals
          - predictive_signals
          - recommended_actions
        """
        # Historical patterns (simple aggregations)
        hist = self._historical_patterns(origin, destination, product)

        # Price intelligence
        price_intel = self._price_intel(origin, destination, product)

        # Capacity
        capacity_trends = self._capacity_trends(origin, destination, product)

        # Predictive signals
        pred = self._predictive_signals(origin, destination, product)

        # Risk indicators
        risk = self._risk_indicators(origin, destination, product)

        # Recommended actions (rule-based + model hints)
        recs = self._recommendations(hist, price_intel, capacity_trends, pred, risk)

        summary = {
            "top_recommendation": recs[0] if recs else "No recommendation available",
            "expected_price_usd": price_intel.get("expected_price_usd"),
            "expected_delay_days": pred.get("expected_delay_days"),
        }

        return {
            "summary": summary,
            "historical_patterns": hist,
            "price_intel": price_intel,
            "capacity_trends": capacity_trends,
            "risk_signals": risk,
            "predictive_signals": pred,
            "recommended_actions": recs,
        }

    def _historical_patterns(self, origin, destination, product):
        df = self.shipments
        if df.empty:
            return {"note": "No shipment history available for this lane/product. Use external market data."}
        subset = df[
            (df["origin"].str.upper() == origin.upper()) &
            (df["destination"].str.upper() == destination.upper()) &
            (df["product"].str.lower() == product.lower())
        ]
        if subset.empty:
            # Fallback: lane-level or product-level
            lane = df[(df["origin"].str.upper() == origin.upper()) & (df["destination"].str.upper() == destination.upper())]
            if lane.empty:
                return {"note": "No exact matches. Returning broad stats." , "global_counts": len(df)}
            else:
                return {
                    "lane_count": len(lane),
                    "avg_price": float(lane["price_usd"].mean()),
                    "delay_rate": float(lane["delayed"].mean()) if "delayed" in lane.columns else None
                }
        return {
            "count": len(subset),
            "avg_transit_time_days": float((subset["arrived_at"] - subset["departed_at"]).dt.days.mean()) if "arrived_at" in subset and "departed_at" in subset else None,
            "avg_price_usd": float(subset["price_usd"].mean()) if "price_usd" in subset else None,
            "delay_rate": float(subset["delayed"].mean()) if "delayed" in subset else None,
            "top_carriers": subset["carrier"].value_counts().head(5).to_dict() if "carrier" in subset.columns else {}
        }

    def _price_intel(self, origin, destination, product):
        # Combine internal price model + external market (mock)
        expected_price = self.price_model.predict(origin, destination, product)
        # External market placeholder: if external df present, try to find price; else mock
        external_price = None
        if not self.external.empty:
            # user can map external data here
            external_price = None  # left as TODO

        # Simple signal: if expected_price < external_price -> underpriced etc. (external_price might be None)
        price_signal = "internal_only"
        if external_price:
            if expected_price < external_price * 0.95:
                price_signal = "offer_below_market"
            elif expected_price > external_price * 1.05:
                price_signal = "offer_above_market"
            else:
                price_signal = "in_line_with_market"

        return {
            "expected_price_usd": float(expected_price),
            "external_market_price_usd": float(external_price) if external_price is not None else None,
            "price_signal": price_signal,
            "confidence": 0.75  # placeholder confidence
        }

    def _capacity_trends(self, origin, destination, product):
        # Use capacity model to forecast short term capacity %
        cap_forecast = self.capacity_model.forecast(origin, destination, product, periods=4)
        return {
            "capacity_forecast": cap_forecast,
            "note": "Forecast returns next 4 weekly capacity % points"
        }

    def _predictive_signals(self, origin, destination, product):
        # ETA / delay prediction
        expected_delay_days = self.eta_model.predict_delay_days(origin, destination, product)
        # Rate change signal (very simple)
        expected_rate_change_pct = self.price_model.predict_rate_change_pct(origin, destination, product)
        return {
            "expected_delay_days": expected_delay_days,
            "expected_rate_change_pct": expected_rate_change_pct
        }

    def _risk_indicators(self, origin, destination, product):
        # Rule-based risk checks: known congested ports, high delay lanes, seasonality bumps
        risks = []
        # Example: port congestion keyword match (replace with real data)
        congested_ports = ["SHA", "HKG", "LON"]  # placeholder
        if origin.upper() in congested_ports or destination.upper() in congested_ports:
            risks.append("Port congestion risk")
        # Delay threshold
        hist = self._historical_patterns(origin, destination, product)
        if hist.get("delay_rate") and hist["delay_rate"] > 0.3:
            risks.append("High historical delay rate (>30%)")
        # Capacities
        cap = self._capacity_trends(origin, destination, product)
        if isinstance(cap.get("capacity_forecast"), list) and any(x > 85 for x in cap["capacity_forecast"]):
            risks.append("High capacity utilization expected -> reduced availability")

        return {"risks": risks, "score": min(1.0, 0.1 * len(risks))}

    def _recommendations(self, hist, price_intel, capacity_trends, pred, risk):
        recs = []
        # Pricing
        if price_intel.get("price_signal") == "offer_below_market":
            recs.append("Consider increasing price 3-7% to capture margin; still competitive vs market.")
        elif price_intel.get("price_signal") == "offer_above_market":
            recs.append("Consider discounting 3-5% to be market-competitive.")
        else:
            recs.append("Pricing looks in line. Monitor for short term rate changes.")

        # Routing/time
        if pred.get("expected_delay_days", 0) and pred["expected_delay_days"] > 3:
            recs.append("Delay risk high — consider alternative carriers or earlier departure.")
        if any("Port congestion" in r for r in risk.get("risks", [])):
            recs.append("Avoid congested ports if possible; route via alternate port or shift schedule.")

        # Capacity
        capf = capacity_trends.get("capacity_forecast", [])
        if capf and isinstance(capf, list) and np.mean(capf) > 80:
            recs.append("Capacity tight — prioritize high-value bookings and negotiate with carriers for space.")
        else:
            recs.append("Sufficient capacity expected — proceed with standard booking process.")

        # Agent-level suggestions
        if hist.get("top_carriers"):
            recs.append(f"Top carriers for this lane: {', '.join(list(hist['top_carriers'].keys())[:3])}")

        return recs
