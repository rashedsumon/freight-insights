# app.py
# Main Streamlit app (entrypoint). Deploy this file in Streamlit Cloud or run locally: `streamlit run app.py`

import streamlit as st
from data_loader import DataLoader
from insights_engine import InsightsEngine
from utils import cached_run

st.set_page_config(page_title="Freight Booking Insights", layout="wide")

st.title("Real-time Freight Booking Insights")
st.markdown(
    "Enter booking fields (Agent, Origin, Destination, Product) and get instant "
    "historical patterns, price intelligence, capacity trends, risk indicators, predictive signals, "
    "and recommended actions."
)

# Initialize data loader and engine (loads data in background/cached)
@cached_run(ttl_seconds=3600)
def init():
    dl = DataLoader()
    df = dl.load_all()  # returns a dict of DataFrames
    engine = InsightsEngine(dataframes=df)
    return dl, engine

dl, engine = init()

with st.form("booking_form"):
    col1, col2, col3, col4 = st.columns(4)
    agent = col1.text_input("Agent", value="")
    origin = col2.text_input("Origin (UN/Port/City)", value="")
    destination = col3.text_input("Destination (UN/Port/City)", value="")
    product = col4.text_input("Product (HS / Product Category)", value="")
    submitted = st.form_submit_button("Get Insights")

if submitted:
    # Simple validation
    if not (origin and destination and product):
        st.warning("Please provide Origin, Destination and Product.")
    else:
        with st.spinner("Generating insights..."):
            # Real-time scoring
            insights = engine.generate_insights(
                agent=agent,
                origin=origin,
                destination=destination,
                product=product,
            )
        st.success("Insights ready")

        # Render top-line
        st.subheader("Summary Recommendations")
        st.write(insights["summary"])

        # Historical patterns
        st.subheader("Historical Shipment Patterns")
        st.write(insights["historical_patterns"])

        # Price intelligence
        st.subheader("Price Intelligence (internal + external)")
        st.write(insights["price_intel"])

        # Capacity & trends
        st.subheader("Capacity Trends")
        st.write(insights["capacity_trends"])

        # Risk indicators & predictive signals
        st.subheader("Risk & Predictive Signals")
        st.write(insights["risk_signals"])
        st.write(insights["predictive_signals"])

        # Recommended actions
        st.subheader("Recommended Actions")
        for i, rec in enumerate(insights["recommended_actions"], 1):
            st.markdown(f"**{i}.** {rec}")

        # Option to download JSON
        st.download_button("Download insights (JSON)", data=str(insights), file_name="insights.json")
