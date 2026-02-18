import streamlit as st
import pandas as pd
from services.database import get_price_history, get_pool
from services.ingestion import run_ingestion
from services.forecaster import run_forecast
from services.ai_narrative import generate_commentary
from config import FORECAST_HORIZONS
import plotly.graph_objects as go
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Zinc Price Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Zinc Forecast Tool")
st.sidebar.caption("AZZ DGS Division | Powered by JMJ Cloud")

if st.sidebar.button("Refresh Data & Run Forecast"):
    with st.spinner("Fetching latest zinc prices..."):
        result = run_ingestion()
        st.sidebar.success(f"Data updated: {result.get('rows_inserted', 0)} new rows")

# Route to main dashboard
st.title("Zinc Price Dashboard")
st.caption("LME Zinc Spot Price Forecast for AZZ DGS Procurement Planning")

# Horizon selector
horizon_key = st.radio(
    "Forecast Horizon",
    options=list(FORECAST_HORIZONS.keys()),
    format_func=lambda k: FORECAST_HORIZONS[k]["label"],
    horizontal=True,
    index=1  # Default to 3 months
)

col1, col2, col3, col4 = st.columns(4)

# Unit toggle
unit = st.toggle("Show prices in USD/lb (default: USD/MT)", value=True)
divisor = 2204.62 if unit else 1
unit_label = "USD/lb" if unit else "USD/MT"

# Load data and run forecast
@st.cache_data(ttl=300)
def load_forecast(horizon_key):
    return run_forecast(horizon_key)

try:
    with st.spinner(f"Running {FORECAST_HORIZONS[horizon_key]['label']} forecast..."):
        result = load_forecast(horizon_key)

    hist_df = get_price_history()
    hist_df["price_date"] = pd.to_datetime(hist_df["price_date"])

    current_price = result["current_price"]
    end_price = result["end_price"]
    mape = result["mape"]

    # Trend calculations
    hist_sorted = hist_df.sort_values("price_date")
    recent_30 = hist_sorted.tail(30)
    recent_90 = hist_sorted.tail(90)
    trend_30d = ((current_price - float(recent_30["price_usd_mt"].iloc[0])) / float(recent_30["price_usd_mt"].iloc[0])) * 100
    trend_90d = ((current_price - float(recent_90["price_usd_mt"].iloc[0])) / float(recent_90["price_usd_mt"].iloc[0])) * 100

    pct_change = ((end_price - current_price) / current_price) * 100

    # KPI cards
    col1.metric("Current Price", f"${current_price/divisor:.4f} {unit_label}", f"{trend_30d:+.1f}% (30d)")
    col2.metric(f"Forecast ({FORECAST_HORIZONS[horizon_key]['label']})", f"${end_price/divisor:.4f} {unit_label}", f"{pct_change:+.1f}%")
    col3.metric("Model MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error - lower is better")
    col4.metric("90-Day Trend", f"{trend_90d:+.1f}%")

    # Main chart
    fig = go.Figure()

    # Historical (last 12 months)
    hist_12m = hist_sorted.tail(252)
    fig.add_trace(go.Scatter(
        x=hist_12m["price_date"],
        y=hist_12m["price_usd_mt"] / divisor,
        name="Historical",
        line=dict(color="#1f77b4", width=2),
    ))

    # Forecast
    future_dates = result["future_dates"]
    forecast_vals = [v / divisor for v in result["forecast"].values]
    lower_80 = [v / divisor for v in result["lower_80"]]
    upper_80 = [v / divisor for v in result["upper_80"]]
    lower_95 = [v / divisor for v in result["lower_95"]]
    upper_95 = [v / divisor for v in result["upper_95"]]

    # 95% band
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_95 + lower_95[::-1],
        fill="toself",
        fillcolor="rgba(255, 165, 0, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence",
        showlegend=True,
    ))
    # 80% band
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_80 + lower_80[::-1],
        fill="toself",
        fillcolor="rgba(255, 165, 0, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Confidence",
        showlegend=True,
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_vals,
        name="Forecast",
        line=dict(color="orange", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=f"LME Zinc Spot Price — Historical + {FORECAST_HORIZONS[horizon_key]['label']} Forecast",
        xaxis_title="Date",
        yaxis_title=f"Price ({unit_label})",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # AI Commentary
    st.subheader("AI Purchasing Recommendation")
    with st.spinner("Generating AI commentary..."):
        commentary = generate_commentary(
            current_price=current_price,
            end_price=end_price,
            lower_80=result["lower_80"][-1],
            upper_80=result["upper_80"][-1],
            lower_95=result["lower_95"][-1],
            upper_95=result["upper_95"][-1],
            mape=mape,
            horizon_label=FORECAST_HORIZONS[horizon_key]["label"],
            trend_30d=trend_30d,
            trend_90d=trend_90d,
            seasonality_mode=result["params"].get("seasonal", "unknown"),
        )
    st.info(commentary)

    # Model details
    with st.expander("Model Details"):
        params = result["params"]
        st.write(f"**Trend:** {params.get('trend', 'N/A')}")
        st.write(f"**Seasonality:** {params.get('seasonal', 'N/A')} (periods: {params.get('seasonal_periods', 'N/A')})")
        st.write(f"**AIC Score:** {result['aic']:.2f}")
        st.write(f"**MAPE:** {mape:.2f}%")
        st.write(f"**RMSE:** {result['rmse']:.2f} USD/MT")

except Exception as e:
    st.error(f"Forecast error: {e}")
    st.info("Try clicking 'Refresh Data & Run Forecast' in the sidebar to initialize data.")
