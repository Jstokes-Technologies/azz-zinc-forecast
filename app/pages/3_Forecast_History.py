import streamlit as st
import pandas as pd
from services.database import execute_query
import plotly.graph_objects as go

st.title("Forecast History")
st.caption("Past forecast runs and accuracy tracking")

runs_df = execute_query("""
    SELECT run_id, run_timestamp, horizon_label, data_points_used,
           mape, rmse, aic_score, seasonality_mode, seasonal_periods, status
    FROM zinc_forecast.forecast_runs
    WHERE status = 'COMPLETED'
    ORDER BY run_timestamp DESC
    FETCH FIRST 50 ROWS ONLY
""")

if runs_df.empty:
    st.info("No completed forecast runs yet. Run a forecast from the Dashboard.")
else:
    st.dataframe(runs_df, use_container_width=True)

    # Pick a run to view
    selected_run = st.selectbox("View forecast details", runs_df["run_id"].tolist())

    if selected_run:
        results_df = execute_query("""
            SELECT forecast_date, predicted_usd_mt, predicted_usd_lb,
                   lower_80_usd_mt, upper_80_usd_mt, lower_95_usd_mt, upper_95_usd_mt,
                   actual_usd_mt
            FROM zinc_forecast.forecast_results
            WHERE run_id = :run_id
            ORDER BY forecast_date
        """, {"run_id": selected_run})

        results_df["forecast_date"] = pd.to_datetime(results_df["forecast_date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df["forecast_date"], y=results_df["predicted_usd_mt"], name="Forecast", line=dict(color="orange")))
        if results_df["actual_usd_mt"].notna().any():
            fig.add_trace(go.Scatter(x=results_df["forecast_date"], y=results_df["actual_usd_mt"], name="Actual", line=dict(color="blue")))
        fig.update_layout(title=f"Run #{selected_run} Forecast vs Actual", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(results_df, use_container_width=True)
