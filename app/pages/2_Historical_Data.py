import streamlit as st
import pandas as pd
from services.database import get_price_history
import plotly.graph_objects as go

st.title("Historical Zinc Prices")
st.caption("LME Zinc Spot Price History via Yahoo Finance (ZNC=F)")

col1, col2 = st.columns(2)
start_date = col1.date_input("Start Date", value=pd.Timestamp.now() - pd.DateOffset(years=2))
end_date = col2.date_input("End Date", value=pd.Timestamp.now())

unit = st.toggle("Show in USD/lb", value=True)
divisor = 2204.62 if unit else 1
unit_label = "USD/lb" if unit else "USD/MT"

df = get_price_history(str(start_date), str(end_date))
df["price_date"] = pd.to_datetime(df["price_date"])

st.metric("Records", len(df))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["price_date"], y=df["price_usd_mt"] / divisor,
    name="Close", line=dict(color="#1f77b4")
))
fig.update_layout(title="LME Zinc Price", xaxis_title="Date", yaxis_title=f"Price ({unit_label})", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Table
display_df = df.copy()
display_df["price_usd_lb"] = display_df["price_usd_mt"] / 2204.62
st.dataframe(display_df[["price_date", "price_usd_mt", "price_usd_lb", "high_usd_mt", "low_usd_mt", "volume"]].sort_values("price_date", ascending=False), use_container_width=True)

csv = display_df.to_csv(index=False)
st.download_button("Download CSV", csv, "zinc_prices.csv", "text/csv")
