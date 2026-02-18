import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from services.database import get_latest_price_date, execute_many, execute_query
from config import YFINANCE_TICKER, HISTORY_YEARS
import logging

logger = logging.getLogger(__name__)

def get_source_id() -> int:
    """Get the Yahoo Finance source_id."""
    df = execute_query(
        "SELECT source_id FROM zinc_forecast.data_sources WHERE source_name = 'Yahoo Finance'"
    )
    return int(df["source_id"].iloc[0])

def fetch_zinc_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch zinc prices from Yahoo Finance via yfinance."""
    ticker = yf.Ticker(YFINANCE_TICKER)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])

def run_ingestion() -> dict:
    """
    Main ingestion function. Checks latest date in DB, fetches missing data,
    upserts into zinc_prices. Returns summary dict.
    """
    source_id = get_source_id()
    latest = get_latest_price_date()

    if latest is None:
        # First run: seed 5 years of history
        start = (datetime.now() - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y-%m-%d")
    else:
        start = (latest + timedelta(days=1)).strftime("%Y-%m-%d")

    end = datetime.now().strftime("%Y-%m-%d")

    if start >= end:
        return {"status": "up_to_date", "rows_inserted": 0, "start": start, "end": end}

    df = fetch_zinc_prices(start, end)
    if df.empty:
        return {"status": "no_data", "rows_inserted": 0, "start": start, "end": end}

    # Upsert using MERGE
    merge_sql = """
        MERGE INTO zinc_forecast.zinc_prices tgt
        USING (SELECT TO_DATE(:price_date, 'YYYY-MM-DD') AS price_date,
                      :price_usd_mt AS price_usd_mt,
                      :source_id AS source_id,
                      :volume AS volume,
                      :high_usd_mt AS high_usd_mt,
                      :low_usd_mt AS low_usd_mt,
                      :open_usd_mt AS open_usd_mt,
                      :close_usd_mt AS close_usd_mt
               FROM DUAL) src
        ON (tgt.price_date = src.price_date AND tgt.source_id = src.source_id)
        WHEN NOT MATCHED THEN
          INSERT (price_date, price_usd_mt, source_id, volume,
                  high_usd_mt, low_usd_mt, open_usd_mt, close_usd_mt)
          VALUES (src.price_date, src.price_usd_mt, src.source_id, src.volume,
                  src.high_usd_mt, src.low_usd_mt, src.open_usd_mt, src.close_usd_mt)
        WHEN MATCHED THEN
          UPDATE SET price_usd_mt = src.price_usd_mt
    """

    data = [
        {
            "price_date": str(row["date"]),
            "price_usd_mt": float(row["close"]),
            "source_id": source_id,
            "volume": float(row["volume"]) if pd.notna(row["volume"]) else None,
            "high_usd_mt": float(row["high"]) if pd.notna(row["high"]) else None,
            "low_usd_mt": float(row["low"]) if pd.notna(row["low"]) else None,
            "open_usd_mt": float(row["open"]) if pd.notna(row["open"]) else None,
            "close_usd_mt": float(row["close"]) if pd.notna(row["close"]) else None,
        }
        for _, row in df.iterrows()
    ]

    rows = execute_many(merge_sql, data)
    logger.info(f"Ingestion complete: {rows} rows upserted ({start} to {end})")
    return {"status": "ok", "rows_inserted": rows, "start": start, "end": end}
