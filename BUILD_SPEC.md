# AZZ Zinc Forecasting Tool — Build Spec

## What to Build

A zinc price forecasting proof-of-concept for AZZ Inc.'s DGS (Digital Galvanizing System) division. Users: Debbie and Stephanie (procurement/operations).

**Stack:**
- Oracle 23ai Free (existing Docker container `oracle23ai` on localhost:1521/FREEPDB1)
- Python 3.11 app with Streamlit frontend
- Holt-Winters exponential smoothing (statsmodels)
- Claude AI narrative via Anthropic Python SDK
- yfinance for zinc price data (ticker: ZNC=F)
- Docker Compose (app container only — reuse existing Oracle container)

---

## Project Structure

```
azz-zinc-forecast/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
├── db/
│   └── scripts/
│       └── 01_schema_setup.sql
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── config.py
    ├── streamlit_app.py
    ├── pages/
    │   ├── 1_Dashboard.py
    │   ├── 2_Historical_Data.py
    │   └── 3_Forecast_History.py
    └── services/
        ├── __init__.py
        ├── database.py
        ├── ingestion.py
        ├── forecaster.py
        └── ai_narrative.py
```

---

## Step 1: Database Schema (db/scripts/01_schema_setup.sql)

Run as SYSDBA against FREEPDB1. Create schema, tablespace, and all tables.

```sql
-- Run as SYSDBA
ALTER SESSION SET CONTAINER = FREEPDB1;

-- Tablespace
CREATE TABLESPACE zinc_forecast_ts
  DATAFILE 'zinc_forecast_ts01.dbf'
  SIZE 100M AUTOEXTEND ON NEXT 50M MAXSIZE 2G;

-- User
CREATE USER zinc_forecast
  IDENTIFIED BY "ZincForecast2024!"
  DEFAULT TABLESPACE zinc_forecast_ts
  TEMPORARY TABLESPACE temp
  QUOTA UNLIMITED ON zinc_forecast_ts;

GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW,
      CREATE SEQUENCE, CREATE PROCEDURE, CREATE TRIGGER,
      CREATE TYPE TO zinc_forecast;

ALTER SESSION SET CURRENT_SCHEMA = zinc_forecast;

-- Table: data_sources
CREATE TABLE zinc_forecast.data_sources (
  source_id    NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_name  VARCHAR2(100) NOT NULL,
  source_type  VARCHAR2(50) NOT NULL
               CONSTRAINT chk_source_type CHECK (source_type IN ('FREE_API','LME_PAID','MANUAL')),
  api_endpoint VARCHAR2(500),
  description  VARCHAR2(1000),
  is_active    NUMBER(1) DEFAULT 1 NOT NULL,
  created_date TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

-- Table: zinc_prices
CREATE TABLE zinc_forecast.zinc_prices (
  price_id      NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  price_date    DATE NOT NULL,
  price_usd_mt  NUMBER(12,2) NOT NULL,
  price_usd_lb  NUMBER(10,6),
  source_id     NUMBER NOT NULL REFERENCES zinc_forecast.data_sources(source_id),
  volume        NUMBER(15,2),
  high_usd_mt   NUMBER(12,2),
  low_usd_mt    NUMBER(12,2),
  open_usd_mt   NUMBER(12,2),
  close_usd_mt  NUMBER(12,2),
  load_timestamp TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
  CONSTRAINT uq_zinc_price_date_source UNIQUE (price_date, source_id)
);
CREATE INDEX idx_zinc_prices_date ON zinc_forecast.zinc_prices(price_date);

-- Trigger: auto-calculate price_usd_lb
CREATE OR REPLACE TRIGGER zinc_forecast.trg_calc_price_lb
  BEFORE INSERT OR UPDATE ON zinc_forecast.zinc_prices
  FOR EACH ROW
BEGIN
  :NEW.price_usd_lb := ROUND(:NEW.price_usd_mt / 2204.62, 6);
END;
/

-- Table: forecast_runs
CREATE TABLE zinc_forecast.forecast_runs (
  run_id              NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_timestamp       TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
  model_type          VARCHAR2(50) DEFAULT 'HOLT_WINTERS' NOT NULL,
  seasonality_mode    VARCHAR2(20),
  seasonal_periods    NUMBER,
  horizon_days        NUMBER NOT NULL,
  horizon_label       VARCHAR2(50) NOT NULL,
  training_start_date DATE,
  training_end_date   DATE,
  data_points_used    NUMBER,
  aic_score           NUMBER(15,6),
  mape                NUMBER(8,4),
  rmse                NUMBER(12,4),
  parameters_json     CLOB,
  ai_commentary       CLOB,
  status              VARCHAR2(20) DEFAULT 'RUNNING' NOT NULL
                      CONSTRAINT chk_run_status CHECK (status IN ('RUNNING','COMPLETED','FAILED')),
  error_message       VARCHAR2(4000)
);

-- Table: forecast_results
CREATE TABLE zinc_forecast.forecast_results (
  result_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id            NUMBER NOT NULL REFERENCES zinc_forecast.forecast_runs(run_id) ON DELETE CASCADE,
  forecast_date     DATE NOT NULL,
  predicted_usd_mt  NUMBER(12,2) NOT NULL,
  predicted_usd_lb  NUMBER(10,6),
  lower_80_usd_mt   NUMBER(12,2),
  upper_80_usd_mt   NUMBER(12,2),
  lower_95_usd_mt   NUMBER(12,2),
  upper_95_usd_mt   NUMBER(12,2),
  actual_usd_mt     NUMBER(12,2)
);
CREATE INDEX idx_forecast_results_run ON zinc_forecast.forecast_results(run_id);
CREATE INDEX idx_forecast_results_date ON zinc_forecast.forecast_results(forecast_date);

-- Trigger: auto-calculate predicted_usd_lb
CREATE OR REPLACE TRIGGER zinc_forecast.trg_calc_forecast_lb
  BEFORE INSERT OR UPDATE ON zinc_forecast.forecast_results
  FOR EACH ROW
BEGIN
  :NEW.predicted_usd_lb := ROUND(:NEW.predicted_usd_mt / 2204.62, 6);
END;
/

-- Seed data source
INSERT INTO zinc_forecast.data_sources (source_name, source_type, api_endpoint, description)
VALUES ('Yahoo Finance', 'FREE_API', 'yfinance:ZNC=F', 'CME Zinc Futures via yfinance - tracks LME closely');
COMMIT;
```

---

## Step 2: Docker Compose (docker-compose.yml)

**IMPORTANT:** Do NOT spin up a new Oracle container. Reuse the existing `oracle23ai` container.

```yaml
version: '3.8'

services:
  zinc-app:
    build:
      context: ./app
      dockerfile: Dockerfile
    container_name: zinc-forecast-app
    ports:
      - "8501:8501"
    environment:
      - ORACLE_DSN=${ORACLE_DSN:-localhost:1521/FREEPDB1}
      - ORACLE_USER=zinc_forecast
      - ORACLE_PASSWORD=ZincForecast2024!
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    network_mode: host
    volumes:
      - ./app:/app
    command: streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

Note: `network_mode: host` so the container can reach localhost:1521 (the existing Oracle container port-mapped to host).

---

## Step 3: app/Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Oracle thin client deps (libaio for linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Step 4: app/requirements.txt

```
streamlit>=1.30
plotly>=5.18
pandas>=2.1
numpy>=1.26
statsmodels>=0.14
oracledb>=2.0
anthropic>=0.40
yfinance>=0.2.31
apscheduler>=3.10
python-dotenv>=1.0
```

---

## Step 5: app/config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

ORACLE_DSN = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1")
ORACLE_USER = os.getenv("ORACLE_USER", "zinc_forecast")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "ZincForecast2024!")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

FORECAST_HORIZONS = {
    "4_WEEKS":   {"trading_days": 20,  "calendar_days": 28,  "label": "4 Weeks"},
    "3_MONTHS":  {"trading_days": 63,  "calendar_days": 90,  "label": "3 Months"},
    "6_MONTHS":  {"trading_days": 126, "calendar_days": 180, "label": "6 Months"},
    "12_MONTHS": {"trading_days": 252, "calendar_days": 365, "label": "12 Months"},
}

YFINANCE_TICKER = "ZNC=F"
HISTORY_YEARS = 5  # Years of historical data to seed
```

---

## Step 6: app/services/database.py

Oracle connection pool using oracledb thin mode (no Oracle Client install needed).

```python
import oracledb
import pandas as pd
from config import ORACLE_DSN, ORACLE_USER, ORACLE_PASSWORD

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = oracledb.create_pool(
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            dsn=ORACLE_DSN,
            min=2,
            max=5,
            increment=1
        )
    return _pool

def execute_query(sql: str, params: dict = None) -> pd.DataFrame:
    """Execute a SELECT query and return a DataFrame."""
    pool = get_pool()
    with pool.acquire() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
            cols = [d[0].lower() for d in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)

def execute_dml(sql: str, params: dict = None) -> int:
    """Execute INSERT/UPDATE/DELETE and commit. Returns rowcount."""
    pool = get_pool()
    with pool.acquire() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
            rowcount = cur.rowcount
        conn.commit()
    return rowcount

def execute_many(sql: str, data: list) -> int:
    """Bulk insert. Returns rowcount."""
    pool = get_pool()
    with pool.acquire() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, data)
            rowcount = cur.rowcount
        conn.commit()
    return rowcount

def get_latest_price_date() -> pd.Timestamp | None:
    """Return the most recent price_date in zinc_prices, or None."""
    df = execute_query("SELECT MAX(price_date) AS max_date FROM zinc_forecast.zinc_prices")
    val = df["max_date"].iloc[0]
    return pd.Timestamp(val) if val else None

def get_price_history(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Return historical prices sorted by date."""
    where = ""
    params = {}
    if start_date and end_date:
        where = "WHERE price_date BETWEEN TO_DATE(:start, 'YYYY-MM-DD') AND TO_DATE(:end, 'YYYY-MM-DD')"
        params = {"start": start_date, "end": end_date}
    return execute_query(
        f"SELECT price_date, price_usd_mt, price_usd_lb, high_usd_mt, low_usd_mt, volume "
        f"FROM zinc_forecast.zinc_prices {where} ORDER BY price_date",
        params
    )
```

---

## Step 7: app/services/ingestion.py

```python
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
```

---

## Step 8: app/services/forecaster.py

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from services.database import execute_query, execute_dml, execute_many, get_price_history
from config import FORECAST_HORIZONS
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def run_forecast(horizon_key: str) -> dict:
    """
    Run Holt-Winters forecast for given horizon.
    Saves results to forecast_runs and forecast_results tables.
    Returns run metadata dict.
    """
    config = FORECAST_HORIZONS[horizon_key]
    horizon_days = config["trading_days"]
    
    # Load historical prices
    df = get_price_history()
    if len(df) < 60:
        raise ValueError(f"Insufficient data: {len(df)} rows (need >= 60)")
    
    df["price_date"] = pd.to_datetime(df["price_date"])
    df = df.set_index("price_date").sort_index()
    series = df["price_usd_mt"].dropna()
    
    training_start = series.index[0].date()
    training_end = series.index[-1].date()
    n_points = len(series)
    
    # Create run record
    run_id_df = execute_query("""
        INSERT INTO zinc_forecast.forecast_runs
          (model_type, horizon_days, horizon_label, training_start_date, training_end_date,
           data_points_used, status)
        VALUES ('HOLT_WINTERS', :horizon_days, :horizon_label,
                TO_DATE(:train_start, 'YYYY-MM-DD'), TO_DATE(:train_end, 'YYYY-MM-DD'),
                :n_points, 'RUNNING')
        RETURNING run_id INTO :run_id
    """, {
        "horizon_days": horizon_days,
        "horizon_label": horizon_key,
        "train_start": str(training_start),
        "train_end": str(training_end),
        "n_points": n_points,
    })
    # Simpler approach: get max run_id after insert
    execute_dml("""
        INSERT INTO zinc_forecast.forecast_runs
          (model_type, horizon_days, horizon_label, training_start_date, training_end_date,
           data_points_used, status)
        VALUES ('HOLT_WINTERS', :horizon_days, :horizon_label,
                TO_DATE(:train_start, 'YYYY-MM-DD'), TO_DATE(:train_end, 'YYYY-MM-DD'),
                :n_points, 'RUNNING')
    """, {
        "horizon_days": horizon_days,
        "horizon_label": horizon_key,
        "train_start": str(training_start),
        "train_end": str(training_end),
        "n_points": n_points,
    })
    run_id_df = execute_query("SELECT MAX(run_id) AS run_id FROM zinc_forecast.forecast_runs")
    run_id = int(run_id_df["run_id"].iloc[0])
    
    try:
        # Fit best Holt-Winters model (try additive/multiplicative combos)
        best_model = None
        best_aic = np.inf
        best_params = {}
        
        for seasonal in ["add", "mul"]:
            for trend in ["add", "mul"]:
                for sp in [5, 21]:  # weekly or monthly seasonality
                    try:
                        m = ExponentialSmoothing(
                            series,
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=sp,
                            initialization_method="estimated"
                        ).fit(optimized=True)
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best_model = m
                            best_params = {
                                "trend": trend,
                                "seasonal": seasonal,
                                "seasonal_periods": sp,
                                "alpha": float(m.params.get("smoothing_level", 0)),
                                "beta": float(m.params.get("smoothing_trend", 0)),
                                "gamma": float(m.params.get("smoothing_seasonal", 0)),
                            }
                    except Exception:
                        continue
        
        if best_model is None:
            raise RuntimeError("All Holt-Winters model fits failed")
        
        # Generate forecast
        forecast = best_model.forecast(horizon_days)
        
        # Bootstrap prediction intervals
        residuals = best_model.resid.dropna().values
        sims = []
        for _ in range(1000):
            sampled = np.random.choice(residuals, size=horizon_days)
            sims.append(forecast.values + np.cumsum(sampled) * 0.5)
        sims = np.array(sims)
        
        lower_80 = np.percentile(sims, 10, axis=0)
        upper_80 = np.percentile(sims, 90, axis=0)
        lower_95 = np.percentile(sims, 2.5, axis=0)
        upper_95 = np.percentile(sims, 97.5, axis=0)
        
        # MAPE on last 20% of training data (in-sample)
        fitted = best_model.fittedvalues
        test_slice = series.iloc[int(len(series) * 0.8):]
        fitted_slice = fitted.iloc[int(len(fitted) * 0.8):]
        mape = float(np.mean(np.abs((test_slice.values - fitted_slice.values) / test_slice.values)) * 100)
        rmse = float(np.sqrt(np.mean((test_slice.values - fitted_slice.values) ** 2)))
        
        # Generate future business dates
        last_date = series.index[-1]
        future_dates = []
        current = last_date
        while len(future_dates) < horizon_days:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Mon-Fri
                future_dates.append(current)
        
        # Insert forecast results
        results_data = [
            {
                "run_id": run_id,
                "forecast_date": d.strftime("%Y-%m-%d"),
                "predicted_usd_mt": float(forecast.iloc[i]),
                "lower_80_usd_mt": float(lower_80[i]),
                "upper_80_usd_mt": float(upper_80[i]),
                "lower_95_usd_mt": float(lower_95[i]),
                "upper_95_usd_mt": float(upper_95[i]),
            }
            for i, d in enumerate(future_dates)
        ]
        
        execute_many("""
            INSERT INTO zinc_forecast.forecast_results
              (run_id, forecast_date, predicted_usd_mt, lower_80_usd_mt, upper_80_usd_mt,
               lower_95_usd_mt, upper_95_usd_mt)
            VALUES (:run_id, TO_DATE(:forecast_date, 'YYYY-MM-DD'), :predicted_usd_mt,
                    :lower_80_usd_mt, :upper_80_usd_mt, :lower_95_usd_mt, :upper_95_usd_mt)
        """, results_data)
        
        # Update run record to COMPLETED
        execute_dml("""
            UPDATE zinc_forecast.forecast_runs
            SET status = 'COMPLETED', aic_score = :aic, mape = :mape, rmse = :rmse,
                seasonality_mode = :seasonal, seasonal_periods = :sp,
                parameters_json = :params
            WHERE run_id = :run_id
        """, {
            "aic": best_aic,
            "mape": mape,
            "rmse": rmse,
            "seasonal": best_params["seasonal"],
            "sp": best_params["seasonal_periods"],
            "params": json.dumps(best_params),
            "run_id": run_id,
        })
        
        return {
            "run_id": run_id,
            "horizon_key": horizon_key,
            "horizon_label": config["label"],
            "forecast": forecast,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "future_dates": future_dates,
            "mape": mape,
            "rmse": rmse,
            "aic": best_aic,
            "params": best_params,
            "current_price": float(series.iloc[-1]),
            "end_price": float(forecast.iloc[-1]),
        }
    
    except Exception as e:
        execute_dml(
            "UPDATE zinc_forecast.forecast_runs SET status='FAILED', error_message=:err WHERE run_id=:run_id",
            {"err": str(e)[:4000], "run_id": run_id}
        )
        raise
```

---

## Step 9: app/services/ai_narrative.py

```python
import anthropic
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
import logging

logger = logging.getLogger(__name__)

def generate_commentary(
    current_price: float,
    end_price: float,
    lower_80: float,
    upper_80: float,
    lower_95: float,
    upper_95: float,
    mape: float,
    horizon_label: str,
    trend_30d: float,
    trend_90d: float,
    seasonality_mode: str,
) -> str:
    """Generate AI narrative commentary using Claude."""
    if not ANTHROPIC_API_KEY:
        return "AI commentary unavailable (no API key configured)."
    
    direction = "increase" if end_price > current_price else "decrease"
    pct_change = ((end_price - current_price) / current_price) * 100
    current_lb = current_price / 2204.62
    end_lb = end_price / 2204.62
    lower_80_lb = lower_80 / 2204.62
    upper_80_lb = upper_80 / 2204.62
    
    prompt = f"""You are a commodity pricing analyst for a galvanizing company that consumes zinc as its primary raw material. Analyze the following zinc price forecast and provide actionable commentary for the procurement team.

CURRENT MARKET DATA:
- Latest LME zinc spot price: ${current_lb:.4f}/lb (${current_price:,.0f}/MT)
- 30-day trend: {trend_30d:+.1f}%
- 90-day trend: {trend_90d:+.1f}%

FORECAST ({horizon_label}):
- Predicted price at horizon end: ${end_lb:.4f}/lb (${end_price:,.0f}/MT)
- Direction: {direction} ({pct_change:+.1f}% from current)
- 80% confidence range: ${lower_80_lb:.4f} - ${upper_80_lb:.4f}/lb (${lower_80:,.0f} - ${upper_80:,.0f}/MT)
- 95% confidence range: ${lower_95/2204.62:.4f} - ${upper_95/2204.62:.4f}/lb (${lower_95:,.0f} - ${upper_95:,.0f}/MT)
- Model: Holt-Winters ({seasonality_mode} seasonality)
- Model accuracy (MAPE): {mape:.1f}%

Provide:
1. A 2-3 sentence executive summary of the forecast
2. A purchasing recommendation (accelerate purchases, maintain current pace, or consider deferring)
3. Key risk factors to monitor
4. Any notable seasonal patterns the model detected

Keep the language clear for procurement and finance professionals, not data scientists. Lead with USD per pound since that is what the galvanizing industry uses. Reference USD per metric ton parenthetically. Do not use em dashes; use commas, periods, or restructure sentences instead."""
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"AI narrative error: {e}")
        return f"AI commentary temporarily unavailable: {e}"
```

---

## Step 10: app/streamlit_app.py (Main Entry Point)

```python
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
```

---

## Step 11: pages/2_Historical_Data.py

```python
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
```

---

## Step 12: pages/3_Forecast_History.py

```python
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
```

---

## Step 13: README.md

Write a clear README covering:
- What this tool does and who it's for (AZZ DGS procurement planning)
- Architecture overview (diagram in ASCII is fine)
- Prerequisites (Docker, existing oracle23ai container, Anthropic API key)
- Quick start (3 commands: schema setup, docker compose up, open browser)
- Configuration (.env setup)
- OCI migration path (Data Pump export, point to OCI ATP)
- Development notes (no Oracle Client needed, oracledb thin mode)

---

## Step 14: .gitignore

```
.env
__pycache__/
*.pyc
*.pyo
.streamlit/
*.egg-info/
dist/
build/
.DS_Store
```

---

## Step 15: .env.example

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
ORACLE_DSN=localhost:1521/FREEPDB1
```

---

## Final Steps

1. Create all files per the spec above
2. Review the code for correctness, especially:
   - Oracle MERGE syntax (FREEPDB1, zinc_forecast schema)
   - oracledb thin mode connection (no Oracle Client needed)
   - The INSERT/RETURNING pattern for run_id (simplify to INSERT then SELECT MAX(run_id) to avoid oracledb RETURNING clause complexity)
   - Streamlit pages routing
3. Commit everything to git with message "feat: initial AZZ zinc forecasting tool POC"
4. Push to origin main
5. When done, run: `openclaw gateway wake --text "Done: AZZ Zinc Forecasting Tool POC built and pushed to GitHub. Schema SQL ready to run. Streamlit app ready to docker compose up. Repo: https://github.com/Jstokes-Technologies/azz-zinc-forecast" --mode now`
