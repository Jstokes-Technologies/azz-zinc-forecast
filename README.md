# AZZ Zinc Forecasting Tool

Zinc price forecasting proof-of-concept for AZZ Inc.'s DGS (Digital Galvanizing System) division. Built for procurement and operations planning by Debbie and Stephanie.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Streamlit Frontend                  │
│         (Dashboard / Historical / Forecast History)   │
├──────────────────────────────────────────────────────┤
│                   Python Services                     │
│  ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌─────────┐ │
│  │Ingestion│ │ Forecaster│ │AI Narrat.│ │Database │ │
│  │(yfinance│ │(Holt-     │ │(Claude   │ │(oracledb│ │
│  │ ZNC=F)  │ │ Winters)  │ │ Sonnet)  │ │ thin)   │ │
│  └─────────┘ └───────────┘ └──────────┘ └─────────┘ │
├──────────────────────────────────────────────────────┤
│              Oracle 23ai Free (FREEPDB1)              │
│           zinc_forecast schema / tablespace           │
│    [data_sources, zinc_prices, forecast_runs/results] │
└──────────────────────────────────────────────────────┘
```

## Features

- **Real-time zinc price ingestion** from Yahoo Finance (CME Zinc Futures ZNC=F)
- **Holt-Winters exponential smoothing** with automatic model selection (additive/multiplicative trend and seasonality)
- **Multiple forecast horizons**: 4 weeks, 3 months, 6 months, 12 months
- **Bootstrap prediction intervals** (80% and 95% confidence bands)
- **AI purchasing recommendations** via Claude (Anthropic) with procurement-focused language
- **USD/lb and USD/MT toggle** for galvanizing industry standard units
- **Historical data explorer** with CSV export
- **Forecast history tracking** with accuracy metrics (MAPE, RMSE, AIC)

## Prerequisites

- Docker and Docker Compose
- Existing Oracle 23ai Free container (`oracle23ai`) running on `localhost:1521/FREEPDB1`
- Anthropic API key (for AI commentary; optional)

## Quick Start

1. **Run the schema setup** against your Oracle 23ai instance:

```bash
docker exec -i oracle23ai sqlplus sys/YourSysPassword@FREEPDB1 as sysdba < db/scripts/01_schema_setup.sql
```

2. **Configure environment**:

```bash
cp .env.example .env
# Edit .env with your Anthropic API key
```

3. **Start the application**:

```bash
docker compose up --build -d
```

4. **Open the dashboard**: http://localhost:8501

5. Click **"Refresh Data & Run Forecast"** in the sidebar to seed historical data and generate your first forecast.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_DSN` | `localhost:1521/FREEPDB1` | Oracle connection string |
| `ORACLE_USER` | `zinc_forecast` | Database user |
| `ORACLE_PASSWORD` | `ZincForecast2024!` | Database password |
| `ANTHROPIC_API_KEY` | (none) | Anthropic API key for AI commentary |

## OCI Migration Path

To migrate from local Oracle 23ai Free to OCI Autonomous Transaction Processing (ATP):

1. **Export** using Data Pump:
   ```bash
   expdp zinc_forecast/ZincForecast2024! schemas=zinc_forecast dumpfile=zinc_forecast.dmp
   ```

2. **Upload** the dump file to OCI Object Storage

3. **Import** into ATP using `impdp` or the OCI Database Tools console

4. **Update** `ORACLE_DSN` in `.env` to point to your ATP connection string (wallet-based)

5. **Redeploy** the app container (no code changes needed, oracledb thin mode supports ATP wallets)

## Development Notes

- **No Oracle Client installation needed**: Uses `oracledb` Python package in thin mode
- **Model selection**: Automatically tests 8 Holt-Winters configurations (2 trend x 2 seasonal x 2 periods) and picks the best AIC
- **Data source**: Yahoo Finance ZNC=F (CME Zinc Futures) tracks LME zinc spot closely
- **Conversion factor**: 1 metric ton = 2,204.62 lbs

## Powered By

- [Oracle 23ai Free](https://www.oracle.com/database/free/)
- [Streamlit](https://streamlit.io/)
- [statsmodels](https://www.statsmodels.org/)
- [Anthropic Claude](https://www.anthropic.com/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- JMJ Cloud
