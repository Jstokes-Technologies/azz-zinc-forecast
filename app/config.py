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
