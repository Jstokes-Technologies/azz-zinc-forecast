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
