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
