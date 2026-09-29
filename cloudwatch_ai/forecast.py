"""Trend + weekly-seasonality cost forecast (robust log-linear regression)."""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor


def _design(dates: pd.DatetimeIndex, t0: pd.Timestamp, scale: float) -> np.ndarray:
    t = ((dates - t0).days.values / scale).reshape(-1, 1)
    dow = np.eye(7)[dates.dayofweek.values]
    return np.hstack([t, dow])


def forecast(daily: pd.Series, horizon: int = 30, train_days: int = 90, z: float = 1.28) -> pd.DataFrame:
    """Returns date, forecast, lower, upper (80% band by default)."""
    y = daily.iloc[-train_days:]
    t0, scale = y.index[0], float(max(len(y), 1))
    X = _design(y.index, t0, scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HuberRegressor(alpha=1e-5, max_iter=2000, fit_intercept=False).fit(X, np.log(y.values))
    resid = np.log(y.values) - model.predict(X)
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon)
    mu = model.predict(_design(future, t0, scale))
    return pd.DataFrame({"date": future, "forecast": np.exp(mu), "lower": np.exp(mu - z * sigma), "upper": np.exp(mu + z * sigma)})


def backtest(daily: pd.Series, horizon: int = 30, train_days: int = 90) -> dict:
    train, test = daily.iloc[:-horizon], daily.iloc[-horizon:]
    fc = forecast(train, horizon, train_days)
    mape = float(np.mean(np.abs(fc["forecast"].values - test.values) / test.values) * 100)
    return {"mape_pct": mape, "actual": test, "forecast": fc}


def month_end_projection(daily: pd.Series, budget: float | None = None) -> dict:
    last = daily.index.max()
    remaining = (last + pd.offsets.MonthEnd(0) - last).days
    to_date = float(daily[daily.index.to_period("M") == last.to_period("M")].sum())
    rest = float(forecast(daily, remaining)["forecast"].sum()) if remaining > 0 else 0.0
    projected = to_date + rest
    return {"month": str(last.to_period("M")), "spent_to_date": to_date, "projected": projected,
            "days_remaining": remaining, "budget": budget,
            "over_budget_pct": (100 * (projected - budget) / budget) if budget else None}
