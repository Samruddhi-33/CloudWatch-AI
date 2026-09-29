"""Seasonality-aware robust anomaly detection on cost time series."""
import numpy as np
import pandas as pd


def detect_cost_anomalies(df: pd.DataFrame, by=("service", "environment"), window: int = 14,
                          z_thr: float = 3.5, min_ratio: float = 1.25) -> pd.DataFrame:
    """Flag days where cost is far above a trailing robust baseline (median/MAD on de-seasonalised data)."""
    by = list(by)
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    daily = d.groupby(by + ["date"], as_index=False)["cost"].sum()
    found = []
    for key, g in daily.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < window + 7:
            continue
        factor = (g.groupby(g["date"].dt.dayofweek)["cost"].transform("median") / g["cost"].median()).clip(lower=0.05)
        adj = g["cost"] / factor
        base = adj.shift(1)
        med = base.rolling(window, min_periods=7).median()
        mad = base.rolling(window, min_periods=7).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        mad = np.maximum(mad, 0.02 * med)
        z = 0.6745 * (adj - med) / mad
        expected = med * factor
        flag = (z > z_thr) & (g["cost"] > expected * min_ratio)
        for i in g.index[flag]:
            found.append({**dict(zip(by, key)), "date": g.loc[i, "date"], "cost": round(g.loc[i, "cost"], 2),
                          "expected": round(float(expected[i]), 2), "excess": round(g.loc[i, "cost"] - float(expected[i]), 2),
                          "severity_z": round(float(z[i]), 1)})
    cols = by + ["date", "cost", "expected", "excess", "severity_z"]
    return pd.DataFrame(found, columns=cols).sort_values("excess", ascending=False).reset_index(drop=True)
