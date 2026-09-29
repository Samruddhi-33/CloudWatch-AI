import pandas as pd


def daily_totals(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d.groupby("date")["cost"].sum().sort_index()


def kpis(df: pd.DataFrame) -> dict:
    daily = daily_totals(df)
    last30, prev30 = daily.iloc[-30:].sum(), daily.iloc[-60:-30].sum()
    return {"total": float(daily.sum()), "last_30d": float(last30), "prev_30d": float(prev30),
            "change_pct": float(100 * (last30 - prev30) / prev30) if prev30 else float("nan"),
            "avg_daily": float(daily.iloc[-30:].mean())}


def top_movers(df: pd.DataFrame, by: str = "service", window: int = 30) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    end = d["date"].max()
    cur = d[d["date"] > end - pd.Timedelta(days=window)].groupby(by)["cost"].sum()
    prev = d[(d["date"] <= end - pd.Timedelta(days=window)) & (d["date"] > end - pd.Timedelta(days=2 * window))].groupby(by)["cost"].sum()
    out = pd.DataFrame({"current": cur, "previous": prev}).fillna(0)
    out["change"] = out["current"] - out["previous"]
    out["change_pct"] = (100 * out["change"] / out["previous"].replace(0, float("nan"))).round(1)
    return out.sort_values("change", ascending=False).round(2).reset_index()
