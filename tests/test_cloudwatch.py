import pandas as pd
import pytest

from cloudwatch_ai.analysis import daily_totals, kpis, top_movers
from cloudwatch_ai.anomaly import detect_cost_anomalies
from cloudwatch_ai.data import generate_billing, make_resources
from cloudwatch_ai.forecast import backtest, forecast, month_end_projection
from cloudwatch_ai.optimize import recommendations


@pytest.fixture(scope="module")
def df():
    return generate_billing()


def test_generator(df):
    assert len(df) == 180 * 8 * 3 and (df.cost > 0).all()


def test_detects_planted_anomalies(df):
    a = detect_cost_anomalies(df)
    start = pd.Timestamp("2026-04-01")
    day = lambda i: start + pd.Timedelta(days=i)  # noqa: E731
    dt = a[(a.service == "DataTransfer") & (a.environment == "prod")]["date"].tolist()
    assert day(120) in dt
    assert day(90) in a[(a.service == "Lambda") & (a.environment == "prod")]["date"].tolist()
    dev_ec2 = a[(a.service == "EC2") & (a.environment == "dev")]["date"]
    assert len(dev_ec2) > 0 and dev_ec2.min() == day(150)


def test_few_false_positives(df):
    a = detect_cost_anomalies(df)
    known = {("DataTransfer", "prod"), ("Lambda", "prod"), ("EC2", "dev")}
    assert {(r.service, r.environment) for r in a.itertuples()} <= known


def test_forecast_and_backtest(df):
    daily = daily_totals(df)
    fc = forecast(daily, 30)
    assert len(fc) == 30 and (fc.lower < fc.forecast).all() and (fc.forecast < fc.upper).all()
    assert backtest(daily)["mape_pct"] < 20


def test_month_end_and_budget(df):
    p = month_end_projection(daily_totals(df), budget=1.0)
    assert p["projected"] > p["spent_to_date"] and p["over_budget_pct"] > 0


def test_kpis_and_movers(df):
    assert kpis(df)["last_30d"] > 0
    assert top_movers(df).iloc[0]["service"] == "EC2"


def test_recommendations():
    res = make_resources()
    recs = recommendations(res)
    assert len(recs) > 5
    merged = recs.merge(res[["resource_id", "monthly_cost"]], on="resource_id", suffixes=("", "_r"))
    assert (merged.monthly_savings <= merged.monthly_cost + 0.01).all()
    assert recs.monthly_savings.is_monotonic_decreasing
