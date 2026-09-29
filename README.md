# ☁️ CloudWatch AI - Cloud Cost & Usage Intelligence

Turn raw cloud billing data into answers: **what changed, what's abnormal, what will we spend, and where can we save.**

## Features
- **Overview**: daily cost by service/environment, 30-day KPIs and top cost movers
- **Anomaly detection**: seasonality-aware robust z-score (median/MAD) per service x environment; estimates excess spend
- **Forecasting**: trend + weekly seasonality with a robust (Huber) log-linear model, 80 % band, 30-day backtest (MAPE)
- **Budget guard**: projects month-end spend and warns if you'll overshoot
- **Savings engine**: flags idle instances, rightsizing, unattached volumes, off-hours shutdown for dev/staging, S3 lifecycle rules

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
pytest
```
The demo data has planted problems: a Lambda spike, a 3-day data-transfer spike and a forgotten
dev EC2 cluster that triples cost from day 150 - the Anomalies tab finds all three.

## Use your own data
Billing CSV: `date, service, cost` (+ optional `environment`). For AWS, export from Cost Explorer or the CUR.
Optional resource inventory CSV: `resource_id, type, environment, monthly_cost, avg_cpu_pct, attached, cold_data_pct, uptime_hours`.

## Structure
```
app.py
cloudwatch_ai/
  data.py        synthetic billing + resource inventory
  analysis.py    KPIs and top movers
  anomaly.py     robust anomaly detection
  forecast.py    forecast, backtest, month-end projection
  optimize.py    savings recommendations
tests/
```

## Method notes
Anomaly baseline = trailing 14-day median of de-seasonalised cost; flag when z > threshold **and** cost is
>= 25 % above expected (avoids alerting on tiny noise). Savings percentages are heuristics - validate before acting.

## License
MIT - see [LICENSE](LICENSE). Replace `<Your Name>` with your name.
