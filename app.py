"""CloudWatch AI - Streamlit UI.  Run: streamlit run app.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cloudwatch_ai.analysis import daily_totals, kpis, top_movers
from cloudwatch_ai.anomaly import detect_cost_anomalies
from cloudwatch_ai.data import generate_billing, make_resources
from cloudwatch_ai.forecast import backtest, forecast, month_end_projection
from cloudwatch_ai.optimize import recommendations

st.set_page_config(page_title="CloudWatch AI", page_icon="☁️", layout="wide")
st.title("☁️ CloudWatch AI - Cloud Cost & Usage Intelligence")

with st.sidebar:
    up = st.file_uploader("Billing CSV (date, service, cost, [environment])", type="csv")
    z_thr = st.slider("Anomaly sensitivity (z-score)", 2.5, 6.0, 3.5, 0.5, help="Lower = more alerts")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30)

if up:
    df = pd.read_csv(up)
    if not {"date", "service", "cost"} <= set(df.columns):
        st.error("CSV needs columns: date, service, cost")
        st.stop()
    if "environment" not in df:
        df["environment"] = "all"
    resources = None
else:
    df = generate_billing()
    resources = make_resources()
    st.info("Showing synthetic AWS-style billing data with planted problems (spikes + a forgotten dev cluster).")
df["date"] = pd.to_datetime(df["date"])
daily = daily_totals(df)

tab_o, tab_a, tab_f, tab_s = st.tabs(["📊 Overview", "🚨 Anomalies", "🔮 Forecast & budget", "💰 Savings"])

with tab_o:
    k, cols = kpis(df), st.columns(4)
    cols[0].metric("Last 30 days", f"${k['last_30d']:,.0f}", f"{k['change_pct']:+.1f}% vs prev 30d", delta_color="inverse")
    cols[1].metric("Avg daily (30d)", f"${k['avg_daily']:,.0f}")
    cols[2].metric("Total in data", f"${k['total']:,.0f}")
    cols[3].metric("Services", df["service"].nunique())
    by_day = df.groupby(["date", "service"], as_index=False)["cost"].sum()
    st.plotly_chart(px.area(by_day, x="date", y="cost", color="service", title="Daily cost by service"), width="stretch")
    l, r = st.columns(2)
    l.plotly_chart(px.pie(df.groupby("environment", as_index=False)["cost"].sum(), names="environment", values="cost", hole=.45, title="Cost by environment"), width="stretch")
    r.subheader("Top movers (last 30d vs previous 30d)")
    r.dataframe(top_movers(df), width="stretch")

with tab_a:
    anomalies = detect_cost_anomalies(df, z_thr=z_thr)
    if anomalies.empty:
        st.success("No cost anomalies detected.")
    else:
        m = st.columns(3)
        m[0].metric("Anomalous service-days", len(anomalies))
        m[1].metric("Excess spend", f"${anomalies['excess'].sum():,.0f}")
        m[2].metric("Worst offender", f"{anomalies.iloc[0]['service']} / {anomalies.iloc[0]['environment']}")
        st.dataframe(anomalies, width="stretch")
        pick = st.selectbox("Inspect", [f"{r.service} / {r.environment}" for r in anomalies.drop_duplicates(["service", "environment"]).itertuples()])
        svc, env = pick.split(" / ")
        series = df[(df.service == svc) & (df.environment == env)].groupby("date")["cost"].sum().reset_index()
        pts = anomalies[(anomalies.service == svc) & (anomalies.environment == env)]
        fig = go.Figure(go.Scatter(x=series.date, y=series.cost, mode="lines", name="Cost"))
        fig.add_trace(go.Scatter(x=pts.date, y=pts.cost, mode="markers", name="Anomaly", marker=dict(color="red", size=10)))
        fig.update_layout(title=pick, height=380)
        st.plotly_chart(fig, width="stretch")

with tab_f:
    fc, bt = forecast(daily, horizon), backtest(daily)
    hist = daily.iloc[-90:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual"))
    fig.add_trace(go.Scatter(x=fc.date, y=fc.upper, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fc.date, y=fc.lower, fill="tonexty", line=dict(width=0), name="80% band", fillcolor="rgba(99,110,250,.2)"))
    fig.add_trace(go.Scatter(x=fc.date, y=fc.forecast, name="Forecast", line=dict(dash="dash")))
    fig.update_layout(title=f"{horizon}-day cost forecast", height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Backtest on the last 30 days: MAPE = {bt['mape_pct']:.1f}%")
    default_budget = float(round(daily.iloc[-60:-30].sum() * 1.05, -3))
    budget = st.number_input("Monthly budget ($)", 0.0, 10_000_000.0, default_budget, step=1000.0)
    p = month_end_projection(daily, budget or None)
    c = st.columns(3)
    c[0].metric(f"Spent so far ({p['month']})", f"${p['spent_to_date']:,.0f}")
    c[1].metric("Projected month-end", f"${p['projected']:,.0f}")
    if p["over_budget_pct"] is not None:
        c[2].metric("vs budget", f"{p['over_budget_pct']:+.1f}%", delta_color="inverse")
        (st.error if p["over_budget_pct"] > 0 else st.success)(
            "⚠ Projected to exceed budget" if p["over_budget_pct"] > 0 else "✅ On track to stay within budget")

with tab_s:
    if resources is None:
        res_up = st.file_uploader("Resource inventory CSV (resource_id, type, environment, monthly_cost, avg_cpu_pct, attached, cold_data_pct, uptime_hours)", type="csv")
        resources = pd.read_csv(res_up) if res_up else None
    if resources is None:
        st.info("Upload a resource inventory to get rightsizing / cleanup recommendations.")
    else:
        recs = recommendations(resources)
        m = st.columns(3)
        m[0].metric("Potential monthly savings", f"${recs.monthly_savings.sum():,.0f}")
        m[1].metric("Actionable resources", len(recs))
        m[2].metric("Annualised", f"${recs.monthly_savings.sum() * 12:,.0f}")
        st.plotly_chart(px.bar(recs.groupby("action", as_index=False)["monthly_savings"].sum(), x="monthly_savings", y="action", orientation="h", title="Savings by action"), width="stretch")
        st.dataframe(recs, width="stretch")
        st.download_button("Download recommendations", recs.to_csv(index=False), "savings.csv")
