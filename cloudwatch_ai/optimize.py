"""Rule-based savings recommendations from a resource inventory."""
import pandas as pd


def recommendations(res: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in res.itertuples():
        action, saving, effort = None, 0.0, "Low"
        if r.type in ("EC2", "RDS") and pd.notna(r.avg_cpu_pct):
            if r.avg_cpu_pct < 5:
                action, saving = "Stop / terminate idle instance", 0.95 * r.monthly_cost
            elif r.avg_cpu_pct < 20:
                action, saving, effort = "Rightsize to a smaller instance type", 0.5 * r.monthly_cost, "Medium"
            elif r.environment in ("dev", "staging") and r.uptime_hours == 24:
                action, saving = "Schedule off-hours shutdown (nights/weekends)", 0.65 * r.monthly_cost
        elif r.type == "EBS" and r.attached is False:
            action, saving = "Snapshot & delete unattached volume", r.monthly_cost
        elif r.type == "S3" and pd.notna(r.cold_data_pct) and r.cold_data_pct > 50:
            action, saving = "Add lifecycle rule -> Glacier for cold data", r.monthly_cost * r.cold_data_pct / 100 * 0.7
        if action:
            recs.append({"resource_id": r.resource_id, "type": r.type, "environment": r.environment,
                         "action": action, "monthly_cost": round(r.monthly_cost, 2),
                         "monthly_savings": round(saving, 2), "effort": effort})
    cols = ["resource_id", "type", "environment", "action", "monthly_cost", "monthly_savings", "effort"]
    return pd.DataFrame(recs, columns=cols).sort_values("monthly_savings", ascending=False).reset_index(drop=True)
