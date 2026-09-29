"""Synthetic cloud billing + resource inventory with planted cost problems."""
import numpy as np
import pandas as pd

SERVICES = {"EC2": 420, "RDS": 260, "EKS": 110, "S3": 90, "CloudFront": 70, "EBS": 60, "DataTransfer": 55, "Lambda": 45}
UNIT_PRICE = {"EC2": 0.05, "RDS": 0.12, "EKS": 0.10, "S3": 0.023, "CloudFront": 0.085, "EBS": 0.10, "DataTransfer": 0.09, "Lambda": 0.0000167}
ENV_SHARE = {"prod": 0.70, "staging": 0.15, "dev": 0.15}


def generate_billing(days: int = 180, seed: int = 5, start: str = "2026-04-01") -> pd.DataFrame:
    """Planted issues (day index): Lambda prod spike @90, DataTransfer prod spike @120-122,
    forgotten dev EC2 cluster (x3 cost) from @150 onward."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.date_range(start, periods=days)):
        growth = 1 + 0.0015 * i
        weekend = d.dayofweek >= 5
        for svc, base in SERVICES.items():
            for env, share in ENV_SHARE.items():
                season = (0.92 if env == "prod" else 0.6) if weekend else 1.0
                cost = base * share * growth * season * rng.normal(1, 0.04)
                if svc == "DataTransfer" and env == "prod" and 120 <= i <= 122:
                    cost *= 4.5
                if svc == "Lambda" and env == "prod" and i == 90:
                    cost *= 6
                if svc == "EC2" and env == "dev" and i >= 150:
                    cost *= 3
                rows.append((d, svc, env, round(cost, 2), round(cost / UNIT_PRICE[svc] * rng.normal(1, 0.02), 1)))
    return pd.DataFrame(rows, columns=["date", "service", "environment", "cost", "usage_quantity"])


def make_resources(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(14):
        env = ["prod"] * 6 + ["staging"] * 4 + ["dev"] * 4
        cpu = [2, 3, 1, 4] + [9, 12, 15, 18] + list(rng.uniform(35, 70, 6))
        rng.shuffle(cpu)
        rows.append((f"i-{1000 + i}", "EC2", env[i], round(float(rng.uniform(150, 900)), 2), round(float(cpu[i]), 1), None, None, 24))
    for i in range(6):
        rows.append((f"db-{200 + i}", "RDS", ["prod", "prod", "prod", "staging", "dev", "dev"][i],
                     round(float(rng.uniform(200, 700)), 2), round(float(rng.uniform(4, 60)), 1), None, None, 24))
    for i in range(12):
        rows.append((f"vol-{300 + i}", "EBS", rng.choice(["prod", "staging", "dev"]), round(float(rng.uniform(20, 120)), 2),
                     None, bool(i >= 5), None, None))
    for i in range(6):
        rows.append((f"bucket-{i}", "S3", "prod", round(float(rng.uniform(100, 800)), 2), None, None,
                     round(float(rng.uniform(10, 90)), 0), None))
    return pd.DataFrame(rows, columns=["resource_id", "type", "environment", "monthly_cost", "avg_cpu_pct",
                                       "attached", "cold_data_pct", "uptime_hours"])
