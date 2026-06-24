"""Tabular feature engineering.

Three layers, in increasing order of "effort":
1. Snapshot columns straight from the CSV (numeric as-is, categoricals as
   pandas Categorical so LightGBM can split on them natively).
2. Single-flight engineered ratios -- cheap degradation-rate proxies.
3. Within-drone rolling/trend features over the last `TREND_WINDOW` flights.
   These exist because of the label definition: y=1 means "fails in the next
   5 flights", so the model needs to catch an *inflection point*, not just a
   high absolute reading. A drone that has always run hot isn't more
   dangerous than one that just started running hot -- the trend carries the
   signal that a snapshot can't.

All rolling features only look at the current + past flights of the same
drone_id (ordered by flight_index), so there's no leakage from the future.
"""
import numpy as np
import pandas as pd

from .data import CATEGORY_LEVELS

NUMERIC_COLS = [
    "battery_capacity_mAh", "max_payload_g", "propeller_in", "manufacture_batch",
    "flight_index", "payload_g", "ambient_temp_C", "wind_speed_ms",
    "flight_duration_min", "avg_throttle", "num_aggressive_maneuvers",
    "cumulative_flight_hours", "battery_cycles",
]
CATEGORICAL_COLS = list(CATEGORY_LEVELS.keys())
TREND_WINDOW = 3


def _rolling_slope(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    denom = (t ** 2).sum()
    return float((arr * t).sum() / denom) if denom > 0 else 0.0


def build_tabular_features(tab: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = tab.sort_values(["drone_id", "flight_index"]).copy()

    # --- engineered ratios (single-flight) ---
    df["battery_cycle_rate"] = df["battery_cycles"] / df["cumulative_flight_hours"].replace(0, np.nan)
    df["payload_ratio"] = df["payload_g"] / df["max_payload_g"].replace(0, np.nan)
    df["maneuver_rate"] = df["num_aggressive_maneuvers"] / df["flight_duration_min"].replace(0, np.nan)

    # --- within-drone rolling/trend features ---
    grp = df.groupby("drone_id", observed=True)
    df["battery_cycles_delta"] = grp["battery_cycles"].diff().fillna(0)
    df["flight_hours_delta"] = grp["cumulative_flight_hours"].diff().fillna(0)
    df["throttle_trend"] = grp["avg_throttle"].transform(
        lambda s: s.rolling(TREND_WINDOW, min_periods=1).apply(_rolling_slope, raw=True)
    )
    df["maneuvers_roll_mean"] = grp["num_aggressive_maneuvers"].transform(
        lambda s: s.rolling(TREND_WINDOW, min_periods=1).mean()
    )
    df["payload_roll_std"] = grp["payload_g"].transform(
        lambda s: s.rolling(TREND_WINDOW, min_periods=1).std()
    ).fillna(0)

    engineered_cols = [
        "battery_cycle_rate", "payload_ratio", "maneuver_rate",
        "battery_cycles_delta", "flight_hours_delta", "throttle_trend",
        "maneuvers_roll_mean", "payload_roll_std",
    ]
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + engineered_cols
    out = df[["flight_id"] + feature_cols]
    # restore the caller's original row order (we sorted by drone_id/flight_index
    # above purely so the rolling-window features see chronological order within
    # each drone) so positional indices -- e.g. from cv.get_folds(tab) -- still
    # line up between `tab` and this feature frame.
    out = out.set_index("flight_id").loc[tab["flight_id"].values].reset_index()
    return out, CATEGORICAL_COLS
