"""Hourly station-level charging load for UrbanEV (preprocessed station files of the Dryad archive).

Keeps stations with meaningful load that are active on at least 90% of fit and test days, and writes
data/proc/urbanev/{volume.csv (kWh per hour), stations.csv}.
Usage: python preprocess_urbanev.py
"""
import pandas as pd

from common import ROOT

F = ROOT / "data" / "dryad" / "UrbanEVDataset" / "20220901-20230228_station-processed" / "features"
OUT = ROOT / "data" / "proc" / "urbanev"
OUT.mkdir(parents=True, exist_ok=True)
FIT = ("2022-09-01", "2022-11-30 23:00")
TEST = ("2023-01-28", "2023-02-28")

v = pd.read_csv(F / "volume.csv", index_col=0, parse_dates=True)
v.columns = v.columns.astype(int)
h = v.resample("1h").sum(min_count=1)
fit = h.loc[FIT[0]:FIT[1]]
daily = fit.resample("1D").sum()


def run_stats(s):
    """Share of hours in nonzero runs of >= 6 identical values (forward-filled feed) and longest zero run."""
    run = s.groupby((s != s.shift()).cumsum()).transform("size")
    return pd.Series({"flat_nz_share": ((run >= 6) & (s > 0)).mean(),
                      "max_zero_run_h": run[s == 0].max() if (s == 0).any() else 0})


stats = pd.DataFrame({
    "kwh_day": daily.mean(),
    "active_days": (daily > 0).mean(),
    "zero_hours": (fit == 0).mean(),
    "cv_hourly": fit.std() / fit.mean(),
}).join(h.apply(run_stats).T)
full = h.resample("1D").sum()
stats["test_active_days"] = (full.loc[TEST[0]:TEST[1]] > 0).mean()
# Volume is charging piles x rated power, so constant runs are genuine; only whole-day outages matter,
# and those days are dropped at evaluation time rather than dropping the station.
keep = (stats.kwh_day >= 50) & (stats.active_days >= 0.9) & (stats.test_active_days >= 0.9)
print(stats.describe().round(3).to_string())
print(f"stations kept: {keep.sum()} of {len(stats)}; share of fit-period energy kept: "
      f"{stats.kwh_day[keep].sum() / stats.kwh_day.sum():.3f}")

inf = pd.read_csv(F / "station_inf.csv").set_index("station_id")
st = stats[keep].join(inf)
h = h[st.index].fillna(0.0)
h.index.name = "time"
h.to_csv(OUT / "volume.csv")
st.to_csv(OUT / "stations.csv")
print("hours", len(h), "range", h.index[0], h.index[-1])
print("kept kWh/day quantiles:", st.kwh_day.quantile([0, .1, .5, .9, 1]).round(1).to_dict())
print("kept piles quantiles:", st.charge_count.quantile([0, .1, .5, .9, 1]).to_dict())
