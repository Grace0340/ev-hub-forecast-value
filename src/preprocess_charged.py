"""Hourly site-level load for a CHARGED city (2023-04-01..09-30).

First drops flat-lined, creeping and intermittent feeds (site_stats), then applies the UrbanEV station rule
(>= 50 kWh/day and active on >= 90% of fit and test days). Writes data/proc/<CITY>/{volume.csv, stations.csv,
retail_price.csv}. CHARGED's price columns are identical and mostly zero, so the shedding price (lost charging
revenue) is a city-level public charging price in local currency per kWh.
Usage: DATASET=LOA python preprocess_charged.py
"""
import numpy as np
import pandas as pd

from common import DATA, DATASET, ROOT, SPLITS

RETAIL = {"SZH": 1.56,   # UrbanEV Shenzhen median electricity + service fee (CNY)
          "LOA": 0.48,   # typical US public DC/L2 network price (USD)
          "MEL": 0.60}   # typical Australian public charging price (AUD)

SRC = ROOT / "data" / "charged" / DATASET


def site_stats(v):
    """Per site: median within-day coefficient of variation, share of flat (repeated nonzero) hours, share of
    creeping hours (rises below 5%), share of active days, first active day and mean hourly load."""
    x = v.to_numpy()
    days = x[: len(x) // 24 * 24].reshape(-1, 24, x.shape[1])
    dm, ds = days.mean(1), days.std(1)
    cv = np.where(dm > 0, ds / np.maximum(dm, 1e-9), np.nan)
    d = np.diff(x, axis=0)
    flat = ((d == 0) & (x[1:] > 0)).mean(0)
    up = ((d > 0) & (d < 0.05 * np.maximum(x[1:], 1e-9))).mean(0)
    active_days = (dm > 0).mean(0)
    first = np.argmax(dm > 0, axis=0)
    return pd.DataFrame(dict(cv=np.nanmedian(cv, 0), flat=flat, creep=up, active=active_days, first_day=first,
                             mean=x.mean(0)), index=v.columns)


DATA.mkdir(parents=True, exist_ok=True)
sp = SPLITS["A"]

v = pd.read_csv(SRC / "volume.csv", index_col=0, parse_dates=True)
v.columns = v.columns.astype(int)
q = site_stats(v)
daily = v.resample("1D").sum()
fit_d = daily.loc[sp["fit"][0]:sp["fit"][1]]
test_d = daily.loc[sp["test"][0]:sp["test"][1]]
stats = pd.DataFrame({"kwh_day": fit_d.mean(), "active_days": (fit_d > 0).mean(),
                      "test_active_days": (test_d > 0).mean()}).join(q)
clean = (stats.cv >= 0.15) & (stats.flat < 0.2) & (stats.creep < 0.3) & (stats.active >= 0.8)
keep = clean & (stats.kwh_day >= 50) & (stats.active_days >= 0.9) & (stats.test_active_days >= 0.9)
print(f"{DATASET}: sites {len(stats)}, clean {clean.sum()}, kept {keep.sum()}; "
      f"fit energy kept {stats.kwh_day[keep].sum() / stats.kwh_day.sum():.3f}")

sites = pd.read_csv(SRC / "sites.csv")
sites = sites.rename(columns={sites.columns[0]: "site"}).set_index("site")
st = stats[keep].join(sites[["longitude", "latitude", "charger_num"]])
h = v[st.index].fillna(0.0)
h.index.name = "time"
h.to_csv(DATA / "volume.csv")
st.index.name = "station"
st.to_csv(DATA / "stations.csv")

rp = pd.DataFrame({"retail": RETAIL[DATASET]}, index=st.index)
rp.index.name = "station"
rp.to_csv(DATA / "retail_price.csv")
print("hours", len(h), h.index[0], h.index[-1])
print("kWh/day quantiles", st.kwh_day.quantile([0, .1, .5, .9, 1]).round(1).to_dict())
print("centroid lat/lon", round(st.latitude.mean(), 3), round(st.longitude.mean(), 3))
