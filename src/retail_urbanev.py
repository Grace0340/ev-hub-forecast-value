"""Per-station mean retail price (electricity fee + service fee, CNY/kWh) over the UrbanEV fit period.

Writes data/proc/urbanev/retail_price.csv; the retail column is the price of shed load.
Usage: DATASET=uev python retail_urbanev.py
"""
import pandas as pd

from common import DATA, ROOT, SPLITS

F = ROOT / "data" / "dryad" / "UrbanEVDataset" / "20220901-20230228_station-processed" / "features"
ids = pd.read_csv(DATA / "stations.csv", index_col=0).index.astype(str).tolist()
start, end = SPLITS["A"]["fit"]
parts = {}
for name in ("e_price", "s_price"):
    df = pd.read_csv(F / f"{name}.csv", index_col=0, parse_dates=True, usecols=["time"] + ids)
    parts[name] = df.loc[start:f"{end} 23:55"].mean()
out = pd.DataFrame(parts)
out["retail"] = out.e_price + out.s_price
out.index.name = "station"
out.to_csv(DATA / "retail_price.csv")
print(out.describe().round(3).to_string())
