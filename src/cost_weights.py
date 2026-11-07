"""Cost weights per tariff period, estimated on the calibration period only.

Regresses calibration regret (within station, scaled by mean daily load) on under/over-forecast energy in
the valley, flat and peak periods, and writes outputs/<dataset>/cost_weights_<tag>.json with the per-kWh costs
alpha (under-forecast) and beta (over-forecast), floored at FLOOR so every error keeps some weight.
Usage: DATASET=uev TARIFF=sz python cost_weights.py --tag fx_base
"""
import argparse
import json

import numpy as np
import pandas as pd

from analysis import error_table, load_preds, split_name
from common import OUT, load_volume

FLOOR = 0.01
PERIODS = ("valley", "flat", "peak")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fx_base")
    args = ap.parse_args()
    v = load_volume()
    hubs = pd.read_csv(sorted(OUT.glob(f"hubs_{args.tag}*.csv"))[0], index_col=0).loc[v.columns]
    files = sorted(OUT.glob(f"cost_cal_{args.tag}*.parquet"))
    cost = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(["model", "station", "day"])
    cost = cost[~cost.model.isin(["oracle", "rule", "no_battery"]) & ~cost.model.str.contains("|", regex=False)]
    cost["day"] = cost["day"].astype(str)
    cost = cost[cost.model.map(lambda m: split_name(m)[1] in ("mse", "mae", "pinball70", "pinball90"))]
    d = cost.merge(error_table(load_preds(), "cal", hubs, v), on=["model", "station", "day"])
    feats = [f"{s}_{p}" for p in PERIODS for s in ("under", "over")]
    scale = d.groupby("station").day_kwh.transform("mean")
    y = d.regret / scale
    X = d[feats].div(scale, axis=0)
    Xd = X - X.groupby(d.station).transform("mean")
    yd = y - y.groupby(d.station).transform("mean")
    beta, *_ = np.linalg.lstsq(Xd.to_numpy(), yd.to_numpy(), rcond=None)
    coef = dict(zip(feats, beta))
    w = {p: {"alpha": max(coef[f"under_{p}"], FLOOR), "beta": max(coef[f"over_{p}"], FLOOR)} for p in PERIODS}
    out = {"tag": args.tag, "n_rows": int(len(d)), "raw": coef, "weights": w}
    (OUT / f"cost_weights_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
