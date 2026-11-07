"""Rolling-horizon (shrinking) re-optimisation instead of following the day-ahead plan.

At every hour t the controller observes the current net load (unless MPC_PEEK=0), keeps the day-ahead forecast
for hours t+1..23, re-solves the LP from the current state of charge to the end of the day, and applies the first
decision (with the same shedding/export protection as cost_dayahead.realise). Scores the selected forecasts plus
oracle and the forecast-free rule. Writes cost_{period}_{tag}.parquet and hubs_{tag}.csv.
Usage: DATASET=uev TARIFF=sz python rolling.py --period test --pattern 'preds_A_*_mse_s0.npz' --tag fx_mpc
"""
import argparse
import glob
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from common import DATA, OUT, SPLITS, hour_index, load_volume
from cost_dayahead import (CYCLE_EPS, DEG, ETA, P_FIT, P_REFILL, T, VOLL_MULT, day_tariff, hour_bounds,
                           pv_per_kwp, realise, solve_hub)

MODELS = ["SNaive", "HistMean", "DLinear", "GRU", "GraphGRU", "iTransformer"]
RHO_PV, RHO_B, GQ = 0.3, 0.25, 0.95
# MPC_PEEK=0: the decision for hour t uses the day-ahead forecast of hour t too (state of charge still measured)
PEEK = os.environ.get("MPC_PEEK", "1") == "1"


def build(H):
    nv = 5 * H + H + 1
    U = lil_matrix((H, nv))
    E = lil_matrix((H + 1, nv))
    for t in range(H):
        g, x, c, d, s = (5 * t + k for k in range(5))
        U[t, g], U[t, x], U[t, c], U[t, d], U[t, s] = -1, 1, 1, -1, -1
        so, sn = 5 * H + t, 5 * H + t + 1
        E[t, sn], E[t, so], E[t, c], E[t, d] = 1, -1, -ETA, 1 / ETA
    E[H, 5 * H] = 1
    return U.tocsr(), E.tocsr(), nv


LP = {H: build(H) for H in range(1, T + 1)}


def first_step(net, t0, soc, hub, price):
    H = len(net)
    U, E, nv = LP[H]
    cost = np.zeros(nv)
    for t in range(H):
        cost[5 * t:5 * t + 5] = [price[t0 + t], -P_FIT, CYCLE_EPS + DEG, CYCLE_EPS + DEG, hub["voll"]]
    cost[5 * H + H] = -P_REFILL
    G, pb, emax = hub["G"], hub["pb"], hub["emax"]
    bounds = hour_bounds(net, G, pb) + [(0.1 * emax, 0.9 * emax)] * (H + 1)
    bounds[5 * H] = (soc, soc)
    b_eq = np.r_[np.zeros(H), soc]
    x = solve_hub(cost, U, E, b_eq, bounds, net, price[t0:])
    if x is None:
        return 0.0
    return float(x[3] - x[2])


def mpc_day(net_true, net_fc, hub, tp):
    plan = np.zeros(T)
    soc = 0.5 * hub["emax"]
    for t in range(T):
        f = net_fc[t:].copy()
        if PEEK:
            f[0] = net_true[t]
        plan[t] = first_step(f, t, float(np.clip(soc, 0.1 * hub["emax"], 0.9 * hub["emax"])), hub, tp["price"])
        # replay the protected controller up to hour t to get the true state of charge
        part = realise(np.r_[net_true[:t + 1], np.zeros(T - t - 1)], np.r_[plan[:t + 1], np.zeros(T - t - 1)],
                       hub, **tp)
        soc = 0.5 * hub["emax"] - part["terminal"] / P_REFILL
    return realise(net_true, plan, hub, **tp)


def one_station(args):
    sid, hub, true, pv, preds, dates = args
    out = []
    for k in range(true.shape[0]):
        L = true[k]
        if L.sum() <= 0:
            continue
        net = L - pv[k]
        tp = dict(zip(("price", "period"), day_tariff(dates[k])))
        out.append(dict(station=sid, model="rule", day=k, **realise(net, None, hub, **tp)))
        out.append(dict(station=sid, model="oracle", day=k, **mpc_day(net, net, hub, tp)))
        for name, P in preds.items():
            out.append(dict(station=sid, model=name, day=k, **mpc_day(net, P[k] - pv[k], hub, tp)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="test", choices=["cal", "test"])
    ap.add_argument("--pattern", nargs="+", default=None,
                    help="prediction files to score; default MSE seed 0 of each family")
    ap.add_argument("--tag", default="fx_mpc")
    ap.add_argument("--pv", type=float, default=RHO_PV)
    ap.add_argument("--batt", type=float, default=RHO_B)
    ap.add_argument("--gq", type=float, default=GQ)
    args = ap.parse_args()
    v = load_volume()
    ids = v.columns.to_numpy()
    sp = SPLITS["A"]
    fit = v.iloc[hour_index(v.index, *sp["fit"])]
    pv1 = pv_per_kwp(v.index)
    fit_pv_day = pv1[hour_index(v.index, *sp["fit"])].sum() / (len(fit) / 24)
    voll = VOLL_MULT * pd.read_csv(DATA / "retail_price.csv", index_col=0)["retail"].reindex(ids).fillna(1.7).to_numpy()
    e_day = fit.sum().to_numpy() / (len(fit) / 24)
    hubs = [dict(kwp=args.pv * e_day[j] / fit_pv_day, emax=args.batt * e_day[j], pb=args.batt * e_day[j] / 2,
                 G=float(np.quantile(fit.iloc[:, j], args.gq)), voll=voll[j]) for j in range(len(ids))]
    first = np.load(OUT / "preds_A_DLinear_mse_s0.npz", allow_pickle=True)
    truth = first[f"{args.period}_true"]
    origins = pd.to_datetime(first[f"{args.period}_origins"])
    pos = np.array([v.index.get_loc(o) for o in origins])
    pv_days = np.stack([pv1[p:p + T] for p in pos])
    files = (sorted({f for p in args.pattern for f in glob.glob(str(OUT / p))}) if args.pattern
             else [str(OUT / f"preds_A_{m}_mse_s0.npz") for m in MODELS])
    preds = {Path(f).stem.replace("preds_A_", ""): np.load(f)[f"{args.period}_pred"] for f in files}
    print("scoring", len(preds), "forecasts", flush=True)
    jobs = [(sid, hubs[j], truth[:, :, j], pv_days * hubs[j]["kwp"], {k: p[:, :, j] for k, p in preds.items()},
             origins) for j, sid in enumerate(ids)]
    jobs = jobs[:int(os.environ.get("MPC_LIMIT", len(jobs)))]
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        rows = [r for chunk in pool.imap_unordered(one_station, jobs, chunksize=2) for r in chunk]
    df = pd.DataFrame(rows)
    df["day"] = origins[df["day"].to_numpy()].strftime("%Y-%m-%d")
    orc = df[df.model == "oracle"].set_index(["station", "day"]).total
    df["regret"] = df.total - orc.reindex(pd.MultiIndex.from_frame(df[["station", "day"]])).to_numpy()
    df.to_parquet(OUT / f"cost_{args.period}_{args.tag}.parquet")
    pd.DataFrame(hubs, index=ids).to_csv(OUT / f"hubs_{args.tag}.csv")
    print(df.groupby("model")[["total", "regret", "shed_kwh"]].mean().sort_values("regret").round(3).to_string())


if __name__ == "__main__":
    main()
