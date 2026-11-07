"""Operating cost of day-ahead load forecasts in a PV-storage-charging hub at every station (day-ahead following).

Each station gets a hub sized from its fit-period load: PV yielding RHO_PV of mean daily load, a battery
of RHO_B x mean daily load at C/2, and a grid connection G at the fit-period hourly-load P95.
Day ahead, an LP schedules battery charge/discharge and grid purchase against the forecast load.
In real time the battery follows its schedule except to avoid shedding or exporting (see realise; the
"|rigid" variants follow it strictly), the grid absorbs the deviation up to G, surplus is exported at the
feed-in price, and load above G is shed at the station's retail price (lost revenue).
Benchmarks: no battery, and a forecast-free rule (fill at valley, cover load at peak).
The oracle runs the same LP on the realised load; regret = realised cost - oracle cost.
Environment: DATASET, TARIFF (sz, szt, gd, sce, vdo), DEG_COST, VOLL_MULT, END_VALUE, NPROC.
Usage: DATASET=uev TARIFF=sz python cost_dayahead.py --period test --tag fx_base [--pv 0.3 --batt 0.25 --gq 0.95]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, hstack, lil_matrix

from common import DATA, DATASET, OUT, ROOT, SPLITS, hour_index, load_volume

T = 24
ETA = 0.95
# sz: Shenzhen Power Supply Bureau agency-purchase tariff, large C&I 101-3000 kVA, 10 kV, <=250 kWh/kVA,
#     all-in incl. government funds (monthly bulletins Sep 2022-Feb 2023), with the peak/valley ratios of the
#     Shenzhen DRC notice of 2021-12-31 (Shenzhen keeps its own C&I ratios; periods as Guangdong)
# szt: sz with the test-period flat price (Feb 2023 bulletin)
# gd: Guangdong provincial structure, Yue-FGJ [2021] 331 (excludes Shenzhen) with a 0.65 flat price
# External validation (CHARGED, local currency per kWh):
# sce: SCE TOU-EV-8 (separately metered EV charging, 20-500 kW; demand charge zero until 2024), summer
#      Jun-Sep: weekday 16-21 on-peak, weekend 16-21 mid-peak, else off-peak; winter 16-21 mid, 8-16 super-off
# vdo: Victorian Default Offer 2023-24 small-business two-period TOU, CitiPower zone, peak 9-21 weekdays
TARIFF = os.environ.get("TARIFF", "sz")
P_FLAT, RATIO = {"sz": (0.74, {"peak": 1.505, "flat": 1.0, "valley": 0.364}),
                 "szt": (0.795, {"peak": 1.505, "flat": 1.0, "valley": 0.364}),
                 "gd": (0.65, {"peak": 1.7, "flat": 1.0, "valley": 0.38})}.get(TARIFF, (None, None))
P_FIT = {"sz": 0.453, "szt": 0.453, "gd": 0.453,  # Guangdong coal-fired benchmark price for PV export
         "sce": 0.08,               # stylised net-billing export value (USD)
         "vdo": 0.049}[TARIFF]      # Victorian minimum feed-in tariff 2023-24 (AUD)
PERIOD = ["valley"] * 8 + ["flat"] * 2 + ["peak"] * 2 + ["flat"] * 2 + ["peak"] * 5 + ["flat"] * 5
PRICE = np.array([P_FLAT * RATIO[p] for p in PERIOD]) if P_FLAT else None
CYCLE_EPS = 1e-4                    # tiny throughput cost so the LP does not cycle the battery idly
DEG = float(os.environ.get("DEG_COST", 0.0))   # degradation cost per kWh charged or discharged
VOLL_MULT = float(os.environ.get("VOLL_MULT", 1.0))   # scales the shed-load price
# END_VALUE=flat values end-of-day battery energy at the flat price instead of the valley refill cost (fixed tariffs)
END_VALUE = os.environ.get("END_VALUE", "valley")
PV_FILE = {"uev": "shenzhen_hourly.csv", "SZH": "szh_2023_hourly.csv", "LOA": "loa_2023_hourly.csv",
           "MEL": "mel_2023_hourly.csv"}[DATASET]
PV_SHIFT_H = {"LOA": 1}.get(DATASET, 0)   # NASA local standard time -> civil clock (Pacific daylight time)


def day_tariff(day):
    """Hourly price and period labels (valley/flat/peak, used by the rule baseline) for one calendar day."""
    if PRICE is not None:
        return PRICE, PERIOD
    h = np.arange(T)
    wkday = pd.Timestamp(day).dayofweek < 5
    if TARIFF == "sce":
        if 6 <= pd.Timestamp(day).month <= 9:
            p = np.where((h >= 16) & (h < 21), 0.59521 if wkday else 0.35999, 0.15808)
        else:
            p = np.where((h >= 16) & (h < 21), 0.40046, np.where((h >= 8) & (h < 16), 0.09772, 0.16777))
    else:
        p = np.where(wkday & (h >= 9) & (h < 21), 0.3428, 0.2230)
    lab = np.where(p == p.min(), "valley", np.where(p == p.max(), "peak", "flat"))
    return p.astype(float), list(lab)


P_MIN = PRICE.min() if PRICE is not None else {"sce": 0.15808, "vdo": 0.2230}[TARIFF]


def pv_per_kwp(index):
    """Hourly PV output (kWh per kWp) from NASA POWER GHI and 2-m temperature, on the load's clock."""
    f = ROOT / "data" / "nasa" / PV_FILE
    lines = f.read_text().splitlines()
    df = pd.read_csv(f, skiprows=lines.index("-END HEADER-") + 1)
    t = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY, hour=df.HR)) + pd.Timedelta(hours=PV_SHIFT_H)
    ghi = df.ALLSKY_SFC_SW_DWN.clip(lower=0).to_numpy()
    tcell = df.T2M.to_numpy() + ghi / 800 * 20
    out = ghi / 1000 * 0.85 * (1 - 0.004 * (tcell - 25))
    return pd.Series(out, index=t).reindex(index).fillna(0).to_numpy()


def build_lp():
    """Variables per hour: g (grid buy), x (export), c (charge), d (discharge), s (shed); then soc_0..soc_24.

    Supply must cover load (g - x - c + d + s >= L - PV); any further PV surplus is curtailed.
    """
    nv = 5 * T + T + 1
    U = lil_matrix((T, nv))
    E = lil_matrix((T + 1, nv))
    for t in range(T):
        g, x, c, d, s = (5 * t + k for k in range(5))
        U[t, g], U[t, x], U[t, c], U[t, d], U[t, s] = -1, 1, 1, -1, -1
        so, sn = 5 * T + t, 5 * T + t + 1
        E[t, sn], E[t, so], E[t, c], E[t, d] = 1, -1, -ETA, 1 / ETA
    E[T, 5 * T] = 1
    return U.tocsr(), E.tocsr(), nv


A_UB, A_EQ, NV = build_lp()
P_REFILL = (P_FLAT if END_VALUE == "flat" else P_MIN) / ETA   # value of end-of-day energy relative to 50%


def hour_bounds(net, G, pb):
    """Export only PV surplus and shed only load above G, exactly as realise() settles them; otherwise the LP
    could export battery energy or shed instead of buying at peak."""
    return [b for n in net for b in ((0, G), (0, min(G, max(-n, 0.0))), (0, pb), (0, pb), (0, max(n - G, 0.0)))]


def solve_hub(cost, A_ub, A_eq, b_eq, bounds, net, price):
    """Solve the hub schedule; returns the variable vector or None if infeasible.

    realise() settles the net flow of each hour, so an hour either imports or exports. When the feed-in price
    exceeds the import price in a PV-surplus hour that settlement is non-convex (the LP would buy to charge and
    export PV at once), so those hours get a binary import/export switch and the problem becomes a small MILP.
    """
    wash = [t for t in range(len(net)) if net[t] < 0 and price[t] < P_FIT]
    if not wash:
        r = linprog(cost, A_ub=A_ub, b_ub=-net, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        return r.x if r.status == 0 else None
    nv, k = len(cost), len(wash)
    S = lil_matrix((2 * k, nv + k))
    ub = np.zeros(2 * k)
    for i, t in enumerate(wash):
        S[2 * i, 5 * t], S[2 * i, nv + i] = 1, -bounds[5 * t][1]             # import only if y = 1
        S[2 * i + 1, 5 * t + 1], S[2 * i + 1, nv + i] = 1, bounds[5 * t + 1][1]  # export only if y = 0
        ub[2 * i + 1] = bounds[5 * t + 1][1]
    pad = csr_matrix((A_ub.shape[0], k))
    cons = [LinearConstraint(hstack([A_ub, pad]), -np.inf, -net),
            LinearConstraint(hstack([A_eq, csr_matrix((A_eq.shape[0], k))]), b_eq, b_eq),
            LinearConstraint(S.tocsr(), -np.inf, ub)]
    lb, hb = np.array(bounds + [(0, 1)] * k, dtype=float).T
    r = milp(np.r_[cost, np.zeros(k)], constraints=cons, integrality=np.r_[np.zeros(nv), np.ones(k)],
             bounds=Bounds(lb, hb))
    return r.x[:nv] if r.status == 0 else None


def solve_da(net, G, pb, emax, voll, price=PRICE):
    """End-of-day energy is free but valued at P_REFILL, matching how realise() prices it."""
    cost = np.zeros(NV)
    for t in range(T):
        cost[5 * t:5 * t + 5] = [price[t], -P_FIT, CYCLE_EPS + DEG, CYCLE_EPS + DEG, voll]
    cost[5 * T + T] = -P_REFILL
    b = np.r_[np.zeros(T), 0.5 * emax]
    bounds = hour_bounds(net, G, pb) + [(0.1 * emax, 0.9 * emax)] * (T + 1)
    x = solve_hub(cost, A_UB, A_EQ, b, bounds, net, price)
    if x is None:
        raise RuntimeError("hub schedule infeasible")
    z = x[:5 * T].reshape(T, 5)
    return z[:, 2], z[:, 3]


def realise(net, plan, hub, protect=True, price=PRICE, period=PERIOD):
    """Run one day hour by hour. plan[t] is the scheduled net discharge (d - c), or None for the rule baseline.

    With protect=True the controller deviates from the plan only to avoid shedding load above G and to keep
    battery energy from being exported; a perfect-foresight plan never triggers either, so the oracle stays
    optimal for this settlement. State of charge is tracked and the end-of-day gap to 50% is priced at
    P_REFILL (valley refill cost, or the flat price with END_VALUE=flat).
    """
    G, pb, E, voll = hub["G"], hub["pb"], hub["emax"], hub["voll"]
    soc = 0.5 * E
    g = np.zeros(T)
    x = np.zeros(T)
    shed = np.zeros(T)
    thru = 0.0
    for t in range(T):
        lo = -min(pb, max(0.9 * E - soc, 0) / ETA)          # most charging possible
        hi = min(pb, max(soc - 0.1 * E, 0) * ETA)           # most discharging possible
        if plan is None:                                    # forecast-free rule: fill at valley, cover peak
            b = lo if period[t] == "valley" else (min(max(net[t], 0), hi) if period[t] == "peak" else 0.0)
        else:
            b = plan[t]
        if protect:
            if net[t] - b > G:
                b = net[t] - G
            elif b > 0 and net[t] - b < 0:
                b = max(net[t], 0.0)
        b = float(np.clip(b, lo, hi))
        soc += -b / ETA if b >= 0 else -b * ETA
        thru += abs(b)
        r = net[t] - b
        g[t], shed[t], x[t] = min(max(r, 0), G), max(r - G, 0), min(max(-r, 0), G)
    term = P_REFILL * (0.5 * E - soc)
    energy, export = float(price @ g), float(P_FIT * x.sum())
    return dict(energy=energy, export=export, shed_kwh=float(shed.sum()), shed=float(voll * shed.sum()),
                terminal=float(term), throughput=thru, degradation=DEG * thru,
                total=energy - export + float(voll * shed.sum()) + float(term) + DEG * thru)


def one_station(args):
    j, sid, hub, days = args
    out = []
    nobatt = dict(hub, pb=0.0, emax=0.0)
    for k in range(days["true"].shape[0]):
        L, pv = days["true"][k], days["pv"][k]
        if L.sum() <= 0:
            continue
        tp = dict(zip(("price", "period"), day_tariff(days["dates"][k])))
        out.append(dict(station=sid, model="no_battery", day=k, **realise(L - pv, np.zeros(T), nobatt, **tp)))
        out.append(dict(station=sid, model="rule", day=k, **realise(L - pv, None, hub, **tp)))
        for name, P in days["preds"].items():
            c, d = solve_da(P[k] - pv, hub["G"], hub["pb"], hub["emax"], hub["voll"], tp["price"])
            out.append(dict(station=sid, model=name, day=k, **realise(L - pv, d - c, hub, **tp)))
            if days["rigid"] and name != "oracle":
                out.append(dict(station=sid, model=name + "|rigid", day=k,
                                **realise(L - pv, d - c, hub, protect=False, **tp)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv", type=float, default=0.3)
    ap.add_argument("--batt", type=float, default=0.25)
    ap.add_argument("--gq", type=float, default=0.95)
    ap.add_argument("--period", default="test", choices=["cal", "test"])
    ap.add_argument("--tag", default="fx_base")
    ap.add_argument("--pattern", nargs="+", default=["preds_A_*.npz"])
    ap.add_argument("--rigid", action="store_true", help="also score strict schedule following")
    ap.add_argument("--skip-tag", default=None, help="skip forecasts already scored under tags with this prefix")
    args = ap.parse_args()

    v = load_volume()
    ids = v.columns.to_numpy()
    sp = SPLITS["A"]
    fit = v.iloc[hour_index(v.index, *sp["fit"])]
    pv1 = pv_per_kwp(v.index)
    fit_pv_day = pv1[hour_index(v.index, *sp["fit"])].sum() / (len(fit) / 24)
    st = pd.read_csv(DATA / "stations.csv", index_col=0).loc[ids]
    voll = VOLL_MULT * pd.read_csv(DATA / "retail_price.csv", index_col=0)["retail"].reindex(ids).fillna(1.7).to_numpy()
    e_day = fit.sum().to_numpy() / (len(fit) / 24)
    hubs = [dict(kwp=args.pv * e_day[j] / fit_pv_day, emax=args.batt * e_day[j],
                 pb=args.batt * e_day[j] / 2, G=float(np.quantile(fit.iloc[:, j], args.gq)), voll=voll[j])
            for j in range(len(ids))]

    files = sorted({f for p in args.pattern for f in glob.glob(str(OUT / p))})
    if args.skip_tag:
        done = set().union(*[set(pd.read_parquet(f, columns=["model"]).model)
                             for f in OUT.glob(f"cost_{args.period}_{args.skip_tag}*.parquet")])
        files = [f for f in files if Path(f).stem.replace("preds_A_", "") not in done]
    print("scoring", len(files), "forecasts", flush=True)
    first = np.load(files[0], allow_pickle=True)
    truth = first[f"{args.period}_true"]
    origins = pd.to_datetime(first[f"{args.period}_origins"])
    preds = {Path(f).stem.replace("preds_A_", ""): np.load(f)[f"{args.period}_pred"] for f in files}
    pos = np.array([v.index.get_loc(o) for o in origins])
    pv_days = np.stack([pv1[p:p + T] for p in pos])
    jobs = []
    for j, sid in enumerate(ids):
        P = {name: z[:, :, j] for name, z in preds.items()}
        P["oracle"] = truth[:, :, j]
        jobs.append((j, sid, hubs[j], dict(preds=P, true=truth[:, :, j], pv=pv_days * hubs[j]["kwp"],
                                           rigid=args.rigid, dates=origins)))
    jobs = jobs[:int(os.environ.get("STATION_LIMIT", len(jobs)))]
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        rows = [r for chunk in pool.imap_unordered(one_station, jobs, chunksize=4) for r in chunk]
    df = pd.DataFrame(rows)
    df["day"] = origins[df["day"].to_numpy()].strftime("%Y-%m-%d")
    orc = df[df.model == "oracle"].set_index(["station", "day"]).total
    df["regret"] = df.total - orc.reindex(pd.MultiIndex.from_frame(df[["station", "day"]])).to_numpy()
    df.to_parquet(OUT / f"cost_{args.period}_{args.tag}.parquet")
    pd.DataFrame(hubs, index=ids).to_csv(OUT / f"hubs_{args.tag}.csv")
    summ = df.groupby("model")[["total", "regret", "shed_kwh"]].mean().sort_values("regret")
    print(json.dumps(vars(args)))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
