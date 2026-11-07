"""Forecast accuracy against operating cost for one scoring tag.

Reads outputs/<dataset>/cost_{cal,test}_<tag>.parquet and the matching prediction files.
  1. aggregate: MAE / RMSE / bias versus mean regret for every forecast
  2. per station: choose a model on the calibration period by MAE or by cost, score both choices on test
  3. decomposition: regret per station-day regressed (station fixed effects) on under/over-forecast energy
     by tariff period and in hours where the realised net load exceeds the grid connection
Writes analysis_<tag>.txt, summary_<tag>.json and agg_<tag>.csv.
Usage: DATASET=uev TARIFF=sz python analysis.py --tag fx_base
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, load_volume
from cost_dayahead import day_tariff, pv_per_kwp

FAMILY_ORDER = ["SNaive", "HistMean", "DLinear", "GRU", "GraphGRU", "iTransformer"]


def load_preds():
    out = {}
    for f in sorted(glob.glob(str(OUT / "preds_A_*.npz"))):
        z = np.load(f, allow_pickle=True)
        out[Path(f).stem.replace("preds_A_", "")] = z
    return out


def error_table(preds, period, hubs, v):
    """Per forecast, station and day: absolute error and decomposed error energy (all in kWh)."""
    first = next(iter(preds.values()))
    truth = first[f"{period}_true"]                            # days x 24 x stations
    origins = pd.to_datetime(first[f"{period}_origins"])
    pv1 = pv_per_kwp(v.index)
    pos = np.array([v.index.get_loc(o) for o in origins])
    pv = np.stack([pv1[p:p + 24] for p in pos])[:, :, None] * hubs.kwp.to_numpy()[None, None, :]
    over_cap = (truth - pv) > hubs.G.to_numpy()[None, None, :]
    per = np.array([day_tariff(o)[1] for o in origins])       # days x 24
    rows = []
    for name, z in preds.items():
        e = z[f"{period}_pred"] - truth
        rec = {"model": name}
        rec["mae"] = np.abs(e).mean(1)
        rec["bias"] = e.mean(1)
        for p in ("valley", "flat", "peak"):
            m = (per == p)[:, :, None]
            rec[f"under_{p}"] = np.where(m, np.clip(-e, 0, None), 0).sum(1)
            rec[f"over_{p}"] = np.where(m, np.clip(e, 0, None), 0).sum(1)
        rec["under_cap"] = np.where(over_cap, np.clip(-e, 0, None), 0).sum(1)
        rec["over_cap"] = np.where(over_cap, np.clip(e, 0, None), 0).sum(1)
        cols = {k: val.ravel() for k, val in rec.items() if k != "model"}
        df = pd.DataFrame(cols)
        df["model"] = name
        df["day"] = np.repeat(origins.strftime("%Y-%m-%d").to_numpy(), truth.shape[2])
        df["station"] = np.tile(hubs.index.to_numpy(), truth.shape[0])
        df["day_kwh"] = truth.sum(1).ravel()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def split_name(m):
    if "_" not in m:
        return m, "", -1
    fam, loss, seed = m.split("_")
    return fam, loss, int(seed[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fx_base")
    args = ap.parse_args()
    v = load_volume()
    hubs = pd.read_csv(sorted(OUT.glob(f"hubs_{args.tag}*.csv"))[0], index_col=0).loc[v.columns]
    L = [f"tag={args.tag}"]
    res = {}
    for period in ("cal", "test"):
        files = sorted(OUT.glob(f"cost_{period}_{args.tag}*.parquet"))
        if not files:
            continue
        cost = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(["model", "station", "day"])
        cost["day"] = cost["day"].astype(str)
        err = error_table(load_preds(), period, hubs, v)
        cost = cost[~cost.model.str.contains("|", regex=False)]                    # strict-following variants
        res[period] = cost.merge(err, on=["model", "station", "day"], how="left")   # cost skips outage days

    test = res["test"]
    fc = test[~test.model.isin(["oracle", "rule", "no_battery"])].copy()
    fc[["family", "loss", "seed"]] = pd.DataFrame(fc.model.map(split_name).tolist(), index=fc.index)
    ref = test[test.model.isin(["oracle", "rule", "no_battery"])].groupby("model").total.mean()
    L.append("\nreference cost per station-day (CNY): " + ", ".join(f"{k}={v_:.2f}" for k, v_ in ref.items()))
    L.append(f"value of perfect information over the rule: {ref['rule'] - ref['oracle']:.2f}")

    # 1. aggregate ranking, seeds averaged within family x loss
    agg = fc.groupby(["family", "loss"]).agg(mae=("mae", "mean"), bias=("bias", "mean"), regret=("regret", "mean"),
                                             shed_kwh=("shed_kwh", "mean")).reset_index()
    agg["gain_vs_rule"] = ref["rule"] - (agg.regret + ref["oracle"])
    agg["rank_mae"] = agg.mae.rank()
    agg["rank_cost"] = agg.regret.rank()
    L.append("\n1. aggregate (test, mean per station-day):")
    L.append(agg.sort_values("regret").round(3).to_string(index=False))
    S = dict(tag=args.tag, n_stations=int(fc.station.nunique()), oracle=ref["oracle"], rule=ref["rule"],
             no_battery=ref.get("no_battery", np.nan), vpi=ref["rule"] - ref["oracle"],
             best_model=agg.sort_values("regret").iloc[0][["family", "loss"]].str.cat(sep="_"),
             best_gain=float(agg.gain_vs_rule.max()), mae_best_model=agg.sort_values("mae").iloc[0][
                 ["family", "loss"]].str.cat(sep="_"), mae_best_rank_cost=float(agg.sort_values("mae").iloc[0].rank_cost))
    rho, p = stats.spearmanr(agg.mae, agg.regret)
    L.append(f"Spearman(MAE, regret) across {len(agg)} forecasts: rho={rho:.3f} p={p:.2g}")
    S.update(rho_all=rho, p_all=p, n_all=len(agg))
    mse = agg[agg.loss.isin(["mse"])]
    rho, p = stats.spearmanr(mse.mae, mse.regret)
    L.append(f"   MSE-trained families only (n={len(mse)}): rho={rho:.3f} p={p:.2g}")
    S.update(rho_mse=rho, p_mse=p, n_mse=len(mse))
    comp = agg[agg.gain_vs_rule > 0]
    if len(comp) >= 3:
        rho, p = stats.spearmanr(comp.mae, comp.regret)
        L.append(f"   forecasts that beat the rule (n={len(comp)}): rho={rho:.3f} p={p:.2g}")
        S.update(rho_comp=rho, p_comp=p, n_comp=len(comp))

    # 2. per-station selection on calibration, scored on test (MSE-trained families, seed 0 + baselines)
    if "cal" in res:
        def per_station(df):
            d = df[~df.model.isin(["oracle", "rule", "no_battery"])].copy()
            d[["family", "loss", "seed"]] = pd.DataFrame(d.model.map(split_name).tolist(), index=d.index)
            d = d[d.seed <= 0]
            return d.groupby(["station", "model"]).agg(mae=("mae", "mean"), regret=("regret", "mean")).reset_index()
        cal, tst = per_station(res["cal"]), per_station(res["test"])
        pick_mae = cal.loc[cal.groupby("station").mae.idxmin(), ["station", "model"]]
        pick_cost = cal.loc[cal.groupby("station").regret.idxmin(), ["station", "model"]]
        t = tst.set_index(["station", "model"]).regret
        r_mae = t.reindex(pd.MultiIndex.from_frame(pick_mae)).to_numpy()
        r_cost = t.reindex(pd.MultiIndex.from_frame(pick_cost)).to_numpy()
        same = (pick_mae.model.to_numpy() == pick_cost.model.to_numpy()).mean()
        L.append("\n2. per-station model choice (calibration -> test):")
        L.append(f"   stations where the MAE pick differs from the cost pick: {1 - same:.3f}")
        L.append(f"   test regret, MAE pick: {np.nanmean(r_mae):.3f}; cost pick: {np.nanmean(r_cost):.3f}; "
                 f"paired t p={stats.ttest_rel(r_mae, r_cost, nan_policy='omit').pvalue:.2g}")
        best_mae = tst.loc[tst.groupby("station").mae.idxmin(), "model"].to_numpy()
        best_cost = tst.loc[tst.groupby("station").regret.idxmin(), "model"].to_numpy()
        L.append(f"   ex post on test: MAE-best differs from cost-best at {(best_mae != best_cost).mean():.3f} of stations")
        rhos = tst.groupby("station").apply(lambda g: stats.spearmanr(g.mae, g.regret)[0])
        L.append(f"   within-station Spearman(MAE, regret) over models: median={rhos.median():.3f}, "
                 f"share < 0.5: {(rhos < 0.5).mean():.3f}")
        S.update(pick_differ=1 - same, regret_mae_pick=np.nanmean(r_mae), regret_cost_pick=np.nanmean(r_cost),
                 p_pick=stats.ttest_rel(r_mae, r_cost, nan_policy="omit").pvalue,
                 wilcoxon_p_pick=stats.wilcoxon(r_mae, r_cost, nan_policy="omit").pvalue,
                 within_rho_median=rhos.median())

    # 3. decomposition of regret, scaled by station mean daily load, station fixed effects
    feats = ["under_valley", "over_valley", "under_flat", "over_flat", "under_peak", "over_peak",
             "under_cap", "over_cap"]
    d = fc.dropna(subset=feats).copy()
    scale = d.groupby("station").day_kwh.transform("mean")
    y = d.regret / scale
    X = d[feats].div(scale, axis=0)
    Xd = X - X.groupby(d.station).transform("mean")
    yd = y - y.groupby(d.station).transform("mean")
    beta, *_ = np.linalg.lstsq(Xd.to_numpy(), yd.to_numpy(), rcond=None)
    r2 = 1 - ((yd - Xd @ beta) ** 2).sum() / (yd ** 2).sum()
    ma = (d.mae * 24 / scale)
    mad = ma - ma.groupby(d.station).transform("mean")
    b1 = (mad @ yd) / (mad @ mad)
    r2_mae = 1 - ((yd - b1 * mad) ** 2).sum() / (yd ** 2).sum()
    L.append("\n3. regret decomposition (CNY per kWh of error; within-station; scaled by mean daily load):")
    L.append(f"   R2 with absolute error only: {r2_mae:.3f}; with decomposed errors: {r2:.3f}")
    share = (X.mean() * beta) / (X.mean() * beta).sum()
    L.append(pd.DataFrame({"coef_CNY_per_kWh": beta, "mean_error_share_of_load": X.mean().to_numpy(),
                           "share_of_explained_regret": share.to_numpy()}, index=feats).round(3).to_string())
    S.update(r2_mae=r2_mae, r2_dec=r2, **{f"coef_{k}": b for k, b in zip(feats, beta)})
    text = "\n".join(L)
    print(text)
    (OUT / f"analysis_{args.tag}.txt").write_text(text)
    (OUT / f"summary_{args.tag}.json").write_text(json.dumps({k: (v_.item() if hasattr(v_, "item") else v_)
                                                              for k, v_ in S.items()}, indent=1))
    agg.to_csv(OUT / f"agg_{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
