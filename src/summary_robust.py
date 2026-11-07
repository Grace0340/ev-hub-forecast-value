"""Robustness of the controller result: hub/tariff/degradation sensitivity under both controllers, seeds, and
station-bootstrap confidence intervals.

For each case the same seed-0 forecasts are scored under day-ahead following and rolling re-optimisation (test).
Reports the best forecast per controller, the share of the perfect-forecast value it captures, the top three
under rolling, Spearman(regret_da, regret_mpc) and Spearman(MAE, regret) with 95% station-bootstrap intervals.
Seeds: rank agreement of rolling regret between seeds 0-2 and the seed-averaged day-ahead vs rolling correlation.
Writes outputs/summary/robust.csv and robust.txt.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT as PROJECT, SUMMARY

ROOT = PROJECT / "outputs"
DST = SUMMARY
DST.mkdir(parents=True, exist_ok=True)
B = 1000
RNG = np.random.default_rng(0)
UEV = ROOT / "urbanev"
CASES = [("UrbanEV base", UEV, "fx_base", "fx_mpc"), ("PV 0", UEV, "fx_pv0", "ms_pv0"),
         ("PV 50%", UEV, "fx_pv50", "ms_pv50"), ("battery 10%", UEV, "fx_b10", "ms_b10"),
         ("battery 50%", UEV, "fx_b50", "ms_b50"), ("Guangdong tariff", UEV, "fx_gd", "ms_gd"),
         ("degradation 0.025", UEV, "fx_deg25", "ms_deg25"), ("degradation 0.05", UEV, "fx_deg50", "ms_deg50")] + \
        [(f"CHARGED {c} base", ROOT / c, "fx_base", "fx_mpc") for c in ("SZH", "LOA", "MEL")] + \
        [("test-period price", UEV, "fx_szt", "ms_szt"), ("shed price x0.5", UEV, "fx_v05", "ms_v05"),
         ("shed price x2", UEV, "fx_v2", "ms_v2"), ("end-of-day at flat price", UEV, "fx_tflat", "ms_tflat")]


def station_sums(path):
    """Per-station regret sums (stations x models, including the rule) and day counts."""
    c = pd.read_parquet(path, columns=["station", "model", "day", "regret"])
    c["station"] = c.station.astype(str)
    s = c.pivot_table(index="station", columns="model", values="regret", aggfunc="sum")
    n = c[c.model == "oracle"].groupby("station").size()
    return s, n


def mae_sums(d, models, stations):
    out = {}
    for m in models:
        z = np.load(d / f"preds_A_{m}.npz", allow_pickle=True)
        e = np.abs(z["test_pred"] - z["test_true"]).sum(axis=(0, 1))
        out[m] = pd.Series(e / (z["test_pred"].shape[0] * z["test_pred"].shape[1]), index=z["stations"].astype(str))
    return pd.DataFrame(out).reindex(stations)


def boot_spearman(X, Y, w):
    """Spearman over columns of station-weighted means, point estimate and 95% bootstrap interval."""
    def rho(a, b):
        ra, rb = stats.rankdata(a, axis=-1), stats.rankdata(b, axis=-1)
        ra, rb = ra - ra.mean(-1, keepdims=True), rb - rb.mean(-1, keepdims=True)
        return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))
    point = rho(X.sum(0) / w.sum(), Y.sum(0) / w.sum())
    idx = RNG.integers(0, len(w), (B, len(w)))
    ws = w[idx].sum(1, keepdims=True)
    r = rho(X[idx].sum(1) / ws, Y[idx].sum(1) / ws)
    return point, np.quantile(r, 0.025), np.quantile(r, 0.975)


def case_row(name, d, ta, tm):
    fa, fm = d / f"cost_test_{ta}.parquet", d / f"cost_test_{tm}.parquet"
    if not (fa.exists() and fm.exists()):
        return None
    sm, n = station_sums(fm)
    models = [m for m in sm.columns if m.count("_") == 2]
    sa, _ = station_sums(fa)
    models = [m for m in models if m in sa.columns]
    st = n.index
    A, M, w = sa.loc[st, models].to_numpy(), sm.loc[st, models].to_numpy(), n.to_numpy(dtype=float)
    E = mae_sums(d, models, st).to_numpy() * w[:, None]
    ra, rm = A.sum(0) / w.sum(), M.sum(0) / w.sum()
    vpi = sm.loc[st, "rule"].sum() / w.sum()
    order = np.argsort(rm)
    r_dm = boot_spearman(A, M, w)
    r_ea = boot_spearman(E, A, w)
    r_em = boot_spearman(E, M, w)
    f = lambda r: f"{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]"
    return dict(case=name, n_forecasts=len(models), vpi=round(vpi, 2),
                best_da=models[int(np.argmin(ra))], share_da=round(1 - ra.min() / vpi, 3),
                best_mpc=models[order[0]], share_mpc=round(1 - rm.min() / vpi, 3),
                top3_mpc=", ".join(models[i].rsplit("_", 1)[0] for i in order[:3]),
                mae_best_rank_mpc=int(np.argsort(order)[int(np.argmin(E.sum(0)))] + 1),
                rho_da_mpc=f(r_dm), rho_mae_da=f(r_ea), rho_mae_mpc=f(r_em))


def seeds():
    frames = []
    for tag in ("fx_mpc", "rb_seed"):
        c = pd.read_parquet(UEV / f"cost_test_{tag}.parquet", columns=["model", "regret"])
        frames.append(c[c.model.str.count("_") == 2].groupby("model").regret.mean())
    r = pd.concat(frames)
    df = pd.DataFrame({"fl": r.index.str.rsplit("_", n=1).str[0], "seed": r.index.str[-1], "regret": r.to_numpy()})
    tab = df.pivot(index="fl", columns="seed", values="regret")
    full = tab.dropna()
    lines = [f"rolling regret by seed ({len(full)} forecasts with all three seeds):", tab.round(2).to_string(), ""]
    for a, b in (("0", "1"), ("0", "2"), ("1", "2")):
        rho, p = stats.spearmanr(full[a], full[b])
        lines.append(f"Spearman seed {a} vs {b}: {rho:.3f} (p={p:.2g})")
    da = pd.read_parquet(UEV / "cost_test_fx_base.parquet", columns=["model", "regret"])
    da = da[da.model.str.count("_") == 2].groupby("model").regret.mean()
    da = da.groupby(da.index.str.rsplit("_", n=1).str[0]).mean()
    mp = tab.mean(axis=1)
    common = mp.index.intersection(da.index)
    rho, p = stats.spearmanr(da[common], mp[common])
    lines.append(f"seed-averaged Spearman(regret_da, regret_mpc) over {len(common)} forecasts: {rho:.3f} (p={p:.2g})")
    sd = tab.std(axis=1).mean()
    lines.append(f"mean across-seed std of rolling regret: {sd:.3f}; spread of seed-mean regret across forecasts: "
                 f"{mp.min():.2f}-{mp.max():.2f}")
    top = mp.sort_values()
    lines.append("seed-averaged rolling top 5: " + ", ".join(f"{k} {v:.2f}" for k, v in top.head(5).items()))
    return lines


def main():
    rows = [r for r in (case_row(*c) for c in CASES) if r]
    df = pd.DataFrame(rows).set_index("case")
    df.to_csv(DST / "robust.csv")
    pd.set_option("display.width", 250)
    lines = ["1. controller result across cases (test; rho with 95% station-bootstrap interval):",
             df.T.to_string(), "", "2. seeds (UrbanEV base, rolling re-optimisation):"]
    lines += seeds() if (UEV / "cost_test_rb_seed.parquet").exists() else ["(rb_seed not available)"]
    txt = "\n".join(lines)
    print(txt)
    (DST / "robust.txt").write_text(txt)


if __name__ == "__main__":
    main()
