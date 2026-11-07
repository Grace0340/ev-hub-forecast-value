"""Rolling re-optimisation without observing the current hour (tag np_mpc) against day-ahead following (fx_base)
and the base rolling controller (fx_mpc), per dataset; plus the bootstrap distribution of the base-case
controller rank correlation. Writes outputs/summary/nopeek.csv and nopeek.txt.
"""
import numpy as np
import pandas as pd

from summary_robust import DST, ROOT, UEV, boot_spearman, case_row, station_sums

SETS = [("UrbanEV", UEV), ("CHARGED SZH", ROOT / "SZH"), ("CHARGED LOA", ROOT / "LOA"),
        ("CHARGED MEL", ROOT / "MEL")]
B = 1000


def boot_dist(X, Y, w, seed=1):
    def rho(a, b):
        ra, rb = a.argsort(-1).argsort(-1).astype(float), b.argsort(-1).argsort(-1).astype(float)
        ra, rb = ra - ra.mean(-1, keepdims=True), rb - rb.mean(-1, keepdims=True)
        return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), (B, len(w)))
    ws = w[idx].sum(1, keepdims=True)
    return rho(X[idx].sum(1) / ws, Y[idx].sum(1) / ws)


def main():
    rows, lines = [], []
    for name, d in SETS:
        r = case_row(name, d, "fx_base", "np_mpc")
        if r is None:
            continue
        sp, n = station_sums(d / "cost_test_fx_mpc.parquet")
        sn, _ = station_sums(d / "cost_test_np_mpc.parquet")
        models = [m for m in sn.columns if m.count("_") == 2 and m in sp.columns]
        st = n.index
        P, N, w = sp.loc[st, models].to_numpy(), sn.loc[st, models].to_numpy(), n.to_numpy(dtype=float)
        rp = boot_spearman(P, N, w)
        r["rho_peek_nopeek"] = f"{rp[0]:.2f} [{rp[1]:.2f}, {rp[2]:.2f}]"
        r["share_peek_best"] = round(1 - (P.sum(0) / w.sum()).min() / r["vpi"], 3)
        r["regret_rise_mean"] = round(float((N.sum(0) / P.sum(0)).mean() - 1), 3)
        rows.append(r)
    df = pd.DataFrame(rows).set_index("case")
    df.to_csv(DST / "nopeek.csv")
    pd.set_option("display.width", 250)
    lines += ["1. rolling re-optimisation without observing the current hour (np_mpc), test:", df.T.to_string(), ""]

    sa, n = station_sums(UEV / "cost_test_fx_base.parquet")
    sm, _ = station_sums(UEV / "cost_test_fx_mpc.parquet")
    models = [m for m in sm.columns if m.count("_") == 2 and m in sa.columns]
    st = n.index
    A, M, w = sa.loc[st, models].to_numpy(), sm.loc[st, models].to_numpy(), n.to_numpy(dtype=float)
    dist = boot_dist(A, M, w)
    lines += ["2. UrbanEV base, bootstrap distribution of Spearman(regret_da, regret_mpc), "
              f"{len(models)} forecasts, B={B}:",
              f"   median {np.median(dist):.3f}; 2.5/25/75/97.5% {np.quantile(dist, [0.025, 0.25, 0.75, 0.975]).round(3)}; "
              f"share < 0: {(dist < 0).mean():.3f}"]
    txt = "\n".join(lines)
    print(txt)
    (DST / "nopeek.txt").write_text(txt)


if __name__ == "__main__":
    main()
