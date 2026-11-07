"""Summary tables from the analysis.py outputs.

  1. hub-configuration and tariff sensitivity (UrbanEV, day-ahead following)
  2. datasets x controller: day-ahead following (fx_base) and rolling re-optimisation (fx_mpc)
  3. per-forecast regret under both controllers on identical seed-0 forecasts, for every dataset
Writes outputs/summary/{sensitivity,datasets,controller_<ds>}.csv and summary.txt.
"""
import json

import pandas as pd
from scipy import stats

from common import ROOT, SUMMARY

DST = SUMMARY
DST.mkdir(parents=True, exist_ok=True)
DIRS = {"UrbanEV": ROOT / "outputs" / "urbanev", "SZH": ROOT / "outputs" / "SZH", "LOA": ROOT / "outputs" / "LOA",
        "MEL": ROOT / "outputs" / "MEL"}
COLS = ["n_stations", "oracle", "rule", "vpi", "best_model", "best_gain", "mae_best_model", "mae_best_rank_cost",
        "rho_all", "rho_mse", "rho_comp", "n_comp", "pick_differ", "regret_mae_pick", "regret_cost_pick", "p_pick",
        "wilcoxon_p_pick", "r2_mae", "r2_dec", "coef_under_peak", "coef_over_peak"]
L = []


def row(case, path):
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    r = dict(case=case, **{k: s.get(k) for k in COLS})
    r["vpi_share_of_rule"] = s["vpi"] / s["rule"]
    r["best_share_of_vpi"] = s["best_gain"] / s["vpi"]
    if s.get("regret_mae_pick"):
        r["pick_gain_pct"] = 100 * (1 - s["regret_cost_pick"] / s["regret_mae_pick"])
    return r


def show(title, rows, name):
    rows = [r for r in rows if r]
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("case")
    df.to_csv(DST / f"{name}.csv")
    L.extend([title, df.drop(columns=["regret_mae_pick", "regret_cost_pick"]).round(3).T.to_string(), ""])


show("1. sensitivity (UrbanEV, day-ahead following, test):",
     [row(c, DIRS["UrbanEV"] / f"summary_{t}.json") for c, t in
      [("base", "fx_base"), ("PV 0", "fx_pv0"), ("PV 50%", "fx_pv50"), ("battery 10%", "fx_b10"),
       ("battery 50%", "fx_b50"), ("grid P90", "fx_g90"), ("grid P99", "fx_g99"), ("Guangdong tariff", "fx_gd"),
       ("test-period price", "fx_szt"), ("shed price x0.5", "fx_v05"), ("shed price x2", "fx_v2"),
       ("end-of-day at flat price", "fx_tflat")]],
     "sensitivity")
show("2. datasets x controller (test):",
     [row(f"{ds} {ctl}", d / f"summary_{tag}.json") for ds, d in DIRS.items()
      for ctl, tag in [("day-ahead", "fx_base"), ("rolling", "fx_mpc")]], "datasets")

L.append("3. regret per seed-0 forecast, day-ahead following vs rolling re-optimisation (test):")
for ds, d in DIRS.items():
    fa, fb = d / "cost_test_fx_base.parquet", d / "cost_test_fx_mpc.parquet"
    if not (fa.exists() and fb.exists()):
        continue
    da, mp = pd.read_parquet(fa), pd.read_parquet(fb)
    da = da[da.model.isin(mp.model.unique())].merge(mp[["station", "day"]].drop_duplicates(), on=["station", "day"])
    g = pd.DataFrame({"regret_da": da.groupby("model").regret.mean(), "regret_mpc": mp.groupby("model").regret.mean()})
    fc = g.drop(index=["oracle", "rule"], errors="ignore")
    g["recovered"] = 1 - g.regret_mpc / g.regret_da
    g["rank_da"], g["rank_mpc"] = fc.regret_da.rank(), fc.regret_mpc.rank()
    rho, p = stats.spearmanr(fc.regret_da, fc.regret_mpc)
    g.sort_values("regret_mpc").to_csv(DST / f"controller_{ds}.csv")
    L.extend([f"-- {ds}: Spearman(regret_da, regret_mpc) over {len(fc)} forecasts rho={rho:.3f} p={p:.2g}",
              g.sort_values("regret_mpc").round(3).to_string(), ""])

text = "\n".join(L)
print(text)
(DST / "summary.txt").write_text(text)
