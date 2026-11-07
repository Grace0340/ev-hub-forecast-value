"""Cost-aware forecasts: regret by family x loss for each controller (UrbanEV, test).

Loss labels: ph/cost use the preliminary weights (config/prelim_cost_weights.json); phd/costd use weights
estimated from day-ahead-following regret (fx_base), phm/costm from rolling re-optimisation regret (fx_mpc).
Regret is averaged over seeds. Writes outputs/summary/costaware.csv and costaware.txt.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT, SUMMARY

ROOT = SUMMARY
LOSSES = ["mse", "mae", "pinball70", "pinball90", "ph", "cost", "phd", "costd", "phm", "costm"]


def regret(tags):
    files = [f for t in tags for f in OUT.glob(f"cost_test_{t}*.parquet")]
    c = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(["model", "station", "day"])
    c = c[c.model.str.count("_") == 2]
    m = c.groupby("model").regret.mean()
    parts = m.index.str.split("_")
    return pd.DataFrame({"family": parts.str[0], "loss": parts.str[1], "regret": m.to_numpy()}) \
        .groupby(["family", "loss"]).regret.mean().unstack().reindex(columns=LOSSES)


def accuracy():
    rows = []
    for f in sorted(glob.glob(str(OUT / "preds_A_*.npz"))):
        fam, loss, _ = Path(f).stem.replace("preds_A_", "").split("_")
        z = np.load(f, allow_pickle=True)
        e = z["test_pred"] - z["test_true"]
        rows.append(dict(family=fam, loss=loss, mae=float(np.abs(e).mean()), bias=float(e.mean())))
    return pd.DataFrame(rows).groupby(["family", "loss"]).mean()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    tabs = {"day-ahead": regret(["fx_base", "cw_da"]), "rolling": regret(["fx_mpc", "cw_mpc"])}
    acc = accuracy()
    lines = []
    for k, t in tabs.items():
        lines += [f"== regret, {k} controller (test, mean over seeds)", t.round(2).to_string(), ""]
    for k in ("mae", "bias"):
        lines += [f"== test {k}", acc[k].unstack().reindex(columns=LOSSES).round(3).to_string(), ""]
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "costaware.txt").write_text(txt)
    pd.concat({k: t.stack() for k, t in tabs.items()}, names=["controller"]).rename("regret") \
        .to_csv(ROOT / "costaware.csv")


if __name__ == "__main__":
    main()
