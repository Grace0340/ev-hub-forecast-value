"""Post-hoc cost-aware shift of existing MSE forecasts (no retraining).

For each station and tariff period, the test forecast is shifted by the tau-quantile of calibration
residuals (truth - forecast) pooled over the period's hours, with tau = alpha / (alpha + beta) from the
calibration cost weights. Writes preds_A_<family>_<suffix>_s<seed>.npz; their cal_pred is in-sample.
Usage: DATASET=uev python posthoc_shift.py [--weights config/prelim_cost_weights.json --suffix ph]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from common import OUT, PRELIM_WEIGHTS

PERIOD = np.array(["valley"] * 8 + ["flat"] * 2 + ["peak"] * 2 + ["flat"] * 2 + ["peak"] * 5 + ["flat"] * 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(PRELIM_WEIGHTS))
    ap.add_argument("--suffix", default="ph", help="loss label of the shifted forecasts")
    args = ap.parse_args()
    w = json.loads(Path(args.weights).read_text())["weights"]
    tau = {p: w[p]["alpha"] / (w[p]["alpha"] + w[p]["beta"]) for p in w}
    print("tau per period:", {p: round(t, 3) for p, t in tau.items()})
    for f in sorted(glob.glob(str(OUT / "preds_A_*_mse_s*.npz"))):
        z = dict(np.load(f, allow_pickle=True))
        res = z["cal_true"] - z["cal_pred"]                      # days x 24 x stations
        shift = np.zeros(res.shape[1:])
        for p, t in tau.items():
            h = PERIOD == p
            shift[h] = np.quantile(res[:, h, :].reshape(-1, res.shape[2]), t, axis=0)[None, :]
        for k in ("cal", "test"):
            z[f"{k}_pred"] = np.clip(z[f"{k}_pred"] + shift[None], 0, None)
        out = Path(f).name.replace("_mse_", f"_{args.suffix}_")
        np.savez_compressed(OUT / out, **z)
        e = z["test_pred"] - z["test_true"]
        print(out, "test MAE", round(float(np.abs(e).mean()), 3), "bias", round(float(e.mean()), 3))


if __name__ == "__main__":
    main()
