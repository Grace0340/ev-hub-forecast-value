"""UrbanEV controller result on the plain seed-0 pool, i.e. without the preliminary-weight shift (_ph) and cost-loss
(_cost) variants. Writes outputs/summary/plain.csv.
"""
import pandas as pd

import summary_robust
from summary_robust import DST, UEV, case_row

CASES = [("UrbanEV base", "fx_base", "fx_mpc"), ("test-period price", "fx_szt", "ms_szt")]


def main():
    orig = summary_robust.station_sums

    def plain(path):
        s, n = orig(path)
        return s[[m for m in s.columns if not any(k in m for k in ("_ph_", "_cost_"))]], n

    summary_robust.station_sums = plain
    rows = [case_row(name, UEV, ta, tm) for name, ta, tm in CASES]
    df = pd.DataFrame([r for r in rows if r]).set_index("case")
    df.to_csv(DST / "plain.csv")
    pd.set_option("display.width", 250)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
