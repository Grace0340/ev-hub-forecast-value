"""Paths, chronological splits and shared helpers.

DATASET=uev (default) selects UrbanEV Shenzhen; DATASET=SZH, LOA or MEL selects a CHARGED city.
EV_HUB_ROOT is the project root holding data/ and outputs/ (default: the repository root).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("EV_HUB_ROOT", REPO))
PRELIM_WEIGHTS = REPO / "config" / "prelim_cost_weights.json"
DATASET = os.environ.get("DATASET", "uev")
LOOKBACK, HORIZON = 168, 24


def dataset_dir(ds):
    return "urbanev" if ds == "uev" else ds


DATA = ROOT / "data" / "proc" / dataset_dir(DATASET)
OUT = ROOT / "outputs" / dataset_dir(DATASET)
SUMMARY = ROOT / "outputs" / "summary"
if DATASET == "uev":
    # Spring Festival 2023 (Jan 21-27) is never an evaluation target.
    SPLITS = {"A": dict(fit=("2022-09-01", "2022-11-30"), val=("2022-12-01", "2022-12-20"),
                        cal=("2022-12-21", "2023-01-20"), test=("2023-01-28", "2023-02-28"))}
else:
    SPLITS = {"A": dict(fit=("2023-04-01", "2023-06-30"), val=("2023-07-01", "2023-07-20"),
                        cal=("2023-07-21", "2023-08-20"), test=("2023-08-21", "2023-09-30"))}
OUT.mkdir(parents=True, exist_ok=True)


def load_volume():
    v = pd.read_csv(DATA / "volume.csv", index_col=0, parse_dates=True)
    v.columns = v.columns.astype(int)
    return v


def stations():
    return pd.read_csv(DATA / "stations.csv", index_col=0)


def knn_adjacency(st, k=6):
    kx = 111.32 * np.cos(np.deg2rad(st.latitude.mean()))
    xy = np.c_[st.longitude * kx, st.latitude * 110.57]
    D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(D, np.inf)
    A = np.zeros_like(D)
    np.put_along_axis(A, np.argsort(D, 1)[:, :k], 1.0, 1)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1)
    return A


def hour_index(index, start, end):
    m = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end) + pd.Timedelta(hours=23))
    return np.where(m)[0]


def day_origins(index, start, end):
    pos = hour_index(index, start, end)
    return [p for p in pos if index[p].hour == 0 and p >= LOOKBACK and p + HORIZON <= len(index)]
