"""Day-ahead station-level charging-load forecasts (daily origins at 00:00, 24-h horizon).

Saves calibration/test predictions in kWh to outputs/<dataset>/preds_{split}_{model}_{loss}_s{seed}.npz.
Losses: mse (default), mae, pinballXX (quantile XX/100), or cost (period-weighted pinball with --weights).
Usage: DATASET=uev python forecast.py [--models DLinear GRU --loss pinball90 --seeds 0 1 2]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (HORIZON, LOOKBACK, OUT, PRELIM_WEIGHTS, SPLITS, day_origins, hour_index, knn_adjacency,
                    load_volume, stations)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_feats(index):
    h, d = index.hour.to_numpy(), index.dayofweek.to_numpy()
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)], 1).astype(np.float32)


class DLinear(nn.Module):
    def __init__(self, n, k=25):
        super().__init__()
        self.pool = nn.AvgPool1d(k, stride=1, padding=k // 2, count_include_pad=False)
        self.lin_s, self.lin_t = nn.Linear(LOOKBACK, HORIZON), nn.Linear(LOOKBACK, HORIZON)

    def forward(self, x, xm):
        z = x.transpose(1, 2)
        trend = self.pool(z)
        return (self.lin_s(z - trend) + self.lin_t(trend)).transpose(1, 2)


class GRUNet(nn.Module):
    """Channel-independent GRU; optional neighbour-mean input and node embedding (graph variant)."""

    def __init__(self, n, adj=None, hid=64, emb=8):
        super().__init__()
        self.graph = adj is not None
        if self.graph:
            a = adj.copy()
            np.fill_diagonal(a, 0)
            a = a / np.maximum(a.sum(1, keepdims=True), 1)
            self.register_buffer("A", torch.tensor(a, dtype=torch.float32))
            self.node = nn.Parameter(torch.randn(n, emb) * 0.1)
        fin = 1 + 4 + (1 + emb if self.graph else 0)
        self.gru = nn.GRU(fin, hid, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hid, HORIZON)

    def forward(self, x, xm):
        b, l, n = x.shape
        feats = [x.unsqueeze(-1), xm.unsqueeze(2).expand(b, l, n, 4)]
        if self.graph:
            feats.append(torch.einsum("ij,blj->bli", self.A, x).unsqueeze(-1))
            feats.append(self.node.view(1, 1, n, -1).expand(b, l, n, -1))
        z = torch.cat(feats, -1).permute(0, 2, 1, 3).reshape(b * n, l, -1)
        out, _ = self.gru(z)
        return self.head(out[:, -1]).view(b, n, HORIZON).transpose(1, 2)


class ITransformer(nn.Module):
    def __init__(self, n, d=128, layers=2, heads=8):
        super().__init__()
        self.emb = nn.Linear(LOOKBACK, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.proj = nn.Linear(d, HORIZON)

    def forward(self, x, xm):
        mu, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-5
        z = ((x - mu) / sd).transpose(1, 2)
        y = self.proj(self.enc(self.emb(z))).transpose(1, 2)
        return y * sd + mu


def build(name, n, adj):
    return {"DLinear": lambda: DLinear(n), "GRU": lambda: GRUNet(n),
            "GraphGRU": lambda: GRUNet(n, adj), "iTransformer": lambda: ITransformer(n)}[name]()


PERIOD = ["valley"] * 8 + ["flat"] * 2 + ["peak"] * 2 + ["flat"] * 2 + ["peak"] * 5 + ["flat"] * 5


def cost_weights(index, origins, path):
    """Per-sample, per-step under/over weights (n, 24, 1) from the calibration cost decomposition."""
    w = json.loads(open(path).read())["weights"]
    a = np.array([w[p]["alpha"] for p in PERIOD])
    b = np.array([w[p]["beta"] for p in PERIOD])
    norm = (a.mean() + b.mean()) / 2
    hours = (index[origins].hour.to_numpy()[:, None] + np.arange(HORIZON)[None, :]) % 24
    return [torch.tensor((m[hours] / norm)[..., None], dtype=torch.float32) for m in (a, b)]


def loss_fn(kind):
    if kind == "mse":
        return nn.functional.mse_loss
    if kind == "mae":
        return nn.functional.l1_loss
    q = int(kind.replace("pinball", "")) / 100

    def pinball(pred, y):
        e = y - pred
        return torch.maximum(q * e, (q - 1) * e).mean()
    return pinball


def windows(Z, TM, origins):
    x = np.stack([Z[p - LOOKBACK:p] for p in origins])
    xm = np.stack([TM[p - LOOKBACK:p] for p in origins])
    y = np.stack([Z[p:p + HORIZON] for p in origins])
    return [torch.tensor(a) for a in (x, xm, y)]


def predict(model, Z, TM, origins, bs=16):
    model.eval()
    x, xm, _ = windows(Z, TM, origins)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            outs.append(model(x[i:i + bs].to(DEV), xm[i:i + bs].to(DEV)).cpu().numpy())
    return np.concatenate(outs)


def train(name, Z, TM, adj, idx, sp, seed, loss, epochs=60, patience=8, bs=32, lr=1e-3, weights=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit_pos = hour_index(idx, *sp["fit"])
    fit_orig = [p for p in fit_pos if p >= LOOKBACK and p + HORIZON - 1 <= fit_pos[-1]]
    fit_orig = fit_orig[::3] if "GRU" in name else fit_orig
    if "GRU" in name:                      # GRU runs one sequence per station: keep ~8,800 sequences per batch
        bs = max(1, 8800 // Z.shape[1])
    val_orig = day_origins(idx, *sp["val"])
    model = build(name, Z.shape[1], adj).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x, xm, y = windows(Z, TM, fit_orig)
    xv, xmv, yv = windows(Z, TM, val_orig)
    if loss == "cost":
        wa, wb = cost_weights(idx, fit_orig, weights)
        wav, wbv = [w.to(DEV) for w in cost_weights(idx, val_orig, weights)]

        def crit(pred, target, j=None):
            a, b = (wa[j].to(DEV), wb[j].to(DEV)) if j is not None else (wav, wbv)
            e = target - pred
            return (a * torch.relu(e) + b * torch.relu(-e)).mean()
    else:
        base = loss_fn(loss)

        def crit(pred, target, j=None):
            return base(pred, target)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), bs):
            j = perm[i:i + bs]
            l = crit(model(x[j].to(DEV), xm[j].to(DEV)), y[j].to(DEV), j)
            opt.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(xv[i:i + bs].to(DEV), xmv[i:i + bs].to(DEV)) for i in range(0, len(xv), bs)])
            vl = crit(pv, yv.to(DEV)).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["SNaive", "HistMean", "DLinear", "GRU", "GraphGRU", "iTransformer"])
    ap.add_argument("--splits", nargs="+", default=["A"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--weights", default=str(PRELIM_WEIGHTS), help="for --loss cost")
    ap.add_argument("--name", default=None, help="loss label in output names (default: --loss)")
    args = ap.parse_args()
    label = args.name or args.loss

    v = load_volume()
    idx = v.index
    ids = v.columns.to_numpy()
    Y = v.to_numpy(dtype=np.float32)
    adj = knn_adjacency(stations().loc[ids])
    TM = time_feats(idx)
    log = []
    for s in args.splits:
        sp = SPLITS[s]
        fit_pos = hour_index(idx, *sp["fit"])
        mu, sd = Y[fit_pos].mean(0), Y[fit_pos].std(0) + 1e-3
        Z = ((Y - mu) / sd).astype(np.float32)
        orig = {k: day_origins(idx, *sp[k]) for k in ("cal", "test")}
        truth = {k: np.stack([Y[p:p + HORIZON] for p in o]) for k, o in orig.items()}
        for m in args.models:
            for seed in (args.seeds if m not in ("SNaive", "HistMean") else [0]):
                t0 = time.time()
                info = {}
                if m == "SNaive":
                    pred = {k: np.stack([Y[p - 168:p - 168 + HORIZON] for p in o]) for k, o in orig.items()}
                elif m == "HistMean":
                    pred = {k: np.stack([np.mean([Y[p - w * 168:p - w * 168 + HORIZON] for w in (1, 2, 3, 4)], 0)
                                         for p in o]) for k, o in orig.items()}
                else:
                    model, vl, ep = train(m, Z, TM, adj, idx, sp, seed, args.loss, weights=args.weights)
                    pred = {k: np.clip(predict(model, Z, TM, o) * sd + mu, 0, None) for k, o in orig.items()}
                    info = {"val_loss": vl, "epochs": ep}
                tag = f"{s}_{m}_{label}_s{seed}"
                np.savez_compressed(OUT / f"preds_{tag}.npz", stations=ids,
                                    cal_pred=pred["cal"], cal_true=truth["cal"],
                                    test_pred=pred["test"], test_true=truth["test"],
                                    cal_origins=np.array([str(idx[p]) for p in orig["cal"]]),
                                    test_origins=np.array([str(idx[p]) for p in orig["test"]]))
                e = pred["test"] - truth["test"]
                rec = dict(split=s, model=m, loss=label, seed=seed, mae=float(np.abs(e).mean()),
                           rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                           wape=float(np.abs(e).sum() / truth["test"].sum()), sec=round(time.time() - t0, 1), **info)
                print(json.dumps(rec), flush=True)
                log.append(rec)
    pd.DataFrame(log).to_csv(OUT / f"forecast_metrics_{'_'.join(args.splits)}_{label}_{'_'.join(args.models)}.csv",
                             index=False)


if __name__ == "__main__":
    main()
