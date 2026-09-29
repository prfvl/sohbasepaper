import numpy as np, pickle
from pathlib import Path
from scipy.signal import savgol_filter

N = 300
PROC = Path("../data/processed")

def seg_features(s, lo, hi, n=N):
    v, q, T = s["v"], s["q"], s["temp"]
    ok = np.isfinite(v) & np.isfinite(q) & np.isfinite(T) & (v >= lo - 0.05)
    v, q, T = v[ok], q[ok], T[ok]
    if len(v) < 50: return None
    v = savgol_filter(v, 15, 3); q = savgol_filter(q, 15, 3)
    v = np.minimum.accumulate(v)
    keep = np.r_[True, np.diff(v) < 0]
    v, q, T = v[keep], q[keep], T[keep]
    if len(v) < 20: return None
    grid = np.linspace(hi, lo, n)
    o = np.argsort(v)
    qg = np.interp(grid, v[o], q[o]); Tg = np.interp(grid, v[o], T[o])
    ica = -np.gradient(qg, grid)                       # positive
    dv = 1.0 / np.clip(ica, 1e-3, None)
    dv = np.clip(dv, *np.percentile(dv, [1, 99]))
    return np.stack([ica, dv, Tg])

def volt_range(segs_all):
    vmin = [np.nanmin(s["v"][s["v"] > 3]) for s in segs_all if (s["v"] > 3).sum() > 50]
    vmax = [np.nanmax(s["v"]) for s in segs_all if (s["v"] > 3).sum() > 50]
    return float(np.percentile(vmin, 95)) + 0.05, float(np.percentile(vmax, 5)) - 0.05

def build_features(tab, lo=None, hi=None):
    tab = tab[tab.use].reset_index(drop=True)
    cache = {p: pickle.load(open(PROC/"segments"/f"{p}.pkl", "rb")) for p in tab.pack.unique()}
    if lo is None or hi is None:
        lo, hi = volt_range([cache[r.pack][int(r.seg)] for r in tab.itertuples()])
    X, M = [], []
    for r in tab.itertuples():
        f = seg_features(cache[r.pack][int(r.seg)], lo, hi)
        if f is not None and np.isfinite(f).all():
            X.append(f); M.append((r.pack, r.group, int(r.seg), r.SOH))
    X = np.stack(X)
    np.savez(PROC/"features_v1.npz", X=X, lo=lo, hi=hi,
             pack=[m[0] for m in M], group=[m[1] for m in M],
             seg=[m[2] for m in M], y=[m[3] for m in M])
    return X, M, lo, hi
