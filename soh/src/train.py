import numpy as np, torch, torch.nn as nn, copy
from pathlib import Path
from .model import SOHNet
torch.set_num_threads(4)
PROC = Path("../data/processed")

def load(name="features_v1.npz"):
    d = np.load(PROC/name, allow_pickle=True)
    X = d["X"].astype(np.float32).copy(); X[:,1] = np.log1p(X[:,1])
    return X, d["y"].astype(np.float32), d["pack"], d["group"]

def scale_fit(X): return X.min((0,2), keepdims=True), X.max((0,2), keepdims=True)
def scale(X, mm): mn, mx = mm; return np.clip((X-mn)/(mx-mn+1e-8), -0.2, 1.2).astype(np.float32)

def metrics(y, p):
    e = p - y
    ss = ((y-y.mean())**2).sum()
    return dict(rmse=float(np.sqrt((e**2).mean())), mae=float(np.abs(e).mean()),
                r2=float(1-(e**2).sum()/ss) if ss > 0 else np.nan,
                mape=float((np.abs(e)/y).mean()*100), hit=float((np.abs(e) <= 0.015).mean()*100))

def predict(m, X, bs=256):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs): out.append(m(torch.from_numpy(X[i:i+bs])).numpy())
    return np.concatenate(out)

def fit(Xtr, ytr, Xva, yva, seed=0, epochs=200, lr=1e-3, bs=64, wd=1e-4, noise=0.02,
        pat=20, model_kw={}, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SOHNet(in_ch=Xtr.shape[1], **model_kw)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    best, bs_state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            if len(idx) < 2: continue
            xb = Xt[idx] + noise*torch.randn_like(Xt[idx])
            loss = nn.functional.mse_loss(m(xb), yt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        v = float(((predict(m, Xva)-yva)**2).mean()) if Xva is not None else loss.item()
        sch.step(v)
        if v < best - 1e-7: best, bs_state, bad = v, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= pat: break
        if verbose and ep % 10 == 0: print(ep, round(v, 6))
    m.load_state_dict(bs_state); return m

def lopo(X, y, pack, seed=0, groups=None, **kw):
    P = np.zeros_like(y); rng = np.random.RandomState(seed)
    ups = np.unique(pack)
    for pk in ups:
        te = pack == pk
        others = np.array([p for p in ups if p != pk])
        vp = rng.choice(others, 2, replace=False)
        va = np.isin(pack, vp); tr = ~te & ~va
        mm = scale_fit(X[tr])
        m = fit(scale(X[tr], mm), y[tr], scale(X[va], mm), y[va], seed=seed, **kw)
        P[te] = predict(m, scale(X[te], mm))
    return P
