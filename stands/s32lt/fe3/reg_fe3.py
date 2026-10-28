"""21.10.2026 f0: чи несе res_e/learned інформацію про похибку останнього шару понад 51 стару колонку LT2.
Лінійний ridge на sign-log ознаках шару 15, ціль e = truth15 - pred15; фіт на першій половині мереж, тест на другій.
ПЕРЕДБАЧЕННЯ: приріст від res_e/learned ≲1% MSE (c2: масштаб nn_asset плоский), шанс >3% 15%.
Контроль: «чужа» res_e (зсув мереж на 1)."""
import glob, os, sys, numpy as np
fl = sorted(f for f in glob.glob("/mnt/gguf/wb_lt2alg/dump2/corpus_*.npz") if not f.endswith(".tmp.npz"))
loc = {}
for fn in sorted(os.listdir("/home/trim/wb2data/corpus14k")):
    if fn.startswith("shard") and fn.endswith(".npz"):
        with np.load("/home/trim/wb2data/corpus14k/" + fn) as z:
            for r, i in enumerate(np.asarray(z["idx"])): loc[int(i)] = (fn, r)
def sl(x): return np.sign(x) * np.log1p(np.abs(x) * 1e3)
XS, RE, E = [], [], []
cache = {}
for f in fl:
    d = np.load(f); k = int(d["idx"]); fn, r = loc[k]
    if fn not in cache:
        cache.clear()
        with np.load("/home/trim/wb2data/corpus14k/" + fn) as z: cache[fn] = np.asarray(z["moments"][:, 0, 15], dtype=np.float64)
    t = cache[fn][r]
    X = d["X"][15].astype(np.float64)
    XS.append(sl(X[:, :51])); RE.append(X[:, 51:53]); E.append(t - d["pred"][15].astype(np.float64))
M = len(E); h = M // 2
print("мереж", M, "raw сер", np.mean([np.mean(e ** 2) for e in E]))
def fit(cols, tr, lam=1e-3):
    A = np.concatenate([cols(i) for i in tr]); y = np.concatenate([E[i] for i in tr])
    mu, sd = A.mean(0), A.std(0) + 1e-12; A = (A - mu) / sd
    A = np.hstack([A, np.ones((len(A), 1))])
    w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ y)
    return lambda B: np.hstack([(B - mu) / sd, np.ones((len(B), 1))]) @ w
def ev(cols):
    out = []
    for tr, te in ((range(h), range(h, M)), (range(h, M), range(h))):
        p = fit(cols, tr)
        out += [np.mean((E[i] - p(cols(i))) ** 2) for i in te]
    return np.mean(out)
base = np.mean([np.mean(e ** 2) for e in E])
rs = lambda i: RE[i] * 1e5
for nm, c in [("51 старих", lambda i: XS[i]),
              ("51 + res_e,learned", lambda i: np.hstack([XS[i], rs(i), sl(RE[i])])),
              ("лише res_e,learned", lambda i: np.hstack([rs(i), sl(RE[i])])),
              ("🔒 51 + ЧУЖІ res_e", lambda i: np.hstack([XS[i], rs((i + 1) % M), sl(RE[(i + 1) % M])]))]:
    v = ev(c); print("%-22s MSE %.4e  ×%.4f" % (nm, v, base / v), flush=True)
