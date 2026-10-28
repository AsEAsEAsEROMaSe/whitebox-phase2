"""wfeat256 — ПЕРЕНЕСЕННЯ ОЗНАКИ НА РОБОЧУ ТОЧКУ. Той самий код ознак, що в
`wfeat99`, але n=256, справжні мережі банку, істина `final_means` N=1e9.

🚨 ЦЕ І Є ВИРІШАЛЬНИЙ ВИМІР. «Відкрито при n=32» саме по собі нічого не варте:
розкид `e` між мережами при n=32 у ×4.8, при n=256 він набагато вужчий ⟹ мода,
що пояснює дисперсію там, може бути придушена тут. Умова 4 контролера
(«ознаки, осмислені при будь-якому n») перевіряється ТІЛЬКИ так.

🔒 ПЛОМБА: mlp_id=0 мусить дати 0.4647% (pairrec97 / anchor97) — інакше стенд
міряє інший об'єкт.
🔒 ПЛОМБА «ЧУЖА W»: 400 перемішувань, як у wfeat99.
🔒 Шуму цілі тут НЕМА взагалі (N=1e9 проти нашого робочого 1e4) ⟹ стеля R² = 1.

ПЕРЕДБАЧЕННЯ (до прогону): якщо при n=32 F1 дасть LOO-R² > 0.6, то при n=256
він дасть > 0.3 — 45%. Механізм може бути суто «малоширинним» (вимирання
нейронів), і тоді тут буде нуль.
"""
import glob
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32
NWID = 256
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def psi(x):
    x = np.asarray(x, float)
    return phi(x) + x * ndtr(x)


def gl(a, b, npan, nnod):
    xs, ws = np.polynomial.legendre.leggauss(nnod)
    ed = np.linspace(a, b, npan + 1)
    lo, hi = ed[:-1, None], ed[1:, None]
    return ((0.5 * (hi - lo) * xs[None, :] + 0.5 * (hi + lo)).ravel(),
            (0.5 * (hi - lo) * ws[None, :]).ravel())


ZQ, WQ = gl(0.0, 40.0, 12, 20)          # як в anchor97 (пломба порівнянності)


def closure_full(W, n, chunk=8192):
    m = np.zeros(n)
    C = np.eye(n)
    gates, rms, tt, prC = [], [], [], []
    for L in range(DEPTH):
        mp = W[L].T @ m
        Cp = W[L].T @ C @ W[L]
        s2 = np.maximum(np.diag(Cp), 1e-300)
        s = np.sqrt(s2)
        t = mp / s
        rho = np.clip(Cp / (s[:, None] * s[None, :]), -0.999999, 0.999999)
        tj = np.repeat(t, n)
        tk = np.tile(t, n)
        rr = rho.ravel()
        out = np.empty(n * n)
        q = np.sqrt(np.maximum(1.0 - rr * rr, 1e-14))
        for o in range(0, n * n, chunk):
            sl = slice(o, min(o + chunk, n * n))
            x = ZQ[None, :] - tj[sl, None]
            u = (rr[sl, None] * x + tk[sl, None]) / q[sl, None]
            out[sl] = (WQ[None, :] * ZQ[None, :] * phi(x) * q[sl, None]
                       * psi(u)).sum(1)
        M2 = out.reshape(n, n) * (s[:, None] * s[None, :])
        np.fill_diagonal(M2, s2 * ((1 + t * t) * ndtr(t) + t * phi(t)))
        ma = s * psi(t)
        Cn = M2 - np.outer(ma, ma)
        C = 0.5 * (Cn + Cn.T)
        m = ma
        gates.append(ndtr(t))
        tt.append(t.copy())
        rms.append(math.sqrt(float((m * m).mean())))
        ev = np.maximum(np.diag(C), 0.0)
        prC.append(float(ev.sum() ** 2 / max((C * C).sum(), 1e-300)))
    return m, gates, np.array(rms), tt, np.array(prC)


# ═══ ОЗНАКИ — КОД ІДЕНТИЧНИЙ wfeat99 (інакше порівняння не апельсини) ═══
FNAMES = [
    "F1_transport", "F2_transp_max", "F3_transp_spec", "F4_gate_mean",
    "F5_alive_frac", "F6_neff_gate", "F7_absT_med", "F8_smallT_frac",
    "F9_prC_last", "F10_spec_kappa", "F11_spec_pr", "F12_logdet",
    "F13_align", "F14_gate_drift",
]


def features(W, n, gates, rms, tt, prC):
    f = {}
    P = np.eye(n)
    terms, sterms = [], []
    for L in range(DEPTH - 1, -1, -1):
        fro = float((P * P).sum())
        spc = float(np.linalg.norm(P, 2))
        terms.append(rms[L] ** 2 * fro / n)
        sterms.append(rms[L] ** 2 * spc ** 2 / n)
        P = (W[L] * gates[L][None, :]) @ P
    den = max(rms[DEPTH - 1] ** 2, 1e-300)
    terms = np.array(terms) / den
    sterms = np.array(sterms) / den
    f["F1_transport"] = 0.5 * math.log(max(terms.sum(), 1e-300))
    f["F2_transp_max"] = 0.5 * math.log(max(terms.max(), 1e-300))
    f["F3_transp_spec"] = 0.5 * math.log(max(sterms.sum(), 1e-300))
    G = np.array(gates)
    f["F4_gate_mean"] = float(G.mean())
    deep = G[DEPTH // 2:]
    f["F5_alive_frac"] = float((deep > 0.01).mean())
    f["F6_neff_gate"] = float(np.mean(
        G.sum(1) ** 2 / np.maximum((G * G).sum(1), 1e-300)) / n)
    T = np.array(tt)
    f["F7_absT_med"] = float(np.median(np.abs(T)))
    f["F8_smallT_frac"] = float((np.abs(T) < 1.0).mean())
    f["F9_prC_last"] = float(prC[-1] / n)
    f["F14_gate_drift"] = float(np.std(G.mean(1)))
    ka, pr, ld, al = [], [], [], []
    v_prev = None
    for L in range(DEPTH):
        u, sv, vt = np.linalg.svd(W[L])
        lam = sv ** 2
        ka.append(math.log(lam.max() / lam.mean()))
        pr.append(lam.sum() ** 2 / (lam * lam).sum() / n)
        ld.append(float(np.log(lam).mean()))
        if v_prev is not None:
            al.append(abs(float(v_prev @ u[:, 0])))
        v_prev = vt[0]
    f["F10_spec_kappa"] = float(np.mean(ka))
    f["F11_spec_pr"] = float(np.mean(pr))
    f["F12_logdet"] = float(np.mean(ld))
    f["F13_align"] = float(np.mean(al))
    return np.array([f[k] for k in FNAMES])


def loo_ridge(X, y, lam):
    m = X.shape[0]
    pred = np.empty(m)
    for i in range(m):
        tr = np.ones(m, bool)
        tr[i] = False
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(0), Xt.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Z = (Xt - mu) / sd
        ym = yt.mean()
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (yt - ym))
        pred[i] = ym + ((X[i] - mu) / sd) @ w
    ss = float(((y - pred) ** 2).sum())
    return 1.0 - ss / float(((y - y.mean()) ** 2).sum()), pred


print(f"wfeat256 · робоча точка n=256 · {NNET} мереж · істина N=1e9", flush=True)
t0 = time.time()
Xs, ys, ids_used = [], [], []
for fpath in sorted(glob.glob(SRC + "/data-*.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(fpath, "rb")).read_all()
    x = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, 256, 256)
    yv = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(yv.type):
        yv = yv.flatten()
    FM = np.asarray(yv).reshape(tb.num_rows, 256)
    ids = tb.column("mlp_id").to_pylist()
    for i in range(tb.num_rows):
        if len(ys) >= NNET:
            break
        W = [np.ascontiguousarray(Wall[i][L], dtype=np.float64)
             for L in range(DEPTH)]
        cm, gates, rms, tt, prC = closure_full(W, NWID)
        tm = FM[i].astype(np.float64)
        sc = math.sqrt(float((tm * tm).mean()))
        e = float(np.sqrt(((cm - tm) ** 2).mean()) / sc)
        Xs.append(features(W, NWID, gates, rms, tt, prC))
        ys.append(math.log(e))
        ids_used.append(ids[i])
        tag = ""
        if ids[i] == 0:
            tag = (f"  🔒 ПЛОМБА mlp_id=0: {100*e:.4f}% проти 0.4647% ⟹ Δ "
                   f"{100*abs(e-0.004647)/0.004647:.2f}%")
        print(f"  мережа {len(ys):>2d} (id {ids[i]:>3d}): e={100*e:.4f}%"
              f"  ({time.time()-t0:.0f} с){tag}", flush=True)
    if len(ys) >= NNET:
        break

X = np.array(Xs)
y = np.array(ys)
M = X.shape[0]
print(f"\n  {M} мереж · {time.time()-t0:.0f} с")
print(f"  медіана e = {100*math.exp(np.median(y)):.4f}%  ·  sd(log e) = "
      f"{y.std():.4f}  ⟹ розкид ±1σ ×{math.exp(2*y.std()):.2f}"
      f"   (при n=32 було ×4.78)")

print("\n" + "=" * 86)
print("ОДИНИЧНІ ОЗНАКИ ПРИ n=256")
print("=" * 86)
for j, nm in enumerate(FNAMES):
    c = np.corrcoef(X[:, j], y)[0, 1]
    c = 0.0 if not np.isfinite(c) else c
    print(f"  {nm:<18s} r = {c:+.4f}   r² = {c*c:.4f}")

r2f1, _ = loo_ridge(X[:, :1], y, 1e-6)
r2r, _ = loo_ridge(X, y, 10.0)
print("\n" + "=" * 86)
print("ЧЕСНИЙ LOO ПО МЕРЕЖАХ")
print("=" * 86)
print(f"  ЛИШЕ F1 (без підбору):  LOO-R² = {r2f1:+.4f}")
print(f"  гребінь усіх 14, λ=10:  LOO-R² = {r2r:+.4f}")

rng = np.random.default_rng(256)
n1, nr = [], []
for _ in range(400):
    p = rng.permutation(M)
    n1.append(loo_ridge(X[:, :1], y[p], 1e-6)[0])
    nr.append(loo_ridge(X, y[p], 10.0)[0])
n1 = np.array(n1)
nr = np.array(nr)
print("\n🔒 КОНТРОЛЬ «ЧУЖА W» (400 перемішувань)")
print(f"  F1:      нуль медіана {np.median(n1):+.4f}  95-й {np.percentile(n1,95):+.4f}"
      f"  ⟹ p = {(n1 >= r2f1).mean():.4f}")
print(f"  гребінь: нуль медіана {np.median(nr):+.4f}  95-й {np.percentile(nr,95):+.4f}"
      f"  ⟹ p = {(nr >= r2r).mean():.4f}")

np.savez("/mnt/win_d/llm/White-Box/arena62/logs/wfeat256.npz",
         X=X, y=y, ids=np.array(ids_used), n1=n1, nr=nr,
         names=np.array(FNAMES))
print(f"\n  час {time.time()-t0:.0f} с")
