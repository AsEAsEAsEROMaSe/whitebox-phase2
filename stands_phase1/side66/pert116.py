"""ВТРУЧАННЯ В МЕРЕЖУ: ЩО САМЕ РОБИТЬ НАШУ ПОХИБКУ ВЕЛИКОЮ (ідея ЮЗЕРА).

Усі попередні спроби «що впливає» були КОРЕЛЯЦІЙНІ — на 24 даних мережах.
Сьогодні це знову обпеклось: corr(log e², log m²)=0.79 при |сер|/sd=46
виявилось чистим МАСШТАБОМ, який метрика вже знімає.
Втручання цієї пастки не має: міняємо властивість — дивимось, чи рухається
похибка. Це причина, а не супутність.

НА ЩО ВКАЗУЮТЬ ЧИСЛА (три незалежні виміри):
  97.8% понейронної варіації γ₃ — залежність між нейронами  [g3an116]
  96%   те саме для γ₄                                       [С102 indep101]
  ряд Мелера мізерний саме тому, що ρ мале                    [meh116]
⟹ ручка: ортогоналізація стовпців W (ρ→0) і, навпаки, ранг-1 спільна мода.

    W_λ = (1−λ)·W + λ·W⊥     (ті самі норми стовпців)
    W_μ = W + μ·u vᵀ         (спільна мода ⟹ ρ РОСТЕ)
🔒 НУЛЬ-КОНТРОЛЬ: W·(1+δ) — косинусна метрика мусить дати РІВНО нуль.

🔮 R1 (65%) ортогоналізація ріже похибку ≥×1.5
🔮 R2 (50%) ранг-1 монотонно псує
🔮 R3 (85%) глобальний масштаб дає нуль (інакше стенд зламаний)
"""
import sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
NTRUTH = 1 << 20
BATCH = 1 << 16
NNET = 4
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def closure(W):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: return m
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))

def truth(W, seed):
    Wg = [cp.asarray(W[k].astype(cp.float32)) for k in range(D)]
    acc = cp.zeros(NW, cp.float64); done = 0; b = 0
    while done < NTRUTH:
        n = min(BATCH, NTRUTH - done)
        rs = cp.random.RandomState(seed=seed + b)
        x = rs.standard_normal((n, NW), dtype=cp.float32)
        for k in range(D - 1):
            x = cp.maximum(x @ Wg[k], 0.0)
        acc += cp.maximum(x @ Wg[D-1], 0.0).astype(cp.float64).sum(0)
        del x; cp.get_default_memory_pool().free_all_blocks()
        done += n; b += 1
    r = cp.asnumpy(acc)/NTRUTH
    del Wg; cp.get_default_memory_pool().free_all_blocks()
    return r

def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T
    return 100*np.sqrt(float(r @ r)/float(T @ T))     # відносна похибка, %

def orth(w):
    """ортогоналізація СТОВПЦІВ зі збереженням їхніх норм"""
    q, _ = np.linalg.qr(w)
    return q * np.linalg.norm(w, axis=0)

def rho_stat(W):
    """медіана |ρ| між передактиваціями шару 15 за замиканням"""
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        if i == D-1:
            R = cp_/np.outer(s, s); iu = np.triu_indices(NW, 1)
            return float(np.median(np.abs(R[iu])))
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
WS = []
for k in range(NNET):
    WS.append((NAMES[k], np.asarray(tb["weights"][k].as_py(), np.float64).reshape(D, NW, NW)))
lg("мереж %d · N істини %d" % (NNET, NTRUTH))

ROWS = []
for tag, mk in (
    ("🔒 масштаб ×1.00", lambda W, p: W),
    ("🔒 масштаб ×1.30", lambda W, p: W*1.30),
    ("ортог λ=0.25", lambda W, p: np.stack([(1-0.25)*W[i] + 0.25*orth(W[i]) for i in range(D)])),
    ("ортог λ=0.50", lambda W, p: np.stack([(1-0.50)*W[i] + 0.50*orth(W[i]) for i in range(D)])),
    ("ортог λ=0.75", lambda W, p: np.stack([(1-0.75)*W[i] + 0.75*orth(W[i]) for i in range(D)])),
    ("ортог λ=1.00", lambda W, p: np.stack([orth(W[i]) for i in range(D)])),
    ("ранг-1 μ=0.05", lambda W, p: np.stack([W[i] + 0.05*np.outer(p[i][0], p[i][1]) for i in range(D)])),
    ("ранг-1 μ=0.15", lambda W, p: np.stack([W[i] + 0.15*np.outer(p[i][0], p[i][1]) for i in range(D)])),
    ("ранг-1 μ=0.30", lambda W, p: np.stack([W[i] + 0.30*np.outer(p[i][0], p[i][1]) for i in range(D)])),
):
    errs, rhos = [], []
    for k, (nm, W) in enumerate(WS):
        rs = np.random.RandomState(900 + k)
        sc = np.sqrt(2.0/NW)
        P = [(rs.standard_normal(NW)*sc, rs.standard_normal(NW)*sc) for _ in range(D)]
        Wp = mk(W, P)
        T = truth(Wp, 60000 + k*137)
        errs.append(cerr(closure(Wp), T)); rhos.append(rho_stat(Wp))
    ROWS.append((tag, float(np.mean(errs)), float(np.mean(rhos))))
    lg("%-18s похибка %.4f%% · мед|ρ| %.4f" % (tag, ROWS[-1][1], ROWS[-1][2]))

b = ROWS[0][1]
print("\n%-18s %12s %10s %10s" % ("конфігурація", "похибка %", "×база", "мед |ρ|"))
for tag, e, r in ROWS:
    print("%-18s %11.4f%% %10.4f %10.4f" % (tag, e, b/e, r))
lg("ГОТОВО")
