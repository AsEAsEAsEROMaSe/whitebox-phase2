"""С117 — ЧИ РОЗШИРЮВАТИ γ₃ З 9..15 ДО 5..15 (рішення про сабміт).

[full117, ТОЧНИЙ δ] 9..15 ×1.2298 · **5..15 ×1.2615** · 1..15 ×1.2375
⟹ оптимальний діапазон 5..15, а не 9..15 (у продакшені).
[SNR] шум ĝ₃ = √(6/N) = 0.0312 при N=6144; розкид по нейронах:
      ш4 0.0297 (SNR 0.95) · ш5–8 ~1.2–1.87 · ш15 3.33
⟹ на шарах 5..8 γ₃ ще вимірна, але потребує СИЛЬНІШОЇ усадки.

Тут — реалістично: γ₃ з N семплів + ПОШАРОВА усадка Джеймса-Стайна
   S_L = σ²_L/(σ²_L + 6/N),  σ_L оцінюється з тих самих семплів.
Порівнюються діапазони при спільному скані `c`.

🔮 R1 (60%) 5..15 б'є 9..15 і при N=6144, але слабше за оракульні ×1.0258
🔮 R2 (45%) пошарова усадка б'є сталу _G3_S=0.75
🔮 R3 (30%) 3..15 ще краще за 5..15 при правильній усадці
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)

def closure(W, g3=None, c=0.0, LAY=()):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if g3 is not None and i in LAY and c != 0.0:
            m = m - c*(s*((g3[i]/6.0)*(t*ph)))
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, T=d["relu_15"],
                  g3=np.stack([d["g3_%d" % L] for L in range(D)])))
lg("мереж %d" % len(B))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
lg("база %.6e · adj %.4e" % (BASE, sc(BASE)))

def noisy(b, N, rep, shrink):
    """γ₃ з N семплів + усадка (None ⟹ пошарова Джеймс-Стайн)"""
    if N is None: return b["g3"]
    sig = np.sqrt(6.0/N); out = np.zeros_like(b["g3"])
    rs = np.random.RandomState(9117 + rep*131 + (hash(b["name"]) % 997))
    for L in range(D):
        g = b["g3"][L] + sig*rs.standard_normal(NW)
        mu = g.mean()
        S = shrink if shrink is not None else \
            max(g.var() - sig*sig, 0.0)/max(g.var(), 1e-30)
        out[L] = mu + S*(g - mu)
    return out

RANGES = [("9..15", tuple(range(9, 16))), ("7..15", tuple(range(7, 16))),
          ("5..15", tuple(range(5, 16))), ("3..15", tuple(range(3, 16)))]
CS = [0.20, 0.30, 0.40, 0.50, 0.65]
REPS = 2
for tag, N, shr in (("ОРАКУЛ", None, None), ("N=6144 S=0.75", 6144, 0.75),
                    ("N=6144 ДС", 6144, None)):
    print("\n── %s ── (×база у MSE)" % tag)
    print("%8s | %s" % ("шари", " ".join("c=%.2f" % c for c in CS)))
    for nm, LAY in RANGES:
        row = []
        for c in CS:
            acc = []
            for rep in range(1 if N is None else REPS):
                gs = [noisy(b, N, rep, shr) for b in B]
                acc.append(np.mean([cerr(closure(b["W"], g, c, LAY), b["T"])
                                    for b, g in zip(B, gs)]))
            row.append(BASE/float(np.mean(acc)))
        print("%8s | %s   найкраще %6.4f" % (nm, " ".join("%6.4f" % x for x in row), max(row)))
        sys.stdout.flush()
lg("ГОТОВО")
print("DONE_RNG117")
