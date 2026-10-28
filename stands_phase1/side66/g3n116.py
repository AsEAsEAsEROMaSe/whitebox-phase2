"""ЧИ ЗДОБУВАНА ПОНЕЙРОННА γ₃ — СКАН (N, c) + УСАДКА.

МОТИВ: [g3sh116] істинна γ₃@9..15 при c*=0.50 дає ×1.1297, контролі ламаються
(чужа/переміш ×0.6997). Але на інференсі γ₃ треба ЗВІДКИСЬ узяти.
🔑 МИ ВЖЕ ПРОГАНЯЄМО 6144 СЕМПЛИ для МК-гілки ⟹ треті центральні моменти на
кожному шарі коштують ~3e8 флопів = 0.014% бюджету, тобто ДАРОМ.
[prop116] міряв вибіркову γ₃ при N=6144 і дістав ×0.8095 — АЛЕ ПРИ c=1.0,
де навіть ІСТИННА γ₃ дає ×0.9059. Тобто той вимір нічого не вирішив.

Модель шуму: sd(γ̂₃) = √(6/N) (гаусова асимптотика), незалежно по нейронах.
УСАДКА: γ̃ = γ̄_шар + S·(γ̂ − γ̄_шар), S = σ²між/(σ²між+σ²шум) — Джеймс-Стайн,
бо шум незалежний по нейронах, а сигнал має скінченну дисперсію.

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 T1 (60%) при N=6144 з фітованим c лишається ≥×1.03 (тобто prop116 хибив знаком)
 T2 (35%) лишається ≥×1.06
 T3 (80%) усадка додає — оптимальний S помітно менший за 1
 T4 (50%) оптимальний c РОСТЕ зі зменшенням N (більше шуму ⟹ сильніша усадка
          вже вбудована в c) АБО навпаки падає; знак не очевидний
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
LAY = tuple(range(9, 16))
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)

def closure(W, g3=None, c=0.0):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if g3 is not None and i in LAY and c != 0.0:
            m = m - c*s*(g3[i]/6.0)*t*ph
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
    B.append(dict(name=nm, W=W, g3=np.stack([d["g3_%d" % L] for L in range(D)]), T=d["relu_15"]))
lg("банків %d · шари %s" % (len(B), str(LAY)))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
ADJ0 = (1.0/(1.0/V_MC + 1.0/ANCH_E2))*MULT
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
lg("база %.6e" % BASE)

def noisy(b, N, rep, shrink):
    """вибіркова γ₃ з N семплів + усадка до пошарового середнього"""
    if N is None: return b["g3"]
    sig = np.sqrt(6.0/N); out = np.zeros_like(b["g3"])
    rs = np.random.RandomState(5150 + rep*97 + hash(b["name"]) % 1000)
    for L in range(D):
        g = b["g3"][L] + sig*rs.standard_normal(NW)
        mu = g.mean()
        if shrink is None:                       # Джеймс-Стайн з оцінених величин
            vb = max(g.var() - sig*sig, 0.0); S = vb/(vb + sig*sig)
        else: S = shrink
        out[L] = mu + S*(g - mu)
    return out

NS = [None, 98304, 24576, 12288, 6144, 3072, 1536]
CS = [0.25, 0.40, 0.50, 0.60, 0.75, 1.00]
REPS = 2
print("\n%10s | %s" % ("N", " ".join("c=%.2f" % c for c in CS)))
BEST = {}
for N in NS:
    row = []
    for c in CS:
        acc = []
        for rep in range(REPS if N else 1):
            for b in B: acc.append(cerr(closure(b["W"], noisy(b, N, rep, None), c), b["T"]))
        v = float(np.mean(acc)); row.append(ADJ0/sc(v))
    tagn = "ІСТИННА" if N is None else str(N)
    print("%10s | %s   ⟹ найкраще ×%.4f при c=%.2f"
          % (tagn, " ".join("%6.4f" % x for x in row), max(row), CS[int(np.argmax(row))]))
    BEST[tagn] = (max(row), CS[int(np.argmax(row))])
    lg("  %s готово" % tagn)

print("\n── СКАН УСАДКИ при N=6144, c=найкраще ──")
c6 = BEST["6144"][1]
for S in (0.0, 0.25, 0.5, 0.75, 1.0, None):
    acc = []
    for rep in range(REPS):
        for b in B: acc.append(cerr(closure(b["W"], noisy(b, 6144, rep, S), c6), b["T"]))
    v = float(np.mean(acc))
    print("  S = %-8s ⟹ ×%.4f" % ("Джеймс-Стайн" if S is None else "%.2f" % S, ADJ0/sc(v)))
lg("ГОТОВО")
