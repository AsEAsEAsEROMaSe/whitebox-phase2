"""ДЕТЕРМІНОВАНА γ₃ ІЗ ЗАМИКАННЯ — БЕЗ СЕМПЛІВ, ЗА ЗАКРИТОЮ ФОРМУЛОЮ.

ПОСТАНОВКА ЮЗЕРА: «через мат суть виводити дешевші».
`p_j = Σ_i w_ij·h_i`, `h_i = ReLU(p_i^{поперед})`. Третій кумулянт СУМИ
незалежних доданків додається ⟹ ДІАГОНАЛЬНЕ наближення:
      κ₃(p_j) ≈ Σ_i w_ij³ · κ₃(h_i),      γ₃_j = κ₃(p_j)/s_j³
а `κ₃(h_i)` для `h = ReLU(N(μ,σ²))` має ЗАКРИТУ ФОРМУ через `t = μ/σ`:
      M₁ = σ(tΦ+φ)
      M₂ = σ²[(t²+1)Φ + tφ]
      M₃ = σ³[(t³+3t)Φ + (t²+2)φ]
      κ₃ = M₃ − 3M₂M₁ + 2M₁³
Усе праворуч замикання ВЖЕ МАЄ. Ціна: куб W поелементно + матвектор
= ~4e6 флопів на шар ⟹ **0.0013% бюджету** на всі сім шарів.

🚨 ЗАСТЕРЕЖЕННЯ ДО ПРОГОНУ: `[indep101, С102]` 96% негаусовості γ₄ породжує
ЗАЛЕЖНІСТЬ між нейронами, а не власна негаусовість `h_i` ⟹ діагональ може
дати нуль. Але там міряли γ₄ і через participation ratio, не γ₃ понейронно.

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 X1 (40%) corr(γ₃_діаг, γ₃_банк) на ш15 > 0.30
 X2 (25%) підстановка γ₃_діаг у рекурсію дає ≥×1.03
 X3 (70%) corr падає з глибиною (залежність накопичується)
🔒 ПЛОМБА: на шарі 1 вхід `h₀ = ReLU(W₀z)` має НЕЗАЛЕЖНІ компоненти лише
   умовно; але `p₁` — сума 1024 доданків із ТОЧНО відомими κ₃ ⟹ там діагональ
   мусить бути найточнішою. Якщо і там corr низька — формула хибна, не гіпотеза.
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

def k3_relu(mp, s):
    """третій ЦЕНТРАЛЬНИЙ момент ReLU(N(mp,s²)) — закрита форма"""
    t = mp/s; ph = phi(t); Ph = Phi(t)
    M1 = s*(t*Ph + ph)
    M2 = s*s*((t*t + 1.0)*Ph + t*ph)
    M3 = s**3*((t**3 + 3.0*t)*Ph + (t*t + 2.0)*ph)
    return M3 - 3.0*M2*M1 + 2.0*M1**3

def closure(W, g3src=None, c3=0.0):
    """g3src: None | 'diag' | масив (D,NW). Повертає m, і список γ₃_діаг по шарах."""
    m = np.zeros(NW); C = np.eye(NW); GD = [None]*D
    k3h = None
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        if k3h is not None:
            GD[i] = (k3h @ (w*w*w)) / (s**3)
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if c3 != 0.0 and i in LAY:
            g = GD[i] if g3src == "diag" else (g3src[i] if g3src is not None else None)
            if g is not None: m = m - c3*s*(g/6.0)*(t*ph)
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        k3h = k3_relu(mp, s)
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m, GD

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, T=d["relu_15"],
                  g3=np.stack([d["g3_%d" % L] for L in range(D)])))
lg("банків %d" % len(B))

# ── 1. кореляція діагональної γ₃ з банковою, пошарово ──
CR = np.zeros((len(B), D)); RT = np.zeros((len(B), D))
for k, b in enumerate(B):
    _, GD = closure(b["W"])
    b["gd"] = GD
    for L in range(1, D):
        if GD[L] is None: continue
        CR[k, L] = np.corrcoef(GD[L], b["g3"][L])[0, 1]
        RT[k, L] = GD[L].std()/max(b["g3"][L].std(), 1e-30)
print("\n── 1. ДІАГОНАЛЬНА γ₃ ПРОТИ БАНКОВОЇ ──")
print("%4s %12s %10s %12s" % ("шар", "corr", "sd розкид", "sd(діаг)/sd(банк)"))
for L in range(1, D):
    print("%4d %11.4f %10.4f %12.4f" % (L, CR[:,L].mean(), CR[:,L].std(), RT[:,L].mean()))
lg("кореляції готові")

# ── 2. підстановка в рекурсію, скан c ──
BASE = float(np.mean([cerr(closure(b["W"])[0], b["T"]) for b in B]))
ADJ0 = (1.0/(1.0/V_MC + 1.0/ANCH_E2))*MULT
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
print("\n── 2. γ₃_ДІАГ У РЕКУРСІЇ (шари 9..15) ──")
print("%8s %12s %10s" % ("c", "косинусна", "×база"))
best = (0.0, ADJ0/sc(BASE))
for c in (0.05, 0.10, 0.20, 0.40, 0.80, -0.20):
    v = float(np.mean([cerr(closure(b["W"], "diag", c)[0], b["T"]) for b in B]))
    r = ADJ0/sc(v); print("%8.2f %12.5e %10.4f" % (c, v, r))
    if r > best[1]: best = (c, r)
print("⟹ найкраще c=%.2f ⟹ ×%.4f" % best)
vb = float(np.mean([cerr(closure(b["W"], np.stack([b["g3"][L] for L in range(D)]), 0.40)[0], b["T"]) for b in B]))
print("   для порівняння: БАНКОВА γ₃ c=0.40 ⟹ ×%.4f" % (ADJ0/sc(vb)))
lg("ГОТОВО")
