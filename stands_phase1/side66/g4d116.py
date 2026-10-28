"""γ₄ У РЕКУРСІЇ НА ГЛИБОКИХ ШАРАХ — ПОВЕРХ ЗНАЙДЕНОЇ γ₃.

[g3sh116] γ₃@9..15 c=0.40 ⟹ ×1.1294. Питання: чи додає ПОНЕЙРОННА γ₄?
Корпус (С116, ceil116b/edge116) казав «вся інформація в γ₃, γ₄ можна СТАЛОЮ»
— але це міряно НА ОСТАННЬОМУ ШАРІ, з ТОЧНИМИ моментами, поза рекурсією.
У рекурсії на наших моментах γ₄ не пробувалась жодного разу.
Еджворт: E[relu]/s = ψ(t) − c₃(γ₃/6)·tφ + c₄(γ₄/24)·(t²−1)φ

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 U1 (35%) γ₄ додає ≥×1.02 понад саму γ₃
 U2 (70%) оптимум c₄ СТИСНЕНИЙ так само (|c₄*| < 1)
 U3 (55%) знак c₄* додатний (як у формулі), а не перевернутий
🔒 КОНТРОЛІ: чужа γ₄ · перемішана γ₄ · спільний скан (c₃,c₄)
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

def closure(W, g3=None, g4=None, c3=0.0, c4=0.0):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if i in LAY:
            if g3 is not None and c3 != 0.0: m = m - c3*s*(g3[i]/6.0)*(t*ph)
            if g4 is not None and c4 != 0.0: m = m + c4*s*(g4[i]/24.0)*((t*t-1.0)*ph)
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
                  g3=np.stack([d["g3_%d" % L] for L in range(D)]),
                  g4=np.stack([d["g4_%d" % L] for L in range(D)])))
lg("банків %d" % len(B))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
ADJ0 = (1.0/(1.0/V_MC + 1.0/ANCH_E2))*MULT
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
def ev(c3, c4, mode="own"):
    a = []
    for k, b in enumerate(B):
        if mode == "own": g3, g4 = b["g3"], b["g4"]
        elif mode == "foreign": g3, g4 = b["g3"], B[(k+1) % len(B)]["g4"]
        else:
            rs = np.random.RandomState(9090+k)
            g3, g4 = b["g3"], np.stack([rs.permutation(b["g4"][L]) for L in range(D)])
        a.append(cerr(closure(b["W"], g3, g4, c3, c4), b["T"]))
    return float(np.mean(a))
lg("база %.6e" % BASE)

C3 = [0.0, 0.30, 0.40, 0.50]
C4 = [-0.75, -0.40, -0.20, 0.0, 0.20, 0.40, 0.75, 1.00]
print("\n     c3 \\ c4 | %s" % " ".join("%7.2f" % c for c in C4))
best = (0.0, 0.0, ADJ0/sc(BASE))
for c3 in C3:
    row = []
    for c4 in C4:
        v = ev(c3, c4); r = ADJ0/sc(v); row.append(r)
        if r > best[2]: best = (c3, c4, r)
    print("     %5.2f    | %s" % (c3, " ".join("%7.4f" % x for x in row)))
    lg("  c3=%.2f готово" % c3)
print("\n🎯 найкраще: c3=%.2f c4=%.2f ⟹ ×%.4f  (γ₃ сама c3=0.40,c4=0: ×%.4f)"
      % (best[0], best[1], best[2], ADJ0/sc(ev(0.40, 0.0))))
if abs(best[1]) > 1e-9:
    print("\n🔒 КОНТРОЛІ на γ₄ (c3=%.2f c4=%.2f)" % (best[0], best[1]))
    for md in ("foreign", "shuffle"):
        v = ev(best[0], best[1], md); print("   🔒 %-8s ⟹ ×%.4f" % (md, ADJ0/sc(v)))
lg("ГОТОВО")
