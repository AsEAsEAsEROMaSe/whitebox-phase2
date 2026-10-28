"""УЗГОДЖЕНА γ₃: ПОПРАВКА НЕ ЛИШЕ ДО СЕРЕДНЬОГО, А Й ДО ДИСПЕРСІЇ.

ВИВЕДЕННЯ (з тих самих коефіцієнтів Ерміта, що й у meh116):
Еджворт `E[g] = E_φ[g] + (γ₃/6)·E_φ[g·He₃]`. Для `g = s²(t+X)₊²`
інтегрування частинами дає `E_φ[g·He₃] = 2s²φ(t)` ⟹
      E[ReLU²] += (γ₃/3)·s²·φ(t)
Для середнього той самий шлях дає `−(γ₃/6)·s·t·φ` — це вже стоїть у нас.
⟹ зараз поправка НЕУЗГОДЖЕНА: середнє виправлене, дисперсія — ні.

🚨 ЧОМУ ЦЕ НЕ ПОВТОР `edgeksig` (×0.875): там та сама формула йшла в рекурсію
на шарах **1..ksig**, тобто МІЛКИХ. Рівно та конфігураційна пастка, що
ховала γ₃ у середньому (prop116 міряв c=1.0 і шари без 9,10,11).
🔑 І це проходить закон lad116: діє на 9..15 БЕЗ РОЗРИВУ до виходу.

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 W1 (45%) поправка до дисперсії ПОНАД працюючу γ₃ дає ≥×1.02
 W2 (25%) ≥×1.05
 W3 (60%) оптимум СТИСНЕНИЙ (|c|<1) — восьме поспіль
 W4 (35%) знак оптимуму ВІД'ЄМНИЙ (як було з κ і табличною γ₄)
⚠️ СТЕЛЯ: канал `C` з ПОВНИМ оракулом = +23% [С115] ⟹ це лише його частина
🔒 ПЛОМБИ: cv=0 ⟹ база біт у біт · чужа γ₃ у поправці дисперсії мусить ламати
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

def closure(W, g3=None, cm=0.0, cv=0.0, lays=LAY):
    """cm — коеф. поправки СЕРЕДНЬОГО (працює, 0.40) · cv — ДИСПЕРСІЇ (нове)"""
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        act = (g3 is not None) and (i in lays)
        if act and cm != 0.0:
            m = m - cm*s*(g3[i]/6.0)*(t*ph)
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        e2 = (mp*mp + s*s)*Ph + (mp*s)*ph
        if act and cv != 0.0:
            e2 = e2 + cv*(g3[i]/3.0)*(s*s)*ph
        dd = np.maximum(e2 - m*m, 1e-30)
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
lg("банків %d" % len(B))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
ADJ0 = (1.0/(1.0/V_MC + 1.0/ANCH_E2))*MULT
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
def ev(cm, cv, mode="own", lays=LAY):
    a = []
    for k, b in enumerate(B):
        g = b["g3"] if mode == "own" else B[(k+1) % len(B)]["g3"]
        a.append(cerr(closure(b["W"], g, cm, cv, lays), b["T"]))
    return float(np.mean(a))
lg("база %.6e" % BASE)
v0 = ev(0.0, 0.0)
print("🔒 ПЛОМБА cm=cv=0: %.6e проти бази %.6e" % (v0, BASE))

CM = [0.0, 0.40]
CV = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]
print("\n%8s | %s" % ("cm \\ cv", " ".join("%7.2f" % c for c in CV)))
best = (0.0, 0.0, ADJ0/sc(BASE))
for cm in CM:
    row = []
    for cv in CV:
        v = ev(cm, cv); r = ADJ0/sc(v); row.append(r)
        if r > best[2]: best = (cm, cv, r)
    print("%8.2f | %s" % (cm, " ".join("%7.4f" % x for x in row)), flush=True)
    lg("  cm=%.2f" % cm)
print("\n🎯 найкраще cm=%.2f cv=%.2f ⟹ ×%.4f   (сама γ₃ у середньому: ×%.4f)"
      % (best[0], best[1], best[2], ADJ0/sc(ev(0.40, 0.0))))
if abs(best[1]) > 1e-9:
    vf = ev(best[0], best[1], "foreign")
    print("🔒 ЧУЖА γ₃ у тій самій конфігурації ⟹ ×%.4f" % (ADJ0/sc(vf)))
    # тонкий скан навколо оптимуму + перевірка параболою
    print("\n── тонкий скан cv при cm=%.2f ──" % best[0])
    lo, hi = best[1]-0.25, best[1]+0.25
    for cv in (lo, best[1], hi):
        v = ev(best[0], cv); print("  cv=%6.3f ⟹ ×%.4f" % (cv, ADJ0/sc(v)), flush=True)
lg("ГОТОВО")
