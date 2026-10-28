"""ДЕ САМЕ МИ ЗЛИВАЄМО ІНФОРМАЦІЮ В КУПУ (ідея ЮЗЕРА).

Кожна наша ручка — це КОЛАПС: багато чисел стискаються в одне.
    _W_BLEND  1024 нейрони → 1   (стеля вже зміряна: ×1.2096 [wneu116])
    κ         15 шарів → 1 · _LAM_FEAT 16 → 1 · _G3_C 7 → 1
Питання не «яке значення краще», а «СКІЛЬКИ КОШТУЄ САМ ФАКТ, ЩО ЧИСЛО ОДНЕ».

МЕТОД (як підкручують одну вагу і одразу дивляться): для кожної ручки на
кожному шарі — збурення ±ε, три точки, вершина параболи
    c* = c₀ + h·(e₋ − e₊)/(2(e₋ − 2e₀ + e₊))
дають і чутливість, і оптимальний крок БЕЗ сканування.
Плоский профіль ⟹ колапс нічого не руйнує. Профіль із формою ⟹ руйнує.

🔒 КРОС A→B ОБОВ'ЯЗКОВИЙ: 39 параметрів на 24 мережі перефітяться гарантовано.
🔮 P1 (50%) хоч одна ручка має пошарову форму, що дає ≥×1.03 на КРОСІ
🔮 P2 (65%) `_LAM_FEAT` найчутливіша (вона коштує +6.2% і глобальна)
🔮 P3 (40%) профілі виявляться плоскими ⟹ колапс безкоштовний, і це теж вирок
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
G3C, LAY3 = 0.40, tuple(range(9, 16))
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)

def closure(W, g3, kap, lam, gc):
    """kap, lam, gc — МАСИВИ по шарах (D,)"""
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*kap[i]
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if i in LAY3 and gc[i] != 0.0:
            m = m - gc[i]*s*(g3[i]/6.0)*(t*ph)
        if lam[i] != 0.0:
            m = m - lam[i]*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
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
K0 = np.full(D, KAPPA); L0 = np.full(D, LAM_FEAT); G0 = np.full(D, G3C)
ADJ0 = (1.0/(1.0/V_MC + 1.0/ANCH_E2))*MULT
def ev(idx, kap, lam, gc):
    return float(np.mean([cerr(closure(B[k]["W"], B[k]["g3"], kap, lam, gc), B[k]["T"])
                          for k in idx]))
A = list(range(0, len(B), 2)); Bx = list(range(1, len(B), 2))
baseA, baseB = ev(A, K0, L0, G0), ev(Bx, K0, L0, G0)
def sc(v, base): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/base))))*MULT
lg("база A %.6e · B %.6e" % (baseA, baseB))

KNOBS = [("kappa", K0, 0.0030, lambda a: (a, L0, G0), range(KSIG+1, D)),
         ("lam_feat", L0, 0.0500, lambda a: (K0, a, G0), range(0, D)),
         ("g3_c", G0, 0.1000, lambda a: (K0, L0, a), LAY3)]
OPT = {}
for name, base_arr, h, mk in [(n, b, h, m) for n, b, h, m, _ in KNOBS]:
    pass
for name, base_arr, h, mk, layers in KNOBS:
    print("\n── ручка `%s` (крок ±%.4g), похідні НА МЕРЕЖАХ A ──" % (name, h), flush=True)
    print("%5s %13s %13s %11s %11s" % ("шар", "e(−h)", "e(+h)", "d/dc", "вершина"), flush=True)
    opt = base_arr.copy()
    for L in layers:
        am = base_arr.copy(); am[L] -= h
        ap = base_arr.copy(); ap[L] += h
        em = ev(A, *mk(am)); ep = ev(A, *mk(ap))
        den = em - 2*baseA + ep
        star = base_arr[L] + (h*(em - ep)/(2*den) if abs(den) > 1e-300 else 0.0)
        star = float(np.clip(star, base_arr[L]-5*h, base_arr[L]+5*h))
        opt[L] = star
        print("%5d %13.6e %13.6e %11.3e %11.5f"
              % (L, em, ep, (ep-em)/(2*h), star), flush=True)
    OPT[name] = opt
    vA = ev(A, *mk(opt)); vB = ev(Bx, *mk(opt))
    print("  ⟹ профіль: A(in-sample) ×%.4f · 🔒 B(крос) ×%.4f"
          % (sc(baseA,baseA)/sc(vA,baseA), sc(baseB,baseB)/sc(vB,baseB)), flush=True)
    lg("  %s готово" % name)

vA = ev(A, OPT["kappa"], OPT["lam_feat"], OPT["g3_c"])
vB = ev(Bx, OPT["kappa"], OPT["lam_feat"], OPT["g3_c"])
print("\n🎯 ВСІ ТРИ ПРОФІЛІ РАЗОМ: A ×%.4f · 🔒 B(крос) ×%.4f"
      % (sc(baseA,baseA)/sc(vA,baseA), sc(baseB,baseB)/sc(vB,baseB)))
lg("ГОТОВО")
