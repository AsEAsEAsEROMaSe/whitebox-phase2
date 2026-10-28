"""С116: ЧИ ВМИКАЄТЬСЯ γ₃ НА МІЛКІШИХ ШАРАХ І ЧИ ПОШИРЮЄТЬСЯ ПОПРАВКА ВНИЗ?

[edge116/sw116] на шарі 15 γ₃ дає ×18 при ТОЧНИХ моментах і ×1.04 при наших,
бо ворота — ДОБУТОК точності (mp ≲0.05%, sd ≲0.7%), а наші там 0.21% / 2.94%.
🔑 АЛЕ похибка НАКОПИЧУЄТЬСЯ ⟹ на мілкіших шарах наші моменти ТОЧНІШІ.
Якщо десь `mp` ще ≲0.05%, γ₃ там увімкнеться, і виправлений крок піде В РЕКУРСІЮ.

Це НЕ те саме, що «часткове наближення до істини» (закон С115-ж): ми не
підміняємо канал стану напівточним значенням, а виправляємо САМУ ФОРМУЛУ кроку.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 профіль похибки нашого `mp` монотонно росте з глибиною, на ш8 ≲0.1%   70%
  P2 γ₃ на ВСІХ глибоких шарах дасть більше, ніж лише на ш15               55%
  P3 при НАШИХ моментах сумарний ефект лишиться <×1.15                     65%
  P4 🔒 чужа γ₃ зламає                                                     85%
ГЕЙТ: якщо γ₃@всі_глибокі на НАШИХ моментах дає <×1.05 — гілка мертва
      і `mp` лишається єдиним відкритим об'єктом.
"""
import sys, glob, time, numpy as np
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
import pyarrow.parquet as pq
NW, D = 1024, 16
TRUE2 = 0.836447; V_MC = 3.3478454e-06; MULT = 0.100131; ANCH = 1.4355576e-06/1.1417e-06
LAY = [0, 8, 12, 13, 14, 15]
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = (x@T)/max((x*x).sum(), 1e-300); return float(np.mean((a*x-T)**2))
def adj(c):
    e2 = c*ANCH; return (1.0/(1.0/V_MC + 1.0/e2))*MULT

def closure(W, G3=None, G4C=0.0, apply_at=(), diag=None):
    """G3[i] — понейронна γ₃ передактивації шару i; застосовується у КРОЦІ"""
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        if diag is not None and i in diag: diag[i] = (mp.copy(), s.copy())
        m = mp*Ph + s*ph
        if i in apply_at and G3 is not None:
            m = m - s*(G3[i]/6.0)*t*ph + s*(G4C/24.0)*(t*t-1.0)*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(D))
        if i == D-1: return m
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
HI = {}
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz")):
    d = np.load(f, allow_pickle=True); HI[str(d["name"])] = d["m"][15].astype(np.float64)
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gam116_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    if nm not in HI: continue
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), dtype=np.float64).reshape(D,NW,NW)
    B.append(dict(name=nm, W=W, T=HI[nm],
                  mpT={L: d["mp_%d" % L] for L in LAY},
                  sdT={L: d["sd_%d" % L] for L in LAY},
                  g3={L: d["g3_%d" % L] for L in LAY},
                  g4={L: d["g4_%d" % L] for L in LAY}))
G4C = float(np.mean([b["g4"][15].mean() for b in B]))
lg("мереж %d · стала γ₄ %.5f" % (len(B), G4C))

print("\n🔒 P1 — ПРОФІЛЬ ПОХИБКИ НАШИХ МОМЕНТІВ ПО ГЛИБИНІ")
print("шар   похибка mp %   похибка sd %   сер |t|   γ₃ sd    ворота (mp≲0.05, sd≲0.7)")
prof = {}
for L in LAY:
    em, es, mt, gs = [], [], [], []
    for b in B:
        dg = {L: None}
        closure(b["W"], diag=dg)
        mp, sd = dg[L]
        em.append(np.linalg.norm(mp-b["mpT"][L])/np.linalg.norm(b["mpT"][L]))
        es.append(np.linalg.norm(sd-b["sdT"][L])/np.linalg.norm(b["sdT"][L]))
        mt.append(np.mean(np.abs(mp/sd))); gs.append(b["g3"][L].std())
    prof[L] = (100*np.mean(em), 100*np.mean(es))
    ok = "ТАК" if prof[L][0] <= 0.05 and prof[L][1] <= 0.7 else "ні"
    print("  %-3d   %10.4f   %10.4f   %7.3f   %.4f   %s"
          % (L, prof[L][0], prof[L][1], np.mean(mt), np.mean(gs), ok))

def run(tag, at, g3src="own", reps=1):
    acc = []
    for r in range(reps):
        for k, b in enumerate(B):
            if g3src == "own":     G3 = b["g3"]
            elif g3src == "foreign": G3 = B[(k+1) % len(B)]["g3"]
            elif g3src == "shuf":
                rs = np.random.RandomState(400+k+r)
                G3 = {L: rs.permutation(b["g3"][L]) for L in LAY}
            else:                                        # вибіркова, N=6144
                rs = np.random.RandomState(900+k*13+r)
                e = np.sqrt(6.0/6144)
                G3 = {L: b["g3"][L] + e*rs.standard_normal(NW) for L in LAY}
            acc.append(cerr(closure(b["W"], G3, G4C, at), b["T"]))
    c = float(np.mean(acc)); return tag, c, 100*np.sqrt(c/TRUE2), adj(c)

base = run("база (γ=0)", ())
print("\n── γ₃ У РЕКУРСІЇ, НА НАШИХ МОМЕНТАХ (як буде насправді) ──")
print("%-30s %11s %9s %12s %s" % ("режим","косинусна","похибка%","adj","×база"))
ROWS = [base]
for L in [8, 12, 13, 14, 15]:
    ROWS.append(run("γ₃@%d" % L, (L,)))
ROWS.append(run("γ₃@12,13,14,15", (12,13,14,15)))
ROWS.append(run("γ₃@8,12,13,14,15", (8,12,13,14,15)))
for t,c,p,a in ROWS:
    print("%-30s %11.4e %8.4f%% %12.4e  ×%.4f" % (t,c,p,a,base[1]/c))

print("\n── 🔒 КОНТРОЛІ на найкращій конфігурації ──")
bestat = (12,13,14,15)
for src, lab in [("foreign","🔒 ЧУЖА γ₃"), ("shuf","🔒 ПЕРЕМІШАНА γ₃"), ("samp","вибіркова γ₃ (N=6144)")]:
    t,c,p,a = run("", bestat, src, reps=3 if src!="foreign" else 1)
    print("  %-24s %11.4e %8.4f%% %12.4e  ×%.4f" % (lab,c,p,a,base[1]/c))

bb = min(ROWS[1:], key=lambda z: z[1])
print("\n🎯 ГЕЙТ ×1.05: найкраще %s ×%.4f ⟹ %s"
      % (bb[0], base[1]/bb[1], "ПРОЙДЕНО" if base[1]/bb[1] >= 1.05 else "НЕ пройдено"))
lg("ГОТОВО")
