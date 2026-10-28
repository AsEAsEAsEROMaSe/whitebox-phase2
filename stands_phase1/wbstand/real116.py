"""С116: РЕАЛІСТИЧНИЙ РЕЖИМ — усе, що можна взяти БЕЗ оракула.

[sw116] ворота — ДОБУТОК точності (mp, sd); синергія ×8.4.
Арифметика семплів при N=6144 (їх МК-гілка вже рахує):
   mp: похибка 1/(t√N) = 0.58%  проти нашої 0.2096%  ⟹ семпли ГІРШІ, беремо наше
   sd: похибка 1/√(2N)  = 0.90%  проти нашої 2.94%   ⟹ семпли КРАЩІ у ×3.3
   γ₃: похибка √(6/N)   = 0.031  при sd(γ₃)=0.105    ⟹ ε=0.30

🔑 І ТРЕТЄ: `sd` можна не ВИБИРАТИ, а ЗМІШАТИ — дві незалежні оцінки
(замикальна і вибіркова) з оберненими дисперсіями дають кращу за обидві.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 вибіркова sd + вибіркова γ₃ дасть ×1.05…1.15 у балі        60%
  P2 змішана sd поб'є обидві окремі                             55%
  P3 оптимальна вага змішування ближча до вибіркової (>0.6)     60%
ГЕЙТ: <×1.03 у adj ⟹ не варте сабміту.
"""
import sys, glob, numpy as np
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
import pyarrow.parquet as pq
NW = 1024; D = 16; TRUE2 = 0.836447
V_MC = 3.3478454e-06; MULT = 0.100131; ANCH = 1.4355576e-06/1.1417e-06
NSAMP = 6144
def psi(t): return phi(t) + t*Phi(t)
def cerr(x, T):
    a = (x@T)/max((x*x).sum(), 1e-300); return float(np.mean((a*x-T)**2))
def adj(c):
    e2 = c*ANCH; return (1.0/(1.0/V_MC + 1.0/e2))*MULT

def closure_mps(W):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        if i == D-1: return mp, s
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
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
    omp, osd = closure_mps(W)
    B.append(dict(name=nm, mp=d["mp_15"], sd=d["sd_15"], g3=d["g3_15"], g4=d["g4_15"],
                  T=HI[nm], omp=omp, osd=osd))
G4C = float(np.mean([b["g4"].mean() for b in B]))
E_SD = 1.0/np.sqrt(2.0*NSAMP)          # відн. похибка вибіркової sd
E_G3 = np.sqrt(6.0/NSAMP)              # абс. похибка вибіркової γ₃
print("мереж %d · N=%d · E_sd %.4f%% · E_γ₃ %.4f · стала γ₄ %.5f"
      % (len(B), NSAMP, 100*E_SD, E_G3, G4C))
esd = np.mean([np.linalg.norm(b["osd"]-b["sd"])/np.linalg.norm(b["sd"]) for b in B])
print("🔒 наша sd похибка %.4f%%  ·  вибіркова %.4f%%  ⟹ ×%.2f" % (100*esd, 100*E_SD, esd/E_SD))

def run(tag, sd_kind, g3_kind, wmix=None, reps=8):
    acc = []
    for r in range(reps):
        for k, b in enumerate(B):
            rs = np.random.RandomState(20000 + k*97 + r*13)
            sd_s = b["sd"]*(1.0 + E_SD*rs.standard_normal(NW))     # вибіркова
            if   sd_kind == "our":   sd = b["osd"]
            elif sd_kind == "samp":  sd = sd_s
            elif sd_kind == "exact": sd = b["sd"]
            else:                                                   # змішана
                sd = wmix*sd_s + (1.0-wmix)*b["osd"]
            if   g3_kind == "no":    g3 = np.zeros(NW)
            elif g3_kind == "samp":  g3 = b["g3"] + E_G3*rs.standard_normal(NW)
            else:                    g3 = b["g3"]
            mp = b["omp"]
            t = mp/sd; ph = phi(t)
            x = sd*(psi(t) - (g3/6.0)*t*ph + (G4C/24.0)*(t*t-1.0)*ph)
            acc.append(cerr(x, b["T"]))
    c = float(np.mean(acc)); return tag, c, 100*np.sqrt(c/TRUE2), adj(c)

BASE = run("НАША БАЗА: наші sd, γ=0", "our", "no", reps=1)
ROWS = [BASE,
        run("наші sd + вибіркова γ₃",      "our",  "samp"),
        run("вибіркова sd, γ=0",           "samp", "no"),
        run("вибіркова sd + вибіркова γ₃", "samp", "samp"),
        run("вибіркова sd + ТОЧНА γ₃",     "samp", "exact"),
        run("🔒 ТОЧНА sd + вибіркова γ₃",  "exact","samp"),
        run("🔒 ТОЧНА sd + ТОЧНА γ₃",      "exact","exact")]
print("\n%-32s %11s %9s %12s %s" % ("режим","косинусна","похибка%","adj","×база"))
for t,c,p,a in ROWS:
    print("%-32s %11.4e %8.4f%% %12.4e  ×%.4f" % (t,c,p,a,BASE[3]/a))

print("\n── ЗМІШУВАННЯ sd: w·вибіркова + (1−w)·замикальна ──")
print("%-8s %11s %9s %12s %s" % ("w","косинусна","похибка%","adj","×база"))
best = None
for w in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]:
    t,c,p,a = run("", "mix", "samp", wmix=w)
    star = ""
    if best is None or a < best[1]: best = (w, a); star = ""
    print("%-8.2f %11.4e %8.4f%% %12.4e  ×%.4f" % (w,c,p,a,BASE[3]/a))
print("  ⟹ найкраще w=%.2f  ×%.4f у балі" % (best[0], BASE[3]/best[1]))

print("\n🎯 ПІДСУМОК")
print("  наш реальний adj (328903)   1.0108e-07")
print("  модельна база стенду        %.4e" % BASE[3])
print("  найкраще реалістичне        %.4e  ⟹ ×%.3f" % (best[1], BASE[3]/best[1]))
print("  ГЕЙТ був ×1.03 ⟹ %s" % ("ПРОЙДЕНО" if BASE[3]/best[1] >= 1.03 else "не пройдено"))
