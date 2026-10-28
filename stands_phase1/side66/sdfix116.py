"""С116: ВИПРАВИТИ СИСТЕМАТИЧНУ ЧАСТИНУ `sd` ДВОМА ЧИСЛАМИ, НЕ ЗАМІНЮЮЧИ ЇЇ.

[real116] вибіркова sd точніша у ×3.26, але ШКОДИТЬ (×0.849): її похибка
НЕЗАЛЕЖНА по нейронах, а наша — майже СПІЛЬНИЙ МНОЖНИК, який `_ai` бере даром.
🔑 виведено: dx/ds = φ(t) ⟹ відн. чутливість φ(t)/ψ(t) = 0.016 при t=2.2,
   але >1 при t<0 ⟹ незалежний шум підсилюється саме на малих |t|.

⟹ беремо з семплів не САМУ sd, а ДВА КОЕФІЦІЄНТИ її систематичної поправки:
      log(sd_true/sd_our) ≈ a + b·t²          [корпус sig1: R² LOO 0.883 на ш15]
   1024 нейрони ⟹ шум у (a,b) мізерний, структура похибки ЗБЕРІГАЄТЬСЯ.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 фіт (a,b) із ВИБІРКОВОЇ sd знімає >60% похибки sd                 60%
  P2 після цього виграш у балі ×1.05…1.15                             50%
  P3 🔒 чужі (a,b) дадуть менше, але НЕ нуль (є спільна компонента)    70%
ГЕЙТ: <×1.05 у adj ⟹ не варте сабміту (планка піднята, бо ×1.029 уже є).
"""
import sys, glob, numpy as np
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
import pyarrow.parquet as pq
NW = 1024; D = 16; TRUE2 = 0.836447
V_MC = 3.3478454e-06; MULT = 0.100131; ANCH = 1.4355576e-06/1.1417e-06
NSAMP = 6144; E_SD = 1.0/np.sqrt(2.0*NSAMP); E_G3 = np.sqrt(6.0/NSAMP)
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
    B.append(dict(name=nm, mp=d["mp_15"], sd=d["sd_15"], g3=d["g3_15"],
                  T=HI[nm], omp=omp, osd=osd, t=omp/osd))
G4C = 0.06236
print("мереж %d" % len(B))

def design(t): return np.stack([np.ones(NW), t*t], 1)

def fit_ab(b, target_sd):
    r = np.log(np.maximum(target_sd, 1e-30)/np.maximum(b["osd"], 1e-30))
    X = design(b["t"])
    return np.linalg.lstsq(X, r, rcond=None)[0]

print("\n── ЯКІСТЬ ФІТУ log(sd_true/sd_our) ≈ a + b·t² ──")
print("%-18s %8s %8s %8s | %10s %10s" % ("мережа","a","b","R²","sd до %","sd після %"))
for b in B:
    ab = fit_ab(b, b["sd"]); X = design(b["t"])
    r = np.log(b["sd"]/b["osd"]); pr = X @ ab
    R2 = 1.0 - np.var(r-pr)/np.var(r)
    s2 = b["osd"]*np.exp(pr)
    print("%-18s %8.5f %8.5f %8.4f | %9.3f%% %9.3f%%"
          % (b["name"], ab[0], ab[1], R2,
             100*np.linalg.norm(b["osd"]-b["sd"])/np.linalg.norm(b["sd"]),
             100*np.linalg.norm(s2-b["sd"])/np.linalg.norm(b["sd"])))

def run(tag, sd_mode, g3_mode, reps=8, foreign=False):
    acc, sderr = [], []
    for r in range(reps):
        for k, b in enumerate(B):
            rs = np.random.RandomState(31000 + k*89 + r*17)
            sd_s = b["sd"]*(1.0 + E_SD*rs.standard_normal(NW))
            if   sd_mode == "our":    sd = b["osd"]
            elif sd_mode == "exact":  sd = b["sd"]
            elif sd_mode == "samp":   sd = sd_s
            else:                                                   # фіт (a,b)
                src = b if not foreign else B[(k+1) % len(B)]
                tgt = src["sd"] if sd_mode == "fit_exact" else \
                      src["sd"]*(1.0 + E_SD*np.random.RandomState(777+k+r).standard_normal(NW))
                ab = fit_ab(src, tgt)
                sd = b["osd"]*np.exp(design(b["t"]) @ ab)
            g3 = np.zeros(NW) if g3_mode == "no" else (
                 b["g3"] if g3_mode == "exact" else b["g3"] + E_G3*rs.standard_normal(NW))
            t = b["omp"]/sd; ph = phi(t)
            x = sd*(psi(t) - (g3/6.0)*t*ph + (G4C/24.0)*(t*t-1.0)*ph)
            acc.append(cerr(x, b["T"]))
            sderr.append(np.linalg.norm(sd-b["sd"])/np.linalg.norm(b["sd"]))
    c = float(np.mean(acc)); return tag, c, 100*np.sqrt(c/TRUE2), adj(c), 100*np.mean(sderr)

BASE = run("НАША БАЗА (наші sd, γ=0)", "our", "no", reps=1)
R = [BASE,
     run("наші sd + вибіркова γ₃",        "our",  "samp"),
     run("ФІТ(a,b) з вибіркової sd, γ=0", "fit_samp", "no"),
     run("ФІТ(a,b) + вибіркова γ₃",       "fit_samp", "samp"),
     run("ФІТ(a,b) + ТОЧНА γ₃",           "fit_samp", "exact"),
     run("🔒 ФІТ із ТОЧНОЇ sd + вибір.γ₃","fit_exact","samp"),
     run("🔒 ТОЧНА sd + вибіркова γ₃",    "exact","samp"),
     run("🔒 ТОЧНА sd + ТОЧНА γ₃",        "exact","exact")]
print("\n%-32s %11s %9s %12s %8s %s" % ("режим","косинусна","похибка%","adj","×база","sd похибка%"))
for t,c,p,a,se in R:
    print("%-32s %11.4e %8.4f%% %12.4e  ×%.4f %8.3f%%" % (t,c,p,a,BASE[3]/a,se))

print("\n── 🔒 КОНТРОЛЬ: ЧУЖІ (a,b) ──")
t,c,p,a,se = run("", "fit_samp", "samp", foreign=True)
print("  чужі (a,b) + вибіркова γ₃:  %.4f%%  adj %.4e  ×база %.4f  sd похибка %.3f%%"
      % (p, a, BASE[3]/a, se))

best = min([r for r in R[1:]], key=lambda z: z[3])
print("\n🎯 ПІДСУМОК")
print("  база стенду           %.4e" % BASE[3])
print("  найкраще реалістичне  %.4e (%s) ⟹ ×%.4f" % (best[3], best[0], BASE[3]/best[3]))
print("  ГЕЙТ ×1.05 ⟹ %s" % ("ПРОЙДЕНО" if BASE[3]/best[3] >= 1.05 else "НЕ пройдено"))
