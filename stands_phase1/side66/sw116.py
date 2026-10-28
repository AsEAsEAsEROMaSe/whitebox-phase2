"""С116: ПОРІГ УВІМКНЕННЯ. Наскільки точними мусять бути (mp, sd) на шарі 15,
щоб понейронна γ₃ почала давати виграш?

[gtol116] на ТОЧНИХ моментах γ₃ дає ×18.2 · на НАШИХ — ×1.04. Це четверте
повторення корпусного «γ вмикається лише на точних (m,C)». Але ПОРІГ ніколи
не мірявся — лише два кінці осі. Міряємо КРИВУ.

  mp(λ) = (1−λ)·наше + λ·істинне ,  так само sd(λ)
  окремо: лише mp · лише sd · обидва — щоб знати, ЯКИЙ канал є воротами

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 крива ОПУКЛА: до λ=0.9 виграшу від γ₃ майже нема          60%
  P2 воротами є `mp`, не `sd`                                  65%
  P3 при λ=0.5 виграш від γ₃ <×1.3                             70%
"""
import sys, glob, numpy as np
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
import pyarrow.parquet as pq
NW = 1024; D = 16; TRUE2 = 0.836447
V_MC = 3.3478454e-06; MULT = 0.100131; ANCH = 1.4355576e-06/1.1417e-06
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
print("мереж %d · стала γ₄ = %.5f" % (len(B), G4C))
print("🔒 відносна похибка наших моментів на ш15:")
for b in B[:3]:
    print("   %-18s mp %.4f%%   sd %.4f%%"
          % (b["name"], 100*np.linalg.norm(b["omp"]-b["mp"])/np.linalg.norm(b["mp"]),
             100*np.linalg.norm(b["osd"]-b["sd"])/np.linalg.norm(b["sd"])))
emp = np.mean([np.linalg.norm(b["omp"]-b["mp"])/np.linalg.norm(b["mp"]) for b in B])
esd = np.mean([np.linalg.norm(b["osd"]-b["sd"])/np.linalg.norm(b["sd"]) for b in B])
print("   сер: mp %.4f%%  ·  sd %.4f%%" % (100*emp, 100*esd))

def ev(lam_mp, lam_sd, g3on):
    acc = []
    for b in B:
        mp = (1-lam_mp)*b["omp"] + lam_mp*b["mp"]
        sd = (1-lam_sd)*b["osd"] + lam_sd*b["sd"]
        t = mp/sd; ph = phi(t)
        g3 = b["g3"] if g3on else np.zeros(NW)
        x = sd*(psi(t) - (g3/6.0)*t*ph + (G4C/24.0)*(t*t-1.0)*ph)
        acc.append(cerr(x, b["T"]))
    c = float(np.mean(acc)); return c, 100*np.sqrt(c/TRUE2), adj(c)

LAMS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
print("\n── ОБИДВА КАНАЛИ РАЗОМ ──")
print("%-6s %10s %10s %9s %12s %s" % ("λ","γ=0 %","+γ₃ %","×від γ₃","adj(+γ₃)","×до #1"))
for L in LAMS:
    c0,p0,a0 = ev(L,L,False); c1,p1,a1 = ev(L,L,True)
    print("%-6.2f %9.4f%% %9.4f%% %9.3f %12.4e %6.2f" % (L,p0,p1,c0/c1,a1,a1/7.30e-09))

print("\n── ЛИШЕ mp ТОЧНЕ (sd наше) ──")
print("%-6s %10s %10s %9s %12s" % ("λ_mp","γ=0 %","+γ₃ %","×від γ₃","adj(+γ₃)"))
for L in LAMS:
    c0,p0,_ = ev(L,0.0,False); c1,p1,a1 = ev(L,0.0,True)
    print("%-6.2f %9.4f%% %9.4f%% %9.3f %12.4e" % (L,p0,p1,c0/c1,a1))

print("\n── ЛИШЕ sd ТОЧНЕ (mp наше) ──")
print("%-6s %10s %10s %9s %12s" % ("λ_sd","γ=0 %","+γ₃ %","×від γ₃","adj(+γ₃)"))
for L in LAMS:
    c0,p0,_ = ev(0.0,L,False); c1,p1,a1 = ev(0.0,L,True)
    print("%-6.2f %9.4f%% %9.4f%% %9.3f %12.4e" % (L,p0,p1,c0/c1,a1))

print("\n🎯 ПОРІГ: яка ТОЧНІСТЬ mp потрібна, щоб γ₃ дала ×2")
for L in LAMS:
    c0,_,_ = ev(L,L,False); c1,_,_ = ev(L,L,True)
    if c0/c1 >= 2.0:
        print("   λ=%.2f ⟹ похибка mp %.4f%% (наша зараз %.4f%%) ⟹ треба ×%.1f краще"
              % (L, 100*emp*(1-L), 100*emp, 1.0/max(1-L,1e-9))); break
else:
    print("   ×2 не досягнуто навіть при λ=1 — перевір")
