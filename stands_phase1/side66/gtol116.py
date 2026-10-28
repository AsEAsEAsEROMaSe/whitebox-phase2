"""С116: ЯКОЇ ТОЧНОСТІ `γ` ДОСТАТНЬО — І ЧИ Є В НИХ БЕЗКОШТОВНА ЧАСТИНА.

[edge116] точні понейронні γ₃,γ₄ знімають похибку одного кроку 0.0459% → ≤0.0108%
(підлога банку). Але γ треба ПОРАХУВАТИ. Питання ціни розпадається на три:
  1. скільки з ефекту дає СТАЛА γ (табличка, ціна НУЛЬ)?
  2. яку відносну точність треба, щоб узяти більшість?
  3. 🔒 чи не є весь ефект артефактом калібровки — контроль ЧУЖА / ПЕРЕМІШАНА
  4. чи вмикається γ на НАШИХ (mp,sd), а не лише на точних  [С101: edg ×1.02]

статистика банку: γ₃ сер ≈0.000 sd 0.104  ·  γ₄ сер +0.062 sd 0.016
⟹ у γ₃ вся інформація в РОЗКИДІ, у γ₄ — переважно в СЕРЕДНЬОМУ.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 стала γ₄ + точна γ₃ тримає >80% ефекту                     60%
  P2 30% відносного шуму в γ₃ з'їдає половину виграшу           55%
  P3 🔒 ЧУЖА γ₃ дає ≈нуль                                       85%
  P4 наші (mp,sd) + точні γ дають лише ×1.0…1.1 (повтор edg)    70%
"""
import sys, glob, numpy as np
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
NW = 1024; TRUE2 = 0.836447; D = 16
V_MC = 3.3478454e-06; MULT = 0.100131; ANCH = 1.4355576e-06/1.1417e-06

def psi(t): return phi(t) + t*Phi(t)
def cerr(x, T):
    a = (x@T)/max((x*x).sum(), 1e-300); return float(np.mean((a*x-T)**2))
def adj(c):
    e2 = c*ANCH; return (1.0/(1.0/V_MC + 1.0/e2))*MULT

def closure_mps(W):
    """НАШІ mp і sd на шарі 15"""
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

import pyarrow.parquet as pq
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
print("мереж %d" % len(B))
G3ALL = np.concatenate([b["g3"] for b in B]); G4ALL = np.concatenate([b["g4"] for b in B])
print("🔒 γ₃ сер %+.5f sd %.5f  ·  γ₄ сер %+.5f sd %.5f"
      % (G3ALL.mean(), G3ALL.std(), G4ALL.mean(), G4ALL.std()))

def pred(mp, sd, g3, g4):
    t = mp/sd; ph = phi(t)
    return sd*(psi(t) - (g3/6.0)*t*ph + (g4/24.0)*(t*t-1.0)*ph)

def run(tag, f_g3, f_g4, use_our=False, reps=1):
    acc = []
    for r in range(reps):
        for k, b in enumerate(B):
            mp = b["omp"] if use_our else b["mp"]
            sd = b["osd"] if use_our else b["sd"]
            acc.append(cerr(pred(mp, sd, f_g3(b, k, r), f_g4(b, k, r)), b["T"]))
    c = float(np.mean(acc)); return tag, c, 100*np.sqrt(c/TRUE2), adj(c)

Z  = lambda b,k,r: np.zeros(NW)
EX3 = lambda b,k,r: b["g3"];  EX4 = lambda b,k,r: b["g4"]
CN3 = lambda b,k,r: np.full(NW, G3ALL.mean()); CN4 = lambda b,k,r: np.full(NW, G4ALL.mean())
def FOR(key):
    return lambda b,k,r: B[(k+1) % len(B)][key]
def SHUF(key):
    return lambda b,k,r: np.random.RandomState(500+k*7+r).permutation(b[key])
def NOISE(key, eps):
    def f(b,k,r):
        rs = np.random.RandomState(9000+k*31+r*7+int(eps*1000))
        return b[key] + eps*b[key].std()*rs.standard_normal(NW)
    return f
def SHRINK(key, a):
    return lambda b,k,r: a*b[key]

ROWS = []
ROWS.append(run("гаус (γ=0)",            Z,   Z))
ROWS.append(run("точні γ₃γ₄  СТЕЛЯ",     EX3, EX4))
ROWS.append(run("точна γ₃ · СТАЛА γ₄",   EX3, CN4))
ROWS.append(run("СТАЛА γ₃(=0) · точна γ₄", Z, EX4))
ROWS.append(run("обидві СТАЛІ (табличка)", CN3, CN4))
print("\n── 1. ЩО ДАЄ БЕЗКОШТОВНА (СТАЛА) ЧАСТИНА ──")
print("%-26s %11s %9s %11s %s" % ("режим","косинусна","похибка%","adj","×до #1"))
for t,c,p,a in ROWS: print("%-26s %11.4e %8.4f%% %11.4e %6.2f" % (t,c,p,a,a/7.30e-09))

print("\n── 2. 🔒 КОНТРОЛІ (найважливіше) ──")
CT = [run("🔒 ЧУЖА γ₃ · точна γ₄",  FOR("g3"), EX4),
      run("🔒 ЧУЖА γ₃ · чужа γ₄",   FOR("g3"), FOR("g4")),
      run("🔒 ПЕРЕМІШ γ₃ · точна γ₄", SHUF("g3"), EX4, reps=5),
      run("🔒 ПЕРЕВЕРНУТА γ₃",      SHRINK("g3",-1.0), EX4)]
for t,c,p,a in CT: print("%-26s %11.4e %8.4f%% %11.4e %6.2f" % (t,c,p,a,a/7.30e-09))

print("\n── 3. ТОЛЕРАНТНІСТЬ ДО ШУМУ В γ₃ (γ₄ точна) ──")
print("%-10s %11s %9s %11s %s" % ("шум ε","косинусна","похибка%","adj","частка виграшу"))
g0 = ROWS[0][1]; g1 = ROWS[1][1]
for eps in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
    t,c,p,a = run("", NOISE("g3", eps), EX4, reps=4)
    frac = (np.sqrt(g0)-np.sqrt(c))/(np.sqrt(g0)-np.sqrt(g1))
    print("%-10.2f %11.4e %8.4f%% %11.4e %13.1f%%" % (eps,c,p,a,100*frac))

print("\n── 4. ЗМІЩЕННЯ (недооцінка γ₃ у a разів), γ₄ точна ──")
for a_ in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    t,c,p,ad = run("", SHRINK("g3", a_), EX4)
    print("  a=%.2f  %8.4f%%  adj %11.4e" % (a_, p, ad))

print("\n── 5. 🚨 НА НАШИХ (mp,sd) — ЧИ ВМИКАЄТЬСЯ γ ВЗАГАЛІ ──")
NR = [run("наші (mp,sd), γ=0",      Z,   Z,   use_our=True),
      run("наші (mp,sd) + точні γ", EX3, EX4, use_our=True),
      run("наші + точна γ₃",        EX3, Z,   use_our=True)]
b0 = NR[0][1]
for t,c,p,a in NR:
    print("%-26s %11.4e %8.4f%% %11.4e  ×до#1 %6.2f  ×база %.4f" % (t,c,p,a,a/7.30e-09, b0/c))
print("  [С101 давав edg ×1.02 проти edgm ×22.8 — той самий візерунок?]")

print("\n── 6. 🎯 РЕАЛІСТИЧНО: γ₃ З ТИХ САМИХ 6144 СЕМПЛІВ, ЩО ВЖЕ РАХУЄ МК-ГІЛКА ──")
print("   теоретичний шум оцінки γ₃ при N: sd ≈ √(6/N)")
for N in [6144, 16384, 65536]:
    e_abs = np.sqrt(6.0/N); e_rel = e_abs/G3ALL.std()
    t,c,pp,a = run("", NOISE("g3", e_rel), EX4, reps=6)
    frac = (np.sqrt(ROWS[0][1])-np.sqrt(c))/(np.sqrt(ROWS[0][1])-np.sqrt(ROWS[1][1]))
    print("   N=%-7d sd(γ̂₃)=%.4f ⟹ ε=%.2f · похибка %.4f%% · adj %.4e · виграшу %.0f%%"
          % (N, e_abs, e_rel, pp, a, 100*frac))
print("   [потрібно для #1: 0.0266%% · adj 7.30e-09]")

print("\n── 7. 🎯 ТЕ САМЕ, АЛЕ НА НАШИХ (mp,sd) — ЄДИНИЙ ЧЕСНИЙ РЕЖИМ ──")
for lab, f3 in [("γ=0", Z), ("точна γ₃", EX3), ("γ₃ з 6144 семплів", NOISE("g3", np.sqrt(6.0/6144)/G3ALL.std()))]:
    t,c,pp,a = run("", f3, EX4, use_our=True, reps=4)
    print("   наші (mp,sd) + %-22s похибка %.4f%%  adj %.4e" % (lab, pp, a))
