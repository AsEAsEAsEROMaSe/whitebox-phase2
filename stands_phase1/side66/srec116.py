"""С116-б: ПОПРАВКА `s` УСЕРЕДИНІ РЕКУРСІЇ (а не на фініші, як робив sdfix116).

ЩО НЕЗМІРЯНЕ: `sdfix116` брав замикання як є, і множив `s` ЛИШЕ на шарі 15.
Але `m = mp·Φ(t) + s·φ(t)` ⟹ похибка `s` тече в `m`, отже в `mp` НАСТУПНОГО
шару, з коефіцієнтом передачі `φ(t)/ψ(t)`. Він ВЕЛИКИЙ на мілких шарах
(t≈0.5 ⟹ 0.52) і мізерний на глибоких (t≈2.7 ⟹ 0.004).
⟹ якщо канал живий, він живе на шарах 1..6, де корпус похибку `s` НЕ МІРЯВ
(profile prop116 починався з шару 8).

ВКИДАННЯ УЗГОДЖЕНЕ: `cp ← Dg·cp·Dg`, `Dg = diag(s_new/s_old)` — міняє
діагональ І позадіагональ разом (гейт узгодженості C).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (записано ДО того, як банк дорахувався):
  P1 (60%) sTRUE@ALL ріже похибку mp₁₅ у ≥×1.5
  P2 (35%) ріже у ≥×3
  P3 (75%) внесок мілких шарів (1..6) БІЛЬШИЙ за внесок глибоких (9..15)
  P4 (50%) наша `s` на мілких уже точна (<0.5%) ⟹ стеля мала, ідея вмирає
🔒 ПЛОМБИ: (A) шар 0 — похибка `s` = шум банку (cp=W₀ᵀW₀ ТОЧНЕ)
           (B) sTRUE@жодного ⟹ база БІТ У БІТ
           (C) sTRUE@ALL + mpTRUE@ALL ⟹ 0 (тавтологія)
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def psi(t): return phi(t) + t*Phi(t)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
BANK = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    BANK.append(dict(name=nm, W=W,
                     mp=np.stack([d["mp_%d" % L] for L in range(D)]),
                     sd=np.stack([d["sd_%d" % L] for L in range(D)]),
                     g3=np.stack([d["g3_%d" % L] for L in range(D)]),
                     T=d["relu_15"]))
lg("банків %d" % len(BANK))

def closure(W, sfix=None, mpfix=None):
    """sfix[l] — цільове `s` на шарі l (або None) · mpfix — те саме для mp.
       Повертає (m_15, OMP, OSD): наші mp/s по шарах."""
    m = np.zeros(NW); C = np.eye(NW); OMP = []; OSD = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20))
        if sfix is not None and sfix[i] is not None:
            g = np.maximum(sfix[i], 1e-20)/s
            cp_ = cp_*np.outer(g, g); s = np.maximum(sfix[i], 1e-20)
        mp = m @ w
        if mpfix is not None and mpfix[i] is not None: mp = mpfix[i]
        OMP.append(mp.copy()); OSD.append(s.copy())
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m, np.stack(OMP), np.stack(OSD)

rel = lambda a, b: 100*np.linalg.norm(a-b)/np.linalg.norm(b)

# ── профіль: похибка mp/sd по глибині + коефіцієнт передачі ──
lg("── ПРОФІЛЬ ПО ГЛИБИНІ (%d мереж) ──" % len(BANK))
P = np.zeros((len(BANK), D, 4))
for k, b in enumerate(BANK):
    _, omp, osd = closure(b["W"])
    b["omp"], b["osd"] = omp, osd
    for L in range(D):
        t = omp[L]/osd[L]
        P[k, L] = [rel(omp[L], b["mp"][L]), rel(osd[L], b["sd"][L]),
                   float(np.mean(np.abs(t))), float(np.mean(phi(t)/np.maximum(psi(t), 1e-12)))]
M = P.mean(0)
print("%4s %10s %10s %8s %10s %12s" % ("шар","mp %","sd %","сер|t|","φ/ψ","sd%·φ/ψ"))
for L in range(D):
    print("%4d %9.4f%% %9.4f%% %8.3f %10.4f %11.4f%%"
          % (L, M[L,0], M[L,1], M[L,2], M[L,3], M[L,1]*M[L,3]))

# ── фіт log(s_true/s_our) = a + b·t² ПОШАРОВО ──
AB = np.zeros((len(BANK), D, 2)); R2 = np.zeros((len(BANK), D))
for k, b in enumerate(BANK):
    for L in range(D):
        t = b["omp"][L]/b["osd"][L]
        X = np.stack([np.ones(NW), t*t], 1)
        r = np.log(np.maximum(b["sd"][L], 1e-30)/np.maximum(b["osd"][L], 1e-30))
        ab = np.linalg.lstsq(X, r, rcond=None)[0]; AB[k, L] = ab
        R2[k, L] = 1.0 - np.var(r - X@ab)/max(np.var(r), 1e-30)
print("\n── ФІТ log(s_true/s_our)=a+b·t² ПОШАРОВО (сер по мережах) ──")
print("%4s %10s %10s %8s %12s" % ("шар","a","b","R²","розкид b %"))
for L in range(D):
    sb = 100*AB[:,L,1].std()/max(abs(AB[:,L,1].mean()), 1e-30)
    print("%4d %10.5f %10.5f %8.4f %11.1f%%" % (L, AB[:,L,0].mean(), AB[:,L,1].mean(), R2[:,L].mean(), sb))

def sfit(b, ab):
    out = []
    for L in range(D):
        t = b["omp"][L]/b["osd"][L]
        out.append(b["osd"][L]*np.exp(ab[L,0] + ab[L,1]*t*t))
    return out

def run(tag, mk):
    ce, mpe = [], []
    for k, b in enumerate(BANK):
        sf = mk(k, b)
        m, omp, _ = closure(b["W"], sfix=sf)
        ce.append(cerr(m, b["T"])); mpe.append(rel(omp[15], b["mp"][15]))
    c = float(np.mean(ce)); e2 = c*(ANCH_E2/BASE_C) if BASE_C else c
    a = (1.0/(1.0/V_MC + 1.0/e2))*MULT
    return tag, c, 100*np.sqrt(c/0.909290), a, float(np.mean(mpe))

NONE = [None]*D
BASE_C = None
BASE_C = float(np.mean([cerr(closure(b["W"])[0], b["T"]) for b in BANK]))
lg("база (косинусна) %.6e" % BASE_C)

SH = list(range(1, 7)); DP = list(range(9, 16))
def mk_true(idx):
    return lambda k, b: [b["sd"][L] if L in idx else None for L in range(D)]
def mk_loo(idx):
    def f(k, b):
        ab = AB[[j for j in range(len(BANK)) if j != k]].mean(0)
        s = sfit(b, ab); return [s[L] if L in idx else None for L in range(D)]
    return f
ALL = list(range(D))
R = [run("база", lambda k, b: NONE),
     run("🔒 sTRUE@ALL",            mk_true(ALL)),
     run("sTRUE@мілкі 1..6",        mk_true(SH)),
     run("sTRUE@глибокі 9..15",     mk_true(DP)),
     run("sTRUE@15 (як sdfix116)",  mk_true([15])),
     run("ФІТ-LOO@ALL (реалістично)",  mk_loo(ALL)),
     run("ФІТ-LOO@мілкі 1..6",         mk_loo(SH)),
     run("ФІТ-LOO@глибокі 9..15",      mk_loo(DP))]
print("\n%-30s %12s %9s %12s %8s %10s" % ("режим","косинусна","похибка%","adj","×база","mp₁₅ %"))
for t, c, p, a, me in R:
    print("%-30s %12.5e %8.4f%% %12.4e  ×%.4f %9.4f%%" % (t, c, p, a, R[0][3]/a, me))

# 🔒 ПЛОМБА C: істинні s І mp на всіх шарах ⟹ ЧИСТА ціна гаусового кроку
cc = []
for b in BANK:
    m, _, _ = closure(b["W"], sfix=[b["sd"][L] for L in range(D)],
                              mpfix=[b["mp"][L] for L in range(D)])
    cc.append(cerr(m, b["T"]))
print("\n🔒 C: sTRUE+mpTRUE@ALL ⟹ %.4f%% (ceil116b дав 0.0459%% — інший стенд)"
      % (100*np.sqrt(float(np.mean(cc))/0.909290)))
print("🔒 A: похибка sd на шарі 0 = %.4f%% (шум банку)" % M[0,1])
lg("ГОТОВО")
