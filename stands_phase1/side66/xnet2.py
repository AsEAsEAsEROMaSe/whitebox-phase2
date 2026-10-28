"""С116-B: ЧИ Є ФОРМА ПОХИБКИ СПЕКТРАЛЬНОЮ ФУНКЦІЄЮ ЗАМИКАЛЬНОЇ `C`?

МОТИВ. (A) закрила понейронні координати: усе, що є функцією `t`, лежить
уздовж `m`, а ціль `e = a·m − T` ортогональна до `m` ЗА ПОБУДОВОЮ. Потрібна
координата ПОЗА сім'єю `t`. Кандидат зміряний [spec.py]: форма сидить у
власних напрямках 2…64 матриці Cov(h) — 15.8% енергії при ізотропних 6.1%.

🚨 ПЕРЕШКОДА, НАЗВАНА ДО ПРОГОНУ: знак і порядок власних векторів між мережами
довільні ⟹ коефіцієнт, навчений на A, на B не значить нічого. ОБХІД: будувати
поправку як МАТРИЧНУ ФУНКЦІЮ, де `u_k` входить ДВІЧІ і знак скорочується:

    ê = Σ_k g(λ̃_k)·⟨v,u_k⟩·u_k  =  U diag(g(λ̃)) Uᵀ v ,   λ̃_k = λ_k/mean(λ)

`g` — універсальна функція нормованого власного числа (бінами по log λ̃),
`v` — обчислюваний вектор мережі. Тоді ВСЯ персональність у (C, v), а
переноситься лише `g`. Це і є коректна форма питання «крос-мережевий механізм».

СТАДІЯ 1 (тут): чи передбачається `e_L` узагалі. Якщо ні in-sample — клас
мертвий без пропагації. СТАДІЯ 2 (лише якщо пройде): пропагувати в рекурсію.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 in-sample стеля спектрального оператора ×1.0…1.1        60%
  P2 LOO між мережами дасть <×1.02                           75%
  P3 найкращий v — `m` або `m/s`                             50%
  P4 🔒 випадковий v дасть ×1.00                             85%
ГЕЙТ ОГОЛОШЕНО ДО ПРОГОНУ: LOO <×1.02 ⟹ мертво, стадії 2 не буде.
"""
import os, sys, glob, time
import numpy as np
np.seterr(all="ignore")
T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
D, NW = 16, 1024
LAYERS = [8, 10, 12, 14, 15]
NBIN = 12                      # бінів по log λ̃ ⟹ 12 параметрів g

import ast as _ast
src = open("/mnt/win_d/llm/White-Box-P2/estimator_p2_BEST_v11.py").read()
ns = {}
for _n in _ast.parse(src).body:
    if isinstance(_n, _ast.Assign) and len(_n.targets) == 1 and isinstance(_n.targets[0], _ast.Name):
        try: ns[_n.targets[0].id] = _ast.literal_eval(_n.value)
        except Exception: pass
FW = np.array(ns["_FW"], np.float64)
KAPPA, LAM_FEAT, FEAT_N0, KSIG = ns["_KAPPA"], ns["_LAM_FEAT"], ns["_FEAT_N0"], ns["_KSIG"]
log("константи: KAPPA=%.6g LAM_FEAT=%.4g KSIG=%d" % (KAPPA, LAM_FEAT, KSIG))
INV = 0.3989422804014327
def phi(x): return np.exp(-0.5*x*x)*INV
from math import erf as _erf
_ve = np.vectorize(_erf)
def Phi(x): return 0.5*(1.0+_ve(x/np.sqrt(2.0)))
def shape_corr(cpm, s, t, Ph, ph, mn, df):
    n = s.shape[0]; iv = 1.0/s
    R = cpm*np.outer(iv, iv); R = R - np.diag(np.diag(R))
    Rs = R*R; sc = FEAT_N0/n
    R2 = Rs.sum(1)*sc; R3 = (Rs*R).sum(1)*sc; R4 = (Rs*Rs).sum(1)*sc
    A1 = np.abs(R).sum(1)*sc
    lPR = np.log(np.maximum(R2*R2/np.maximum(R4, 1e-30), 1e-30))
    sn = s/s.mean(); tp = t*ph; hp = (t*t-1.0)*ph
    y = (FW[0]+FW[1]*t+FW[2]*Ph+FW[3]*ph+FW[4]*tp+FW[5]*hp+FW[6]*R2
         + FW[7]*R3+FW[8]*R4+FW[9]*A1+FW[10]*lPR+FW[11]*sn+FW[12]*df
         + FW[13]*(t*R2)+FW[14]*(ph*R2)+FW[15]*(hp*R2))
    corr = y*s; g = mn/max(np.sqrt((mn*mn).sum()), 1e-30)
    return corr - (corr*g).sum()*g

def closure_keep(W, want):
    """рекурсія; повертає m,s,t і ЗАМИКАЛЬНУ C на потрібних шарах"""
    m = np.zeros(NW); C = np.eye(NW)
    MS, SS, TS, CC = {}, {}, {}, {}
    for i in range(D):
        w = W[i]; cpm = w.T @ C @ w
        if i > KSIG: cpm = cpm*KAPPA
        s = np.sqrt(np.maximum(np.diag(cpm), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cpm, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        if i in want:
            MS[i] = m.copy(); SS[i] = s.copy(); TS[i] = t.copy()
        if i == D-1: break
        C = cpm*np.outer(Ph, Ph) + 0.5*(cpm*cpm)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
        if (i+1) in want: CC[i+1] = C.copy()          # C ПЕРЕД кроком i+1
    return MS, SS, TS, CC

want = set(LAYERS)
NETS = []
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    wf = "/home/trim/wblogs/w2_%s.npy" % nm
    if not os.path.exists(wf): continue
    W = np.load(wf).astype(np.float64)
    MS, SS, TS, CC = closure_keep(W, want)
    TR = d["m"].astype(np.float64)
    nd = dict(name=nm, MS=MS, SS=SS, TS=TS, CC=CC, TR=TR, W=W)
    NETS.append(nd); log("%-18s готове" % nm)
NN = len(NETS)
log("мереж %d · шарів %d" % (NN, len(LAYERS)))

# ── спектр + цільова форма ──
for nd in NETS:
    nd["EIG"] = {}; nd["E"] = {}
    for L in LAYERS:
        if L not in nd["CC"]: continue
        ev, U = np.linalg.eigh(nd["CC"][L])
        ev = np.maximum(ev, 1e-30)
        nd["EIG"][L] = (ev, U)
        m, T = nd["MS"][L], nd["TR"][L]
        a = (m @ T)/max((m*m).sum(), 1e-300)
        nd["E"][L] = a*m - T
    log("%-18s спектр готовий" % nd["name"])

L0 = [L for L in LAYERS if L in NETS[0]["EIG"]]
print("\nшари з доступною C: %s" % L0)

# 🔒 діагностика: де сидить енергія e у власному базисі замикальної C
print("\n🔒 частка ‖e‖² у топ-k власних напрямках C (ізотропно = k/1024)")
print("шар      k=1     k=4    k=16    k=64   k=256")
for L in L0:
    acc = []
    for nd in NETS:
        ev, U = nd["EIG"][L]; c = (nd["E"][L] @ U)**2; c = c[::-1]      # спадання λ
        c = c/max(c.sum(), 1e-300); cs = np.cumsum(c)
        acc.append([cs[0], cs[3], cs[15], cs[63], cs[255]])
    a = np.mean(acc, 0)
    print("  %-3d  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f" % (L, *a))
print("  ізо   %6.3f  %6.3f  %6.3f  %6.3f  %6.3f" % (1/1024, 4/1024, 16/1024, 64/1024, 256/1024))

# ── дизайн-матриця спектрального оператора ──
def design(nd, L, v):
    ev, U = nd["EIG"][L]
    lt = ev/ev.mean()
    lg = np.log(np.maximum(lt, 1e-30))
    lo, hi = np.percentile(lg, 0.5), np.percentile(lg, 99.5)
    b = np.clip(((lg-lo)/max(hi-lo, 1e-12)*NBIN).astype(int), 0, NBIN-1)
    p = nd["E"][L]*0.0
    coef = U.T @ v                                  # ⟨v,u_k⟩
    X = np.zeros((NW, NBIN))
    for k in range(NBIN):
        msk = (b == k)
        if not msk.any(): continue
        X[:, k] = U[:, msk] @ (coef[msk])
    return X

def vecs(nd, L):
    m, s, t = nd["MS"][L], nd["SS"][L], nd["TS"][L]
    rs = np.random.RandomState(4242+L)
    return {"m": m, "s": s, "m/s": m/np.maximum(s, 1e-30),
            "1": np.ones(NW), "phi(t)": phi(t),
            "🔒 випадковий": rs.standard_normal(NW)}

VN = list(vecs(NETS[0], L0[0]).keys())

def fit_apply(train, test, L, vname, lam_ridge=1e-6):
    XT, yT = [], []
    for nd in train:
        X = design(nd, L, vecs(nd, L)[vname]); y = nd["E"][L]
        sc = 1.0/max(np.linalg.norm(y), 1e-300)     # нормуємо мережі однаково
        XT.append(X*sc); yT.append(y*sc)
    X = np.vstack(XT); y = np.concatenate(yT)
    G = X.T @ X + lam_ridge*np.trace(X.T @ X)/NBIN*np.eye(NBIN)
    g = np.linalg.solve(G, X.T @ y)
    out = []
    for nd in test:
        Xt = design(nd, L, vecs(nd, L)[vname]); yt = nd["E"][L]
        r = yt - Xt @ g
        out.append(np.linalg.norm(yt)/max(np.linalg.norm(r), 1e-300))
    return np.mean(out), g

print("\n── LOO між мережами: ‖e‖/‖e−ê‖ (×1.00 = нуль) ──")
print("шар    " + "  ".join("%-13s" % v for v in VN))
LOO = {}
for L in L0:
    row = []
    for vn in VN:
        rs = [fit_apply([NETS[j] for j in range(NN) if j != i], [NETS[i]], L, vn)[0]
              for i in range(NN)]
        row.append(np.mean(rs))
    LOO[L] = row
    print("  %-3d  " % L + "  ".join("×%-12.4f" % r for r in row))

print("\n── 🔒 IN-SAMPLE СТЕЛЯ (g фітується на ТІЙ САМІЙ мережі) ──")
print("шар    " + "  ".join("%-13s" % v for v in VN))
for L in L0:
    row = []
    for vn in VN:
        rs = [fit_apply([NETS[i]], [NETS[i]], L, vn)[0] for i in range(NN)]
        row.append(np.mean(rs))
    print("  %-3d  " % L + "  ".join("×%-12.4f" % r for r in row))

print("\n── 🔒 СТЕЛЯ ВСЬОГО СПЕКТРАЛЬНОГО КЛАСУ: g на КОЖНЕ власне число (1024 параметри) ──")
for L in L0:
    rs = []
    for nd in NETS:
        ev, U = nd["EIG"][L]; v = nd["MS"][L]
        c = U.T @ v; y = U.T @ nd["E"][L]
        # ê_k = g_k·c_k ⟹ ідеально: g_k = y_k/c_k, залишок = 0 якщо c_k ≠ 0
        bad = np.abs(c) < 1e-12*np.abs(c).max()
        r = np.where(bad, y, 0.0)
        rs.append(np.linalg.norm(y)/max(np.linalg.norm(r), 1e-300))
    print("  шар %-3d  ×%.4g   ⟹ тавтологія, підтверджує що обмежує САМЕ g(λ)" % (L, np.mean(rs)))

print("\n[ %.1fs ] ГОТОВО" % (time.time()-T0))
