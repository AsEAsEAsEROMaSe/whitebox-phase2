"""С116-A: ЧИ Є СПІЛЬНА СТРУКТУРА ПОХИБКИ МІЖ РІЗНИМИ МЕРЕЖАМИ —
у координатах, де питання КОРЕКТНЕ.

🚨 ЧОМУ ЦЕ НЕ ПОВТОР СЕМИ ЗАКРИТИХ ВИМІРІВ. Усі вони порівнювали ВЕКТОРИ
(чужа W 0/400 · чужа форма b −0.004…+0.009 · чужа Δ −0.002 · чужа b оптимум
c=0 · крос-ранг радіуса 0.029). Нейрон j мережі A і нейрон j мережі B — різні
нейрони ⟹ ці нулі були НЕМИНУЧІ за побудовою і нічого не закривали.
Коректне питання: чи спільний ПРОФІЛЬ, тобто ФУНКЦІЯ від обчислюваних
координат нейрона. Функція переноситься, вектор — ні.

ЦІЛЬ = косинусна похибка (масштаб уже знято оптимальним a) ⟹ e = a·m − T ⊥ m
за побудовою. Поправку будуємо у формі оцінювача: c = π(q)·s, потім m−λc.

КООРДИНАТИ (усі обчислювані БЕЗ істини):
  q_t = ранг t у шарі / n      насиченість воріт
  q_s = ранг s у шарі / n      масштаб передактивації
  2D  = 16×16 бінів (q_t, q_s)

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (щоб не тлумачити заднім числом):
  P1 профіль π НЕНУЛЬОВИЙ і корелює між мережами            60%
  P2 але LOO-віднімання дасть ×1.00…1.02 (FW уже бере лінійну частину) 70%
  P3 2D-профіль не кращий за 1D по q_t                      55%
  P4 контроль «перемішаний y» дасть рівно нуль              90%
ГЕЙТ ОГОЛОШЕНО ДО ПРОГОНУ: <×1.02 мертво · 1.02–1.05 гранично · >1.05 сабміт
"""
import os, sys, glob, time
import numpy as np
np.seterr(all="ignore")

T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
D, NW = 16, 1024
NB1 = 32          # бінів у 1D
NB2 = 16          # бінів на вісь у 2D

# ─────────── константи оцінювача ───────────
import ast as _ast
src = open("/mnt/win_d/llm/White-Box-P2/estimator_p2_BEST_v11.py").read()
ns = {}
for _n in _ast.parse(src).body:
    if isinstance(_n, _ast.Assign) and len(_n.targets) == 1 and isinstance(_n.targets[0], _ast.Name):
        try: ns[_n.targets[0].id] = _ast.literal_eval(_n.value)
        except Exception: pass
FW = np.array(ns["_FW"], np.float64)
KAPPA, LAM_FEAT, FEAT_N0, KSIG = ns["_KAPPA"], ns["_LAM_FEAT"], ns["_FEAT_N0"], ns["_KSIG"]
log("константи: KAPPA=%.6g LAM_FEAT=%.4g KSIG=%d FW[%d]" % (KAPPA, LAM_FEAT, KSIG, len(FW)))

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
    corr = y*s
    g = mn/max(np.sqrt((mn*mn).sum()), 1e-30)
    return corr - (corr*g).sum()*g

def closure(W):
    m = np.zeros(NW); C = np.eye(NW); MS, SS, TS = [], [], []
    for i in range(D):
        w = W[i]; cpm = w.T @ C @ w
        if i > KSIG: cpm = cpm*KAPPA
        s = np.sqrt(np.maximum(np.diag(cpm), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cpm, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy()); SS.append(s.copy()); TS.append(t.copy())
        if i == D-1: break
        C = cpm*np.outer(Ph, Ph) + 0.5*(cpm*cpm)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return np.stack(MS), np.stack(SS), np.stack(TS)

def cerr(m, T):
    """косинусна (безмасштабна) похибка і залишок e = a·m − T"""
    a = (m @ T)/max((m*m).sum(), 1e-300)
    e = a*m - T
    return np.linalg.norm(e)/np.linalg.norm(T), e

# ─────────── дані ───────────
NETS = []
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    wf = "/home/trim/wblogs/w2_%s.npy" % nm
    if not os.path.exists(wf): continue
    W = np.load(wf).astype(np.float64)
    M, S, Tt = closure(W)
    TR = d["m"].astype(np.float64); NZ = d["noise"].astype(np.float64)
    NETS.append(dict(name=nm, M=M, S=S, T=Tt, TR=TR, NZ=NZ))
    log("%-18s замикання готове · шум банку ш15 %.3e" % (nm, NZ[15].mean()))
NN = len(NETS)
log("мереж: %d" % NN)

# базова похибка
for nd in NETS:
    nd["base"] = np.array([cerr(nd["M"][L], nd["TR"][L])[0] for L in range(D)])
    nd["E"] = np.stack([cerr(nd["M"][L], nd["TR"][L])[1] for L in range(D)])
base15 = np.array([nd["base"][D-1] for nd in NETS])
print("\nбаза (косинусна на ш15): " + " ".join("%.4f%%" % (100*b) for b in base15))
print("  середнє %.4f%%   [корпус 0.1202%%]" % (100*base15.mean()))

# ─────────── профілі ───────────
def ranks(v):
    o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
    return r/(len(v)-1.0)

def prof1(nd, key, nb=NB1):
    """π[L, bin] — середнє y = e/s у бінах за рангом координати"""
    P = np.zeros((D, nb)); Cn = np.zeros((D, nb))
    for L in range(D):
        q = ranks(nd[key][L]); y = nd["E"][L]/np.maximum(nd["S"][L], 1e-30)
        b = np.minimum((q*nb).astype(int), nb-1)
        np.add.at(P[L], b, y); np.add.at(Cn[L], b, 1.0)
    return P/np.maximum(Cn, 1.0)

def prof2(nd, nb=NB2):
    P = np.zeros((D, nb*nb)); Cn = np.zeros((D, nb*nb))
    for L in range(D):
        qt = ranks(nd["T"][L]); qs = ranks(nd["S"][L])
        y = nd["E"][L]/np.maximum(nd["S"][L], 1e-30)
        b = np.minimum((qt*nb).astype(int), nb-1)*nb + np.minimum((qs*nb).astype(int), nb-1)
        np.add.at(P[L], b, y); np.add.at(Cn[L], b, 1.0)
    return P/np.maximum(Cn, 1.0)

for nd in NETS:
    nd["p_t"] = prof1(nd, "T"); nd["p_s"] = prof1(nd, "S"); nd["p_2"] = prof2(nd)

# ─── ПЛОМБА рівня шуму банку в y ───
sig = np.mean([np.abs(nd["E"][15]/nd["S"][15]).mean() for nd in NETS])
nsy = np.mean([(nd["NZ"][15]/nd["S"][15]).mean() for nd in NETS])
print("\n🔒 |y| на ш15 = %.3e · шум банку в тих же одиницях %.3e ⟹ ×%.1f" % (sig, nsy, sig/nsy))

# ─────────── чи корелюють профілі МІЖ мережами ───────────
def xcorr(key, tag):
    print("\n── крос-мережева кореляція профілю %s ──" % tag)
    print("шар    сер.corr пар    🔒перемішаний    пар")
    out = []
    for L in (1, 4, 8, 12, 15):
        cs, cshuf = [], []
        for i in range(NN):
            for j in range(i+1, NN):
                a, b = NETS[i][key][L], NETS[j][key][L]
                if a.std() < 1e-15 or b.std() < 1e-15: continue
                cs.append(np.corrcoef(a, b)[0, 1])
                bp = np.random.RandomState(1000+L*100+i*10+j).permutation(b)
                cshuf.append(np.corrcoef(a, bp)[0, 1])
        print("  %-3d   %+.4f          %+.4f        %d" % (L, np.mean(cs), np.mean(cshuf), len(cs)))
        out.append(np.mean(cs))
    return np.array(out)

c_t = xcorr("p_t", "за рангом t (32 біни)")
c_s = xcorr("p_s", "за рангом s (32 біни)")
c_2 = xcorr("p_2", "2D 16×16 (t,s)")

# ─────────── LOO: фіт профілю на N−1, віднімання на решті ───────────
def apply_prof(nd, P, key2d, lam, nb):
    """m' = m − lam·π(q)·s ; повертає косинусну похибку по шарах"""
    err = np.zeros(D)
    for L in range(D):
        if key2d:
            qt = ranks(nd["T"][L]); qs = ranks(nd["S"][L])
            b = np.minimum((qt*nb).astype(int), nb-1)*nb + np.minimum((qs*nb).astype(int), nb-1)
        else:
            q = ranks(nd[key2d if key2d else "T"][L]) if False else ranks(nd[KEYMAP][L])
            b = np.minimum((q*nb).astype(int), nb-1)
        c = P[L][b]*nd["S"][L]
        err[L] = cerr(nd["M"][L] - lam*c, nd["TR"][L])[0]
    return err

LAMS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
def loo(key, KEY_COORD, is2d, nb, tag):
    global KEYMAP
    KEYMAP = KEY_COORD
    print("\n── LOO %s ──" % tag)
    print("  λ      ×база (ш15)   ×база (сер 8-15)   знаковий")
    best = None
    for lam in LAMS:
        rat15, ratD, sgn = [], [], 0
        for i, nd in enumerate(NETS):
            Pm = np.mean([NETS[j][key] for j in range(NN) if j != i], axis=0)
            e = apply_prof(nd, Pm, is2d, lam, nb)
            r15 = nd["base"][D-1]/e[D-1]
            rD = nd["base"][8:].mean()/e[8:].mean()
            rat15.append(r15); ratD.append(rD); sgn += (r15 > 1.0)
        m15, mD = np.mean(rat15), np.mean(ratD)
        print("  %-5.2f   ×%.4f       ×%.4f          %d/%d" % (lam, m15, mD, sgn, NN))
        if best is None or m15 > best[1]: best = (lam, m15, mD, sgn)
    print("  ⟹ найкраще λ=%.2f  ×%.4f  (%d/%d)" % (best[0], best[1], best[3], NN))
    return best

b_t = loo("p_t", "T", False, NB1, "профіль за рангом t")
b_s = loo("p_s", "S", False, NB1, "профіль за рангом s")
b_2 = loo("p_2", "T", True,  NB2, "2D-профіль (t,s)")

# ─────────── КОНТРОЛІ ───────────
print("\n── 🔒 КОНТРОЛІ ──")
KEYMAP = "T"
# 1. перемішаний y всередині шару ⟹ профіль має бути шумом
for nd in NETS:
    P = np.zeros((D, NB1)); Cn = np.zeros((D, NB1))
    for L in range(D):
        q = ranks(nd["T"][L])
        y = nd["E"][L]/np.maximum(nd["S"][L], 1e-30)
        y = np.random.RandomState(77+L).permutation(y)
        b = np.minimum((q*NB1).astype(int), NB1-1)
        np.add.at(P[L], b, y); np.add.at(Cn[L], b, 1.0)
    nd["p_sh"] = P/np.maximum(Cn, 1.0)
r = []
for i, nd in enumerate(NETS):
    Pm = np.mean([NETS[j]["p_sh"] for j in range(NN) if j != i], axis=0)
    e = apply_prof(nd, Pm, False, b_t[0], NB1); r.append(nd["base"][D-1]/e[D-1])
print("  перемішаний y, λ=%.2f:            ×%.4f  (мусить ≈1.0000)" % (b_t[0], np.mean(r)))

# 2. стеля: віднімаємо ВЛАСНИЙ профіль тієї ж мережі (in-sample)
r = []
for nd in NETS:
    e = apply_prof(nd, nd["p_t"], False, 1.0, NB1); r.append(nd["base"][D-1]/e[D-1])
print("  ВЛАСНИЙ профіль (in-sample):      ×%.4f  ⟹ стеля бінінгу" % np.mean(r))

# 3. стеля класу: віднімаємо ІСТИННУ форму
r = []
for nd in NETS:
    e = np.array([cerr(nd["M"][L] - nd["E"][L], nd["TR"][L])[0] for L in range(D)])
    r.append(nd["base"][D-1]/max(e[D-1], 1e-30))
print("  ІСТИННА форма (тавтологія):       ×%.3e" % np.mean(r))

print("\n[ %.1fs ] ГОТОВО" % (time.time()-T0))
