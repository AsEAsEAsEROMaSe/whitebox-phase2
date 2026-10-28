"""Дані для вкладки «зони»: простір розбито на 1024 зони, зона = ОДНЕ число
ембедінга, і всередині зони показані всі 16 шарів.

Зона j  ⟹  координата j стану, простежена крізь глибину 0..15.
Канали (що саме за число):
   m          наше замикання
   m_істина   банк N=2²³ (незалежний від нашого коду)
   b          похибка m − істина
   b/s        безрозмірна похибка
   s          σ передактивації (замикання)
   t          m_p/s — насиченість воріт
   √dd        σ активації після ReLU
   emp        емпіричне середнє 6144 семплів
   h[0]       стан ОДНОГО семпла
   ‖w_j‖      норма вхідних ваг нейрона (статика шару)

🚨 ЗАСТЕРЕЖЕННЯ, ЯКЕ МУСИТЬ ЇХАТИ РАЗОМ ІЗ КАРТИНКОЮ: індекс нейрона МІЖ
шарами не значить нічого (corr стовпців W ≈ 0, зміряно С115) ⟹ «зона j на
шарі L» і «зона j на шарі L+1» — РІЗНІ нейрони. Тому крім сирого порядку
готуємо ПЕРЕСТАНОВКИ за рангом, де зона = РАНГ, і шари стають порівнянними.
"""
import os, sys, glob, time
import numpy as np

T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

D, NW = 16, 1024
WHICH = sys.argv[1] if len(sys.argv) > 1 else "david-wilson"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 6144

# ─────────── константи оцінювача ───────────
import ast as _ast
src = open("/mnt/win_d/llm/White-Box-P2/estimator_p2_BEST_v11.py").read()
ns = {}
for _n in _ast.parse(src).body:
    if isinstance(_n, _ast.Assign) and len(_n.targets) == 1 and isinstance(_n.targets[0], _ast.Name):
        try: ns[_n.targets[0].id] = _ast.literal_eval(_n.value)
        except Exception: pass
FW = np.array(ns["_FW"], dtype=np.float64)
KAPPA, LAM_FEAT, FEAT_N0, KSIG = ns["_KAPPA"], ns["_LAM_FEAT"], ns["_FEAT_N0"], ns["_KSIG"]
log("константи: KAPPA=%.6g LAM_FEAT=%.4g KSIG=%d FW[%d]" % (KAPPA, LAM_FEAT, KSIG, len(FW)))

INV_SQRT_2PI = 0.3989422804014327
def phi(x): return np.exp(-0.5*x*x)*INV_SQRT_2PI
from math import erf as _erf
_verf = np.vectorize(_erf)
def Phi(x): return 0.5*(1.0+_verf(x/np.sqrt(2.0)))

def shape_corr(cpm, s, t, Ph, ph, mn, depth_frac):
    n = s.shape[0]; inv = 1.0/s
    R = cpm*np.outer(inv, inv); R = R - np.diag(np.diag(R))
    Rs = R*R; sc = FEAT_N0/n
    R2 = Rs.sum(1)*sc; R3 = (Rs*R).sum(1)*sc; R4 = (Rs*Rs).sum(1)*sc
    A1 = np.abs(R).sum(1)*sc
    lPR = np.log(np.maximum(R2*R2/np.maximum(R4, 1e-30), 1e-30))
    sn = s/s.mean(); tp = t*ph; hp = (t*t-1.0)*ph
    y = (FW[0]+FW[1]*t+FW[2]*Ph+FW[3]*ph+FW[4]*tp+FW[5]*hp+FW[6]*R2
         + FW[7]*R3+FW[8]*R4+FW[9]*A1+FW[10]*lPR+FW[11]*sn+FW[12]*depth_frac
         + FW[13]*(t*R2)+FW[14]*(ph*R2)+FW[15]*(hp*R2))
    corr = y*s
    g = mn/max(np.sqrt((mn*mn).sum()), 1e-30)
    return corr - (corr*g).sum()*g

def closure(W):
    m = np.zeros(NW); C = np.eye(NW)
    MS, SS, TS, VV = [], [], [], []
    for i in range(D):
        w = W[i]
        cpm = w.T @ C @ w
        if i > KSIG: cpm = cpm*KAPPA
        s = np.sqrt(np.maximum(np.diag(cpm), 1e-20))
        mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0:
            m = m - LAM_FEAT*shape_corr(cpm, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy()); SS.append(s.copy()); TS.append(t.copy()); VV.append(dd.copy())
        if i == D-1: break
        C = cpm*np.outer(Ph, Ph) + 0.5*(cpm*cpm)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS, SS, TS, VV

# ─────────── дані ───────────
W = np.load("/home/trim/wblogs/w2_%s.npy" % WHICH).astype(np.float64)
bank = None
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz")):
    d = np.load(f, allow_pickle=True)
    if str(d["name"]) == WHICH: bank = d; break
if bank is None: raise SystemExit("нема банку істини для %s" % WHICH)
TR = bank["m"].astype(np.float64)                 # (16,1024) N=2^23
log("мережа %s · банк N=%d" % (WHICH, int(bank["n"])))

MS, SS, TS, VV = closure(W)
log("замикання готове")

# емпіричне середнє і один семпл — тим самим сідом, що й решта вкладок
rs = np.random.RandomState(2026)
x = rs.standard_normal((NS, NW)).astype(np.float32)
EMP = np.zeros((D, NW), np.float64); H0 = np.zeros((D, NW), np.float64)
for i in range(D):
    x = np.maximum(x @ W[i].astype(np.float32), 0.0)
    EMP[i] = x.mean(0); H0[i] = x[0]
log("семпли готові (%d)" % NS)

WN = np.stack([np.linalg.norm(W[i], axis=0) for i in range(D)])   # ‖W_l[:,j]‖

CH, NAMES, SIGNED = [], [], []
def add(nm, arr, signed):
    CH.append(np.asarray(arr, np.float32)); NAMES.append(nm); SIGNED.append(signed)

M = np.stack(MS); S = np.stack(SS); T = np.stack(TS); V = np.stack(VV)
add("m — наше замикання",        M,                       False)
add("m — ІСТИНА (банк 2²³)",     TR,                      False)
add("b = m − істина",            M - TR,                  True)
add("b/s — безрозмірна похибка", (M - TR)/np.maximum(S, 1e-30), True)
add("s — σ передактивації",      S,                       False)
add("t = m_p/s — ворота",        T,                       True)
add("√dd — σ активації",         np.sqrt(np.maximum(V, 0)), False)
add("emp — середнє 6144",        EMP,                     False)
add("emp − істина (шум МК)",     EMP - TR,                True)
add("h[0] — один семпл",         H0,                      False)
add("‖w_j‖ — норма ваг",         WN,                      False)
CHAN = np.stack(CH)                                        # (NCH, 16, 1024)

# перестановки: ранг у СВОЄМУ шарі (за |m|) і ранг за шаром 15
ORD_SELF = np.argsort(-np.abs(M), axis=1).astype(np.int32)          # (16,1024)
ORD_L15  = np.tile(np.argsort(-np.abs(M[D-1])).astype(np.int32), (D, 1))
ORD_T    = np.argsort(-T, axis=1).astype(np.int32)

np.savez_compressed("/home/trim/wblogs/viz_zone.npz", name=WHICH,
                    CHAN=CHAN, NAMES=np.array(NAMES), SIGNED=np.array(SIGNED),
                    ORD_SELF=ORD_SELF, ORD_L15=ORD_L15, ORD_T=ORD_T, ns=NS)
log("записано /home/trim/wblogs/viz_zone.npz · каналів %d" % len(NAMES))

# ─── пломби ───
rel = np.array([np.linalg.norm(M[i]-TR[i])/np.linalg.norm(TR[i]) for i in range(D)])
a = np.array([(M[i]@TR[i])/max((M[i]*M[i]).sum(), 1e-300) for i in range(D)])
cosr = np.array([np.linalg.norm(a[i]*M[i]-TR[i])/np.linalg.norm(TR[i]) for i in range(D)])
emr = np.array([np.linalg.norm(EMP[i]-TR[i])/np.linalg.norm(TR[i]) for i in range(D)])
print("\nшар   сира%%    косинусна%%   МК проти істини%%   медіана|t|")
for i in (0, 1, 4, 8, 12, 15):
    print("  %-3d %7.4f   %8.4f      %9.4f        %6.3f"
          % (i, 100*rel[i], 100*cosr[i], 100*emr[i], np.median(np.abs(T[i]))))
print("\n🔒 косинусна на ш15 = %.4f%%  [корпус: 0.1202%%]" % (100*cosr[D-1]))
print("🔒 шум МК на ш15 = %.4f%% при теорії 1/√%d ≈ %.4f%% розкиду"
      % (100*emr[D-1], NS, 100/np.sqrt(NS)))
