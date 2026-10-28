"""С115-х: СТРУКТУРА ГАСІННЯ. Які доданки похибки одне одного вбивають (ідея ЮЗЕРА).

Рекурсія похибки замкнена (С70, нев'язка 5e-5):
    b_l = Φ_l ⊙ (b_{l-1} @ W_l) + r_l        ⟹  r_l = СВІЖЕ джерело на шарі l
Переносимо кожне джерело до виходу лінеаризованим оператором (Φ у нас уже є):
    T_{l→15}(r) = ((r @ W_{l+1})⊙Φ_{l+1}) @ W_{l+2} ⊙ Φ_{l+2} …
⟹  b_15 = Σ_l T_{l→15}(r_l)   і матриця Грама цих векторів каже, ЩО гаситься.

🔒 ПЛОМБА ТОТОЖНОСТІ: ‖b_15 − Σ T(r_l)‖/‖b_15‖ мусить бути малою
🔒 ПЛОМБА ШАРУ 0: замикання там точне ⟹ r_0 ≈ шум істини
Ключове число: ‖Σ‖² / Σ‖T(r_l)‖²  — менше 1 означає ГАСІННЯ, і скільки.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 тотожність замикається < 5%                                   75%
  P2 сильне гасіння: ‖Σ‖²/Σ‖·‖² = 0.2..0.5                        70%
  P3 гасіння сидить у СУСІДНІХ парах шарів (від'ємна білядіагональ) 55%
  P4 є пара з кореляцією < −0.5                                    45%
"""
import os, sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np
D, NW = 16, 1024
T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def closure(W):
    m = np.zeros(NW); C = np.eye(NW); MS = []; PH = []
    for i in range(D):
        w = W[i]; q = w.T @ C @ w
        if i > KSIG: q = q*KAPPA
        s = np.sqrt(np.maximum(np.diag(q), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t); m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(q, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy()); PH.append(Ph.copy())
        if i == D-1: break
        C = q*np.outer(Ph, Ph) + 0.5*(q*q)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS, PH

GR = []; SEAL = []; RAT = []; R0 = []; NRM = []; ATT = []; LOC = []; l0=0
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:6]:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    wf = "/home/trim/wblogs/w2_%s.npy" % nm
    if not os.path.exists(wf): continue
    T = np.asarray(d["m"]); mh = np.asarray(d["m_half"])
    W = np.load(wf).astype(np.float64)
    MS, PH = closure(W)
    b = [MS[l] - T[l] for l in range(D)]
    r = [b[0].copy()]                                   # джерело шару 0
    for l in range(1, D):
        r.append(b[l] - PH[l]*(b[l-1] @ W[l]))
    Tr = []
    for l in range(D):
        v = r[l].copy()
        for k in range(l+1, D): v = (v @ W[k])*PH[k]
        Tr.append(v)
    Tr = np.array(Tr)
    S = Tr.sum(0)
    SEAL.append(np.linalg.norm(S - b[15])/np.linalg.norm(b[15]))
    RAT.append((S@S)/ (Tr*Tr).sum())
    R0.append(np.linalg.norm(r[0])/np.linalg.norm(T[0]))
    n = np.linalg.norm(Tr, axis=1); Nn = np.maximum(n, 1e-300)
    GR.append((Tr @ Tr.T)/np.outer(Nn, Nn))
    NRM.append(n/np.linalg.norm(T[15]))                    # ‖T(r_l)‖ у частках ‖m‖
    ATT.append((Tr @ S)/max(S@S, 1e-300))                  # внесок у фінальну похибку
    LOC.append(np.linalg.norm(r[l0], axis=0) if False else
               np.array([np.linalg.norm(r[k])/np.linalg.norm(T[k]) for k in range(D)]))
    log("%-18s 🔒тотожність %.4f · ‖Σ‖²/Σ‖·‖² %.4f · 🔒r₀/‖T₀‖ %.2e"
        % (nm, SEAL[-1], RAT[-1], R0[-1]))

G = np.median(np.array(GR), 0)
print("\n🔒 тотожність (медіана): %.5f    🔒 r₀ (шар 0 точний): %.2e   [шум цілі ~1e-4]" 
      % (np.median(SEAL), np.median(R0)))
print("🔑 ‖Σ‖² / Σ‖T(r_l)‖² = %.4f  ⟹ гасіння знімає %.1f%% енергії"
      % (np.median(RAT), 100*(1-np.median(RAT))))
print("\nкореляції перенесених джерел T(r_l) ↔ T(r_k):")
print("     " + " ".join("%5d" % k for k in range(D)))
for l in range(D):
    print("%3d  " % l + " ".join("%+5.2f" % G[l,k] for k in range(D)))
NR = np.median(np.array(NRM),0); AT = np.median(np.array(ATT),0); LC = np.median(np.array(LOC),0)
print("\nшар   локальне джерело r_l   перенесене ‖T(r_l)‖   ВНЕСОК у фінальну похибку")
print("      (у частках ‖m_l‖)       (у частках ‖m_15‖)      (сума = 1.000)")
for l in range(D):
    print("%3d        %.5f                %.5f              %+.4f" % (l, LC[l], NR[l], AT[l]))
print("сума внесків %.4f   ·   сума ‖T(r_l)‖ %.5f проти ‖b_15‖ %.5f"
      % (AT.sum(), NR.sum(), np.sqrt(np.median(RAT))*np.sqrt((NR*NR).sum())))
off = G[np.triu_indices(D,1)]
print("\nпоза діагоналлю: медіана %+.3f · мін %+.3f · від'ємних %.0f%% · |>0.5| %d пар"
      % (np.median(off), off.min(), 100*(off<0).mean(), int((np.abs(off)>0.5).sum())))
ad = np.array([G[l,l+1] for l in range(D-1)])
print("сусідні пари (l,l+1): " + " ".join("%+.2f" % v for v in ad))
log("ГОТОВО")
