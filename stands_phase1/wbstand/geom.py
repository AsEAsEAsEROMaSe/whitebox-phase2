"""С115-ф: ГЕОМЕТРІЯ ПОХИБОК ВІДНОСНО ІДЕАЛЬНОГО ЦЕНТРУ (запит ЮЗЕРА).

ІДЕАЛЬНИЙ ЦЕНТР T = істинне m із банку (N=2²³). Дві гілки похибки:
   e = m_замикання − T      детермінована
   v = m_МК(n=6144) − T     вибіркова
Міряємо радіальну/тангенційну частини кожної відносно T̂, кути між ними,
і рух окремих семплів відносно T.

🔑 КУТ МІЖ e І v — ПРЯМА ПЕРЕВІРКА ПРИПУЩЕННЯ БЛЕНДУ: гармонійний бленд
   1/MSE = 1/V + 1/E² вважає гілки НЕЗАЛЕЖНИМИ. Якщо їхні вектори не
   ортогональні — вага перекошена. (С115-к так і завалила регресійну CV.)

🔒 ізотропна опора: два випадкові вектори в R^1024 дають 90° ± 1/√1024 = 1.8°
🔒 шум цілі: половини банку A vs B

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 e має ВЕЛИКУ радіальну частину (кут до T ~30..45°)        85%
  P2 v майже перпендикулярний до T (88..92°)                   75%
  P3 кут(e, v) ≈ 90° ⟹ припущення бленду тримається            70%
  P4 радіальна частина e РОСТЕ з глибиною                      65%
  P5 зміщення окремого семпла переважно ТАНГЕНЦІЙНЕ            70%
"""
import os, sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp
D, NW = 16, 1024
NMC, REP = 6144, 8
LAYERS = [1, 4, 8, 12, 14, 15]
T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def cl_np(W):
    m = np.zeros(NW); C = np.eye(NW); MS = []
    for i in range(D):
        w = W[i]; q = w.T @ C @ w
        if i > KSIG: q = q*KAPPA
        s = np.sqrt(np.maximum(np.diag(q), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t); m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(q, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy())
        if i == D-1: break
        C = q*np.outer(Ph, Ph) + 0.5*(q*q)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS

def ang(a, b):
    c = float(a @ b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-300)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))
def split(x, that):
    p = float(x @ that); return p, float(np.linalg.norm(x - p*that))

R = {L: {k: [] for k in ("aeT","avT","aev","fe","fv","fs","asT","noise")} for L in LAYERS}
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:6]:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    wf = "/home/trim/wblogs/w2_%s.npy" % nm
    if not os.path.exists(wf): continue
    mh = np.asarray(d["m_half"]); T = np.asarray(d["m"])          # повний банк = центр
    W = np.load(wf); Wg = [cp.asarray(W[i]) for i in range(D)]
    MS = cl_np(W.astype(np.float64))
    for L in LAYERS:
        that = T[L]/np.linalg.norm(T[L])
        e = MS[L] - T[L]
        pe, oe = split(e, that)
        R[L]["aeT"].append(ang(e, T[L])); R[L]["fe"].append(abs(pe)/np.linalg.norm(e))
        R[L]["noise"].append(np.linalg.norm(mh[L,0]-mh[L,1])/2.0/np.linalg.norm(T[L]))
    for r in range(REP):
        z = cp.random.RandomState(seed=9100+37*r).standard_normal((NMC, NW), dtype=cp.float32)
        x = z; smp = {}
        for i in range(D):
            x = cp.maximum(x @ Wg[i], 0.0)
            if i in LAYERS: smp[i] = x.copy()
        for L in LAYERS:
            that = T[L]/np.linalg.norm(T[L])
            S = cp.asnumpy(smp[L].mean(0)).astype(np.float64)
            v = S - T[L]; pv, ov = split(v, that)
            R[L]["avT"].append(ang(v, T[L])); R[L]["fv"].append(abs(pv)/np.linalg.norm(v))
            R[L]["aev"].append(ang(MS[L]-T[L], v))
            if r == 0:                                    # рух ОКРЕМИХ семплів
                H = cp.asnumpy(smp[L][:2048]).astype(np.float64)
                U = H - T[L]; th = cp.asnumpy(cp.asarray(that))
                pp = U @ th; oo = np.sqrt(np.maximum((U*U).sum(1) - pp*pp, 0.0))
                R[L]["fs"].append(float(np.median(np.abs(pp)/np.maximum(np.sqrt((U*U).sum(1)),1e-30))))
                R[L]["asT"].append(float(np.median(np.degrees(np.arctan2(oo, pp)))))
        del smp, z, x
    log("%-18s готово" % nm)
    del Wg; cp.get_default_memory_pool().free_all_blocks()

md = lambda L,k: float(np.median(R[L][k]))
print("\n🔒 ізотропна опора: два випадкові вектори в R^1024 ⟹ 90.0° ± 1.8°")
print("\nшар   кут(e,T)  радіальна e | кут(v,T)  радіальна v | 🔑кут(e,v) | семпл: кут(u,T) ради.частка | 🔒шум цілі")
for L in LAYERS:
    print("%3d    %6.1f°     %.3f    |  %6.1f°    %.3f    |  %6.1f°   |   %6.1f°      %.3f      |  %.5f"
          % (L, md(L,"aeT"), md(L,"fe"), md(L,"avT"), md(L,"fv"), md(L,"aev"),
             md(L,"asT"), md(L,"fs"), md(L,"noise")))
log("ГОТОВО")
