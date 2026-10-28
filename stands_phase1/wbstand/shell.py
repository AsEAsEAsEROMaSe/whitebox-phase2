"""С115-ч: ЧИ ІНФОРМАТИВНИЙ РАДІУС СЕМПЛА? (питання ЮЗЕРА про «стабільну відстань»)

u_s = h_s − T  (T = істинний центр, банк N=2²³); r=‖u‖, радіальна p=⟨u,T̂⟩, танг. q.
Стратифікуємо за r на 8 оболонок і питаємо, чи якась із них систематично
«точніша» — тобто чи її часткове середнє ближче до істини.

🔒 ГАУСОВИЙ КОНТРОЛЬ визначає, що таке «дивне»: та сама статистика для
   синтетичної хмари N(0, Cov(h)) з ТІЄЮ Ж коваріацією. Різниця = справжня
   структура; збіг = все пояснює друга моментна.
🔒 стабільність: ранг радіуса шар8 ↔ шар15 (у мережі) і мережа↔мережа (той самий z)
📐 однорідність: h_l(z) = ‖z‖·h_l(ẑ) ТОЧНО ⟹ радіус = ‖z‖ × кутовий множник;
   ‖z‖ має розкид лише 1/√2048 = 2.2% ⟹ універсальна частка мусить бути малою

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 розподіл r збігається з гаусовим контролем (χ-подібний)        70%
  P2 якась оболонка систематично точніша ПОНАД гаусовий контроль    40%
  P3 крос-мережевий ранг радіуса малий (0.1..0.3)                   65%
  P4 крос-шаровий ранг радіуса великий (>0.7)                       70%
  P5 корел(радіус, внесок у похибку) ≈ гаусова опора                60%
"""
import os, sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp
D, NW, NSMP, NSH = 16, 1024, 8192, 8
T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

BK = {}
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz")):
    d = np.load(f, allow_pickle=True); BK[str(d["name"])] = np.asarray(d["m"])
FILES = [f for f in sorted(glob.glob("/home/trim/wblogs/w2_*.npy"))
         if os.path.basename(f)[3:-4] in BK][:4]

def stats(U, T, ehat):
    """U (N,1024) зміщення від центру; повертає профілі по оболонках"""
    that = T/cp.linalg.norm(T)
    r = cp.linalg.norm(U, axis=1); p = U @ that
    q = cp.sqrt(cp.maximum(r*r - p*p, 0.0))
    c = U @ ehat                                    # внесок у напрямок похибки МК
    qs = cp.percentile(r, cp.linspace(0, 100, NSH+1))
    out = []
    for k in range(NSH):
        msk = (r >= qs[k]) & (r <= qs[k+1] if k == NSH-1 else r < qs[k+1])
        if int(msk.sum()) == 0: out.append((0.,0.,0.)); continue
        out.append((float(p[msk].mean()/r.mean()), float(q[msk].mean()/r.mean()),
                    float(c[msk].mean()/r.mean())))
    return np.array(out), float(cp.corrcoef(cp.stack([r, c]))[0,1]), cp.asnumpy(r)

RES = []; RANKL = []; RS = {}
for f in FILES:
    nm = os.path.basename(f)[3:-4]; T = BK[nm]
    W = np.load(f); Wg = [cp.asarray(W[i]) for i in range(D)]
    z = cp.random.RandomState(seed=2026).standard_normal((NSMP, NW), dtype=cp.float32)
    x = z; H = {}
    for i in range(D):
        x = cp.maximum(x @ Wg[i], 0.0)
        if i in (8, 15): H[i] = x.astype(cp.float64)
    Tg = cp.asarray(T[15])
    U = H[15] - Tg
    e = U.mean(0); ehat = e/cp.linalg.norm(e)
    pr, cr, rr = stats(U, Tg, ehat)
    # 🔒 гаусовий контроль: та сама Cov, той самий центр-зсув
    Cv = cp.cov(U.T)
    L = cp.linalg.cholesky(Cv + cp.eye(NW, dtype=cp.float64)*1e-9*float(cp.trace(Cv))/NW)
    g = cp.random.RandomState(seed=71).standard_normal((NSMP, NW), dtype=cp.float64)
    Ug = (g @ L.T) + U.mean(0)
    eg = Ug.mean(0); pg, cg, rg = stats(Ug, Tg, eg/cp.linalg.norm(eg))
    RES.append((pr, pg, cr, cg))
    r8 = cp.linalg.norm(H[8] - cp.asarray(T[8]), axis=1)
    a = cp.argsort(cp.argsort(r8)).astype(cp.float64); b = cp.argsort(cp.argsort(cp.asarray(rr))).astype(cp.float64)
    RANKL.append(float(cp.corrcoef(cp.stack([a, b]))[0,1]))
    RS[nm] = rr
    log("%-18s корел(r, внесок) %.4f  🔒гаус %.4f  · ранг ш8↔ш15 %.3f"
        % (nm, cr, cg, RANKL[-1]))
    del Wg, H, U, Ug, g, L; cp.get_default_memory_pool().free_all_blocks()

P = np.median(np.array([r[0] for r in RES]), 0); G = np.median(np.array([r[1] for r in RES]), 0)
print("\nоболонка   радіальна p/r̄        тангенційна q/r̄       внесок у похибку c/r̄")
print("           наше    🔒гаус       наше    🔒гаус         наше    🔒гаус")
for k in range(NSH):
    print("  %d      %+.4f  %+.4f     %+.4f  %+.4f      %+.5f  %+.5f"
          % (k+1, P[k,0], G[k,0], P[k,1], G[k,1], P[k,2], G[k,2]))
print("\nкорел(радіус, внесок у похибку): наше %.4f · 🔒гаус %.4f · різниця %+.4f"
      % (np.median([r[2] for r in RES]), np.median([r[3] for r in RES]),
         np.median([r[2] for r in RES]) - np.median([r[3] for r in RES])))
print("🔒 стабільність рангу радіуса шар8↔шар15: %.3f" % np.median(RANKL))
ks = list(RS)
if len(ks) >= 2:
    cc = []
    for i in range(len(ks)):
        for j in range(i+1, len(ks)):
            cc.append(float(np.corrcoef(np.argsort(np.argsort(RS[ks[i]])),
                                        np.argsort(np.argsort(RS[ks[j]])))[0,1]))
    print("🔒 крос-МЕРЕЖЕВИЙ ранг радіуса (той самий z): %.3f  [однорідність дає ‖z‖, розкид 2.2%%]"
          % np.median(cc))
log("ГОТОВО")
