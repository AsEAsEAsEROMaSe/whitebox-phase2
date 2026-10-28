"""С115-с: НАСКІЛЬКИ СТАН ЗМІЩЕНИЙ ВІД НУЛЯ НА КОЖНОМУ ШАРІ (питання ЮЗЕРА).

`t_j = m_p,j / s_j` — відстань ЦЕНТРА передактивації від порога ReLU у власних
одиницях. Плюс індивідуальна відстань семпла `p_j(z)/s_j` і частка «замерзлих».
Усе з СЕМПЛІВ, не із замикання.

🔒 ПЛОМБА: на шарі 0 `p = zW₀`, E[z]=0, біасів нема ⟹ `t ≡ 0` ТОЧНО.
   Ненульове там = вибірковий шум ⟹ це ще й калібровка шуму приладу.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  🔒 P0 шар 0: медіана |t| ~ 1/√N ≈ 0.008                        95%
  P1 медіана |t| росте з глибиною і НАСИЧУЄТЬСЯ на 2.0..2.5      65%
  P2 частка t>0 лишається ≈0.47..0.50 на КОЖНОМУ шарі            80%
  P3 частка «замерзлих» (|t|>3) росте до 30..50% на ш15          60%
  P4 частка (семпл,нейрон) біля межі |p/s|<0.5 падає з глибиною  70%
"""
import os, sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp
D, NW = 16, 1024
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
NN = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CH = 1 << 13
T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

ROWS = []
for f in sorted(glob.glob("/home/trim/wblogs/w2_*.npy"))[:NN]:
    nm = os.path.basename(f)[3:-4]
    W = np.load(f); Wg = [cp.asarray(W[i]) for i in range(D)]
    s1 = [cp.zeros(NW, dtype=cp.float64) for _ in range(D)]
    s2 = [cp.zeros(NW, dtype=cp.float64) for _ in range(D)]
    op = [cp.zeros(NW, dtype=cp.float64) for _ in range(D)]
    done = 0; c = 0
    while done < NS:                       # прохід 1: моменти p
        b = min(CH, NS-done)
        z = cp.random.RandomState(seed=555+c).standard_normal((b, NW), dtype=cp.float32)
        x = z
        for i in range(D):
            p = x @ Wg[i]
            s1[i] += p.sum(0, dtype=cp.float64); s2[i] += (p*p).sum(0, dtype=cp.float64)
            op[i] += (p > 0).sum(0, dtype=cp.float64)
            x = cp.maximum(p, 0.0)
        done += b; c += 1
    inv = 1.0/float(NS)
    mp = [s1[i]*inv for i in range(D)]
    sd = [cp.sqrt(cp.maximum(s2[i]*inv - mp[i]*mp[i], 1e-30)) for i in range(D)]
    tt = [mp[i]/sd[i] for i in range(D)]
    near = [0.0]*D
    done = 0; c = 0
    while done < NS//4:                    # прохід 2: індивідуальна відстань
        b = min(CH, NS//4-done)
        z = cp.random.RandomState(seed=777+c).standard_normal((b, NW), dtype=cp.float32)
        x = z
        for i in range(D):
            p = x @ Wg[i]
            near[i] += float((cp.abs(p/sd[i][None,:].astype(cp.float32)) < 0.5).sum())
            x = cp.maximum(p, 0.0)
        done += b; c += 1
    nrm = 1.0/float((NS//4)*NW)
    ROWS.append(dict(name=nm,
        med=[float(cp.median(cp.abs(tt[i]))) for i in range(D)],
        q25=[float(cp.percentile(cp.abs(tt[i]),25)) for i in range(D)],
        q75=[float(cp.percentile(cp.abs(tt[i]),75)) for i in range(D)],
        pos=[float((tt[i]>0).mean()) for i in range(D)],
        frz=[float((cp.abs(tt[i])>3.0).mean()) for i in range(D)],
        opn=[float(cp.median(op[i]*inv)) for i in range(D)],
        nea=[near[i]*nrm for i in range(D)]))
    log("%-18s |t| ш0 %.4f · ш1 %.3f · ш8 %.3f · ш15 %.3f" %
        (nm, ROWS[-1]["med"][0], ROWS[-1]["med"][1], ROWS[-1]["med"][8], ROWS[-1]["med"][15]))
    del Wg; cp.get_default_memory_pool().free_all_blocks()

def med(k): return np.median(np.array([r[k] for r in ROWS]), 0)
print("\n🔒 ПЛОМБА шар 0: медіана |t| = %.5f  [теорія ~1/√%d = %.5f]"
      % (med("med")[0], NS, 1.0/np.sqrt(NS)))
print("\nшар   медіана|t|  IQR |t|        частка t>0  замерзлих |t|>3  ворота відкриті  біля межі |p/s|<0.5")
for i in range(D):
    print("%3d     %6.3f    %5.3f–%5.3f      %.4f         %5.1f%%           %.4f            %5.1f%%"
          % (i, med("med")[i], med("q25")[i], med("q75")[i], med("pos")[i],
             100*med("frz")[i], med("opn")[i], 100*med("nea")[i]))
log("ГОТОВО")
