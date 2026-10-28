"""С115-т: ВСЯ ПОХИБКА КРОКУ СИДИТЬ У ХВОСТОВОМУ ДОДАНКУ (спостереження ЮЗЕРА).

ТОТОЖНІСТЬ:  relu(p) = p + (−p)₊   ⟹   E[relu(p)] = m_p + E[(−p)₊]
`m_p = m_l@W` ТОЧНЕ при точному m_l ⟹ уся похибка КРОКУ = похибка в E[(−p)₊].

ПИТАННЯ: з якою ВІДНОСНОЮ точністю ми рахуємо цей доданок? Наша сумарна похибка
0.12% від ‖m‖, а сам доданок ~0.22% від ‖m‖ ⟹ якщо вона вся звідти, відносна
похибка в ньому — десятки відсотків. Це виведено змішуванням чисел ⟹ МІРЯЮ.

СХЕМА: істинний стан із семплів. На кожному шарі беремо ІСТИННІ (m_l, розподіл),
рахуємо істинне E[(−p)₊] і гаусове s·φ(t) − m·Φ(−t) з ТИХ САМИХ (m_p, s).
⟹ ізолює похибку ФОРМИ від похибки переносу.
🔒 ПЛОМБА шар 0: p = zW₀ гаусів ТОЧНО ⟹ відносна похибка = вибірковий шум.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  🔒 P0 шар 0 дає ~0 (гаусів точно)                              90%
  P1 відносна похибка хвостового доданка на глибині 10..40%      70%
  P2 вона РОСТЕ з глибиною                                       60%
  P3 частка доданка в ‖m‖ падає з глибиною                       85%
  P4 відносна похибка ПОВНОГО E[relu] на 2 порядки менша         80%
"""
import os, sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp
from cupyx.scipy.special import erf as cerf
D, NW = 16, 1024
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
NN = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CH = 1 << 13
INV = 0.3989422804014327
T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def PHI(x): return 0.5*(1.0 + cerf(x/cp.sqrt(cp.float64(2.0))))
def PHIs(x): return cp.exp(-0.5*x*x)*INV

ROWS = []
for f in sorted(glob.glob("/home/trim/wblogs/w2_*.npy"))[:NN]:
    nm = os.path.basename(f)[3:-4]
    W = np.load(f); Wg = [cp.asarray(W[i]) for i in range(D)]
    s1=[cp.zeros(NW,dtype=cp.float64) for _ in range(D)]
    s2=[cp.zeros(NW,dtype=cp.float64) for _ in range(D)]
    tl=[cp.zeros(NW,dtype=cp.float64) for _ in range(D)]   # Σ (−p)₊
    rl=[cp.zeros(NW,dtype=cp.float64) for _ in range(D)]   # Σ relu(p)
    done=0;c=0
    while done<NS:
        b=min(CH,NS-done)
        z=cp.random.RandomState(seed=31337+c).standard_normal((b,NW),dtype=cp.float32)
        x=z
        for i in range(D):
            p=x@Wg[i]
            s1[i]+=p.sum(0,dtype=cp.float64); s2[i]+=(p*p).sum(0,dtype=cp.float64)
            tl[i]+=cp.maximum(-p,0.0).sum(0,dtype=cp.float64)
            x=cp.maximum(p,0.0); rl[i]+=x.sum(0,dtype=cp.float64)
        done+=b;c+=1
    inv=1.0/float(NS)
    row=dict(name=nm,rel=[],relfull=[],share=[],tshare=[],shP=[],relP=[],errP=[],errN=[],engP=[])
    for i in range(D):
        mp=s1[i]*inv; sd=cp.sqrt(cp.maximum(s2[i]*inv-mp*mp,1e-30)); t=mp/sd
        Ttrue=tl[i]*inv; Tg=sd*PHIs(t)-mp*PHI(-t)
        Rtrue=rl[i]*inv; Rg=mp+Tg
        P=(t>0); Nn=~P
        nrm=lambda v,msk: float(cp.linalg.norm(v[msk]))
        row["rel"].append(float(cp.linalg.norm(Tg-Ttrue)/cp.linalg.norm(Ttrue)))
        row["relfull"].append(float(cp.linalg.norm(Rg-Rtrue)/cp.linalg.norm(Rtrue)))
        row["share"].append(float(cp.linalg.norm(Ttrue)/cp.linalg.norm(Rtrue)))
        row["tshare"].append(float(cp.linalg.norm(Tg-Ttrue)/cp.linalg.norm(Rtrue)))
        # РОЗДІЛ ЗА ЗНАКОМ t
        row["engP"].append(nrm(Rtrue,P)**2/max(float((Rtrue*Rtrue).sum()),1e-300))  # частка енергії m у t>0
        row["shP"].append(nrm(Ttrue,P)/max(nrm(Rtrue,P),1e-300))                    # доданок/значення, t>0
        row["relP"].append(nrm(Tg-Ttrue,P)/max(nrm(Ttrue,P),1e-300))                # відн.похибка доданка, t>0
        row["errP"].append(nrm(Rg-Rtrue,P)/max(nrm(Rtrue,P),1e-300))                # похибка E[relu], t>0
        row["errN"].append(nrm(Rg-Rtrue,Nn)/max(nrm(Rtrue,Nn),1e-300))              # те саме, t<0
    ROWS.append(row)
    log("%-18s хвіст ш0 %.5f · ш8 %.4f · ш15 %.4f" % (nm,row["rel"][0],row["rel"][8],row["rel"][15]))
    del Wg; cp.get_default_memory_pool().free_all_blocks()

def med(k): return np.median(np.array([r[k] for r in ROWS]),0)
print("\n🔒 ПЛОМБА шар 0 (p гаусів ТОЧНО): відн. похибка хвоста %.5f" % med("rel")[0])
print("\nшар  РОЗДІЛ ЗА ЗНАКОМ t")
print("     енергія m у t>0   доданок/знач (t>0)   відн.пох.доданка (t>0)   пох.E[relu] t>0   пох.E[relu] t<0")
for i in range(D):
    print("%3d      %8.5f          %9.5f              %9.4f              %9.6f       %9.4f"
          % (i, med("engP")[i], med("shP")[i], med("relP")[i], med("errP")[i], med("errN")[i]))
log("ГОТОВО")
