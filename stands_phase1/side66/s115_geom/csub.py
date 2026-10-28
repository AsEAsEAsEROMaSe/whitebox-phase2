"""С115-ц: СТЕЛЯ ДРУГОМОМЕНТНОЇ СІМ'Ї — вкидання істинної C НА ВСІХ ШАРАХ РАЗОМ.

Обіцяне після xcov3 і не зроблене: стеля R²=0.022 закрила Δ як ПРЕДИКТОР b⊥,
але не як ЗМІНУ РЕКУРСІЇ. Δ = відхилення Cov(h_l,h_{l+1}) від Стайна — рідний
брат похибки оновлення `C` (обидва є другомоментною мапою під гаусовим
припущенням). ⟹ стеля вкидання істинної C обмежує ВСЮ цю сім'ю, Δ включно.

🚨 `gacc2` міряла C ПО ОДНОМУ шару (×1.037/1.022/1.038/1.011). Наш закон каже
   «КАНАЛИ МНОЖАТЬСЯ» ⟹ разом може бути інакше. Це і перевіряємо.
🔒 вкидання з половини A банку, вирок проти половини B
🔒 контроль k=[] мусить дати базу БІТ У БІТ

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 поодинокі C дають ×1.0x (відтворять форму gacc2)          75%
  P2 C на всіх 4 разом: ×1.05..×1.20                           60%
  P3 m на всіх 4 разом: ×2.0+                                  70%
  P4 mC разом БІЛЬШЕ за добуток окремих                        50%
  P5 неузгоджена пара (істинна C + наше m) може ШКОДИТИ        45%
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np, glob
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
KEEP = [8, 10, 12, 14]

def closure(W, injC=None, injM=None):
    """injC/injM: dict шар -> вектор/матриця (істина). Підміна ПІСЛЯ кроку."""
    m = np.zeros(NW); C = np.eye(NW); MS = []
    for i in range(D):
        w = W[i]; q = w.T @ C @ w
        if i > KSIG: q = q*KAPPA
        s = np.sqrt(np.maximum(np.diag(q), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t); m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(q, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        if injM is not None and i in injM: m = injM[i].copy()
        MS.append(m.copy())
        if i == D-1: break
        C = q*np.outer(Ph, Ph) + 0.5*(q*q)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
        if injC is not None and i in injC: C = injC[i].copy()
    return MS

def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(), 1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

NETS = []
for f in sorted(glob.glob("/home/trim/wblogs/truthmc_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    wf = "/home/trim/wblogs/w2_%s.npy" % nm
    if not os.path.exists(wf): continue
    CA = {k: np.asarray(d["C%d" % k])[0].astype(np.float64) for k in KEEP}
    MA = {k: np.asarray(d["m%d" % k])[0].astype(np.float64) for k in KEEP}
    NETS.append(dict(name=nm, W=np.load(wf).astype(np.float64),
                     CA=CA, MA=MA, tgt=np.asarray(d["mf"])[1].astype(np.float64)))
log("мереж: %d" % len(NETS))

base = [cerr(closure(N["W"])[15], N["tgt"]) for N in NETS]
B0 = float(np.median(base))
log("база %.4f%%   (ціль = половина B банку)" % (100*B0))
z = [cerr(closure(N["W"], injC={}, injM={})[15], N["tgt"]) for N in NETS]
log("🔒 контроль порожньої підміни: %.4f%%  (мусить = база)" % (100*np.median(z)))

def run(tag, ks, what):
    e = []
    for N in NETS:
        iC = {k: N["CA"][k] for k in ks} if "C" in what else None
        iM = {k: N["MA"][k] for k in ks} if "m" in what else None
        e.append(cerr(closure(N["W"], injC=iC, injM=iM)[15], N["tgt"]))
    v = float(np.median(e))
    w = sum(1 for j in range(len(NETS)) if e[j] < base[j])
    print("%-22s %.4f%%   ×%.4f   краще %d/%d" % (tag, 100*v, B0/v, w, len(NETS)), flush=True)
    return B0/v

print("\nрежим                  похибка     ×база    знаковий")
sing = []
for k in KEEP: sing.append(run("C@%d" % k, [k], "C"))
for k in KEEP: run("m@%d" % k, [k], "m")
print()
rc = run("C@ВСІ ЧОТИРИ", KEEP, "C")
rm = run("m@ВСІ ЧОТИРИ", KEEP, "m")
rb = run("mC@ВСІ ЧОТИРИ", KEEP, "mC")
print("\n🔑 добуток поодиноких C = ×%.4f   проти разом ×%.4f   ⟹ синергія ×%.3f"
      % (np.prod(sing), rc, rc/np.prod(sing)))
print("🔑 m разом ×%.4f · C разом ×%.4f · добуток ×%.4f   проти mC разом ×%.4f"
      % (rm, rc, rm*rc, rb))
log("ГОТОВО")
