"""С116: СТЕЛЯ КЛАСУ — ОДИН СТЕНД, ОДНА БАЗА, ПЛОМБИ НА ОБОХ КІНЦЯХ ОСІ.

ПРИЧИНА. У корпусі ДВА несумісні числа стелі двомоментного класу:
   neuroC (фаза 2)  повний оракул ⟹ E² 1.293e-07 ⟹ похибка 0.0393%   ЛОГУ НЕМА
   ginfo  (С115-л)  інформаційна межа двох моментів      0.0841%
різниця ×2.1, і на обох будувалась стратегія. Вони вкидають оракул У РІЗНИХ
ТОЧКАХ, і це ніде не зафіксовано.

🚨 ЗАКОН С105, ЯКИЙ ТУТ ВИРІШУЄ: підстановка оракула НА КОЖНОМУ КРОЦІ не лише
зменшує похибку, а ПРИГНІЧУЄ ЇЇ НАКОПИЧЕННЯ — а накопичення = 84% похибки.
⟹ «@ALL» НЕ Є ДОСЯЖНОЮ СТЕЛЕЮ для реального методу; досяжна — «@K один раз».

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 mpC@ALL ≫ mpC@14 ⟹ ×11.1 корпусу — саме @ALL, тобто НЕДОСЯЖНА  70%
  P2 🔒 mpC@0 ≈ база (шар 0 гаусів ТОЧНО, вкидати нічого)            90%
  P3 mpC@ALL відтворить ×11.1 у MSE                                  60%
  P4 mpC@14 дасть ×2.5…3.0 (gacc2 давав ×2.466)                      65%
ГЕЙТ: якщо P1 справдиться — записане «клас промахується у ×1.31» ХИБНЕ,
і чинним лишається ginfo-подібне число.

🔑 ПІДЛОГА ПРИЛАДУ ВРАХОВАНА: ціль береться з банку N=2²³ (підлога 8.6e-09),
а не з dev24 (підлога 1.81e-07 = 15% нашого MSE). Друкуємо обидві.
"""
import sys, glob, time, numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

V_MC   = 3.3478454e-06     # [С115-е, грейдер] дисперсія МК-гілки
MULT   = 0.100131          # [328903] множник на підлозі
TRUE2  = 0.836447          # mean(true²) по dev24

def closure(W, mpT=None, CT=None, inject=()):
    """inject — множина шарів, де підставляємо ІСТИННІ (mp, C) передактивацій"""
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]
        cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        mp = m @ w
        if i in inject:
            if mpT is not None: mp = mpT[i].copy()
            if CT  is not None: cp = CT[i].astype(np.float64)
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

HI = {}
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz")):
    d = np.load(f, allow_pickle=True); HI[str(d["name"])] = d["m"][15].astype(np.float64)
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]

NETS = []
for f in sorted(glob.glob("/home/trim/wblogs/truthC_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    if nm not in HI or nm not in NAMES: continue
    i = NAMES.index(nm)
    NETS.append(dict(name=nm,
        W=np.asarray(tb["weights"][i].as_py(), dtype=np.float64).reshape(D, NW, NW),
        mpT=d["mp"].astype(np.float64), CT=d["C"],
        T=HI[nm],
        Tdev=np.asarray(tb["all_layer_means"][i].as_py(), dtype=np.float64).reshape(D, NW)[15],
        av=tb["avg_variance"][i].as_py()))
    lg("завантажено %-18s (банк C: N=%d)" % (nm, int(d["n"])))
NN = len(NETS)
lg("мереж %d" % NN)

KS = [0, 4, 8, 10, 12, 14, 15]
MODES = [("база", ())]
for K in KS: MODES.append(("mpC@%d" % K, (K,)))
MODES.append(("mpC@ALL", tuple(range(D))))
MODES.append(("mpC@8..15", tuple(range(8, D))))
for K in [8, 12, 14, 15]: MODES.append(("mp@%d" % K, (K,)))
for K in [8, 12, 14, 15]: MODES.append(("C@%d" % K, (K,)))

RES = {}
for nm_mode, inj in MODES:
    mses, mdev, cos_ = [], [], []
    for nd in NETS:
        onlym = nm_mode.startswith("mp@"); onlyC = nm_mode.startswith("C@")
        m = closure(nd["W"],
                    mpT=None if onlyC else nd["mpT"],
                    CT=None if onlym else nd["CT"],
                    inject=inj)
        T = nd["T"]
        mses.append(float(np.mean((m - T)**2)))
        mdev.append(float(np.mean((m - nd["Tdev"])**2)))
        # КОСИНУСНА: масштаб знято оптимальним a — саме це робить `_ai`
        aopt = float((m @ T)/max((m*m).sum(), 1e-300))
        cos_.append(float(np.mean((aopt*m - T)**2)))
    RES[nm_mode] = (float(np.mean(mses)), float(np.mean(mdev)), float(np.mean(cos_)))
    lg("%-12s сира %.4e  косинусна %.4e" % (nm_mode, RES[nm_mode][0], RES[nm_mode][2]))

E2b = RES["база"][0]; Cb = RES["база"][2]
E2_GRADER = 1.4355576e-06          # [С115-е, грейдер, _W_BLEND=1.0]
ANCH = E2_GRADER/Cb                # прив'язка: наша косинусна база → грейдерне E²
def adj(c):
    e2 = c*ANCH
    return (1.0/(1.0/V_MC + 1.0/e2))*MULT

print("\n" + "="*92)
print("СТЕЛЯ КЛАСУ — ОДНА ТАБЛИЦЯ, ОДНА БАЗА, %d мереж, ціль = банк N=2²³" % NN)
print("МЕТРИКА — КОСИНУСНА (масштаб знято, як це робить `_ai`). Сира — довідково.")
print("="*92)
print("%-12s %11s %9s %8s | %10s %8s | %11s %s" %
      ("режим","косинусна","похибка%","×база","сира","×база","adj(модель)","×до #1"))
SHOW = [m for m,_ in MODES if m not in ("mpC@ALL","mpC@8..15")]
for nm_mode in SHOW:
    e2, ed, c = RES[nm_mode]
    print("%-12s %11.4e %8.4f%% %8.3f | %10.3e %8.3f | %11.4e %6.2f"
          % (nm_mode, c, 100*np.sqrt(c/TRUE2), Cb/c, e2, E2b/e2, adj(c), adj(c)/7.30e-09))

print("\n🚨 mpC@ALL і mpC@8..15 ВИКИНУТО З ТАБЛИЦІ: вони ТОТОЖНІ mpC@15")
print("   (%.4e проти %.4e) — вкидання на ОСТАННЬОМУ шарі перекриває все раніше,"
      % (RES["mpC@ALL"][2], RES["mpC@15"][2]))
print("   бо ціль — вихід шару 15. ⟹ «оракул на кожному кроці» для ЦІЄЇ цілі не існує")
print("   як окремий режим, і суперечність ×2.1 у корпусі — НЕ про це.")

print("\n🔒 ПЛОМБИ")
print("  косинусна база %.4f%%  ·  прив'язка до грейдерного E²=1.4356e-06: ×%.3f"
      % (100*np.sqrt(Cb/TRUE2), ANCH))
print("  🔒 mpC@0 проти бази: ×%.4f  (мусить ≈1.000 — шар 0 гаусів ТОЧНО)" % (Cb/RES["mpC@0"][2]))
print("  🔒 adj(база) = %.4e  проти реального 1.0108e-07  (прив'язка за побудовою)" % adj(Cb))
print("  🔒 підлога dev24: перекіс сирої E²(dev24)/E²(2²³) %+.1f%% при передбачених %+.1f%%"
      % (100*(RES["база"][1]/E2b-1), 100*np.mean([nd["av"] for nd in NETS])/400000.0/E2b))

print("\n🎯 КАНАЛИ ОКРЕМО (косинусна, ×база)")
for K in [8,12,14,15]:
    m_,c_,mc_ = RES["mp@%d"%K][2], RES["C@%d"%K][2], RES["mpC@%d"%K][2]
    print("  шар %-3d  лише mp ×%6.3f · лише C ×%6.3f · разом ×%6.3f · добуток ×%6.3f ⟹ синергія ×%.3f"
          % (K, Cb/m_, Cb/c_, Cb/mc_, (Cb/m_)*(Cb/c_), (Cb/mc_)/((Cb/m_)*(Cb/c_))))

print("\n🎯 ВИРОК ПРО КЛАС")
print("  ідеальний двомоментний стан на ОСТАННЬОМУ шарі ⟹ adj %.3e" % adj(RES["mpC@15"][2]))
print("  #1 Puffi                                          7.30e-09")
print("  ⟹ клас промахується у ×%.2f" % (adj(RES["mpC@15"][2])/7.30e-09))
print("     [корпус казав ×1.31 (neuroC, логу нема) АБО ×2.8 (ginfo)]")
lg("ГОТОВО")
