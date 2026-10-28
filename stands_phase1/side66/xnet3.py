"""С116-C: ЕМПІРИКО-БАЄСІВ ПУЛІНГ ПО МЕРЕЖАХ — ЄДИНИЙ КРОС-МЕРЕЖЕВИЙ КАНАЛ,
ЩО ЛИШИВСЯ ПІСЛЯ (A) І (B).

ФАКТ, ЗМІРЯНИЙ У КОДІ ГРЕЙДЕРА (scoring.py:610, loader.py:30, subprocess_worker):
ОДИН екземпляр оцінювача проходить усі 100 мереж; `self` живе крізь прогін.
Бюджет — свій на кожну мережу ⟹ флопи не перекидаються (це закрив С98).

ЩО САМЕ ПУЛИТИ. Оцінювач має ОДИН помережний підібраний скаляр:
    _ai = ⟨ours, m_deep⟩/‖m_deep‖²          (шумний: МК на 6144 семплах)
    _a  = (1−S)·(1+_ALPHA) + S·_ai ,  S = _A_SHRINK = 0.5  ФІКСОВАНИЙ
Джеймс-Стайн каже, що оптимальне S — НЕ константа:
    S_i = σ²_між / (σ²_між + σ²_шум,i)
`σ²_шум,i` оцінюється з ПОЛОВИН власної вибірки мережі — БЕЗКОШТОВНО і БЕЗ
крос-мережевого стану. `σ²_між` — властивість ансамблю, вшивається офлайн.
⟹ причинності немає, порядок мереж не важить.

🚨 МЕТРИКА ТУТ ІНША, НІЖ У (A)/(B): `_ai` — це САМЕ МАСШТАБ, а косинусна
похибка масштаб знімає ⟹ вона до цього питання СЛІПА. Вирок виносимо на
СИРІЙ відносній похибці ‖out−T‖/‖T‖; косинусну друкуємо як контроль.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 глобальний оптимум S лежить біля 0.5 (офлайн-підбір мав рацію)   70%
  P2 помережне S_i виграє <×1.005                                    65%
  P3 оракульне S_i (стеля) ×1.01…1.05                                55%
  P4 🔒 перемішана оцінка шуму дасть рівно фіксоване S               85%
  P5 помережна стеля W_BLEND < ×1.03 (як у фазі 1)                   60%
ГЕЙТ ОГОЛОШЕНО ДО ПРОГОНУ: реалізовний виграш <×1.005 ⟹ мертво.
"""
import numpy as np, pyarrow.parquet as pq, sys, json, time
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
INV = 0.3989422804014327
NN    = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NPAIR = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
NREP  = int(sys.argv[3]) if len(sys.argv) > 3 else 4
ALPHA, WB, ASHR = -0.00134, 0.820, 0.5

def closure(W):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); MS = []
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        MS.append(m.copy())
        if i == d-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS

def sample3(W, z, iA, iB):
    """як у конвеєрі, але паралельно веде ПОВНУ вибірку і дві ПОЛОВИНИ.
    Половини ріжуться ПО ПАРАХ (антитетика лишається цілою)."""
    d = W.shape[0]
    cov = (z.T @ z)/z.shape[0]
    wz = np.linalg.inv(np.linalg.cholesky(cov)).T
    y = (z @ wz) @ W[0]
    x = np.concatenate([np.maximum(y, 0.0), np.maximum(-y, 0.0)], 0).astype(np.float32)
    npair = z.shape[0]
    jA = np.concatenate([iA, iA + npair]); jB = np.concatenate([iB, iB + npair])
    sg0 = np.sqrt(np.maximum((W[0]*W[0]).sum(0), 1e-30))
    v  = x.mean(0).astype(np.float64) - sg0*INV
    vA = x[jA].mean(0).astype(np.float64) - sg0*INV
    vB = x[jB].mean(0).astype(np.float64) - sg0*INV
    for i in range(1, d):
        p = x @ W[i].astype(np.float32)
        v  = (v  @ W[i]) * (p     > 0.0).mean(0).astype(np.float64)
        vA = (vA @ W[i]) * (p[jA] > 0.0).mean(0).astype(np.float64)
        vB = (vB @ W[i]) * (p[jB] > 0.0).mean(0).astype(np.float64)
        x = np.maximum(p, 0.0)
    f = x.astype(np.float64)
    return f.mean(0) - v, f[jA].mean(0) - vA, f[jB].mean(0) - vB

def rerr(x, tr):   # СИРА відносна похибка — саме її бачить грейдер
    return float(np.linalg.norm(x - tr)/np.linalg.norm(tr))
def cerr(x, tr):   # косинусна — контроль, до масштабу сліпа
    a = (x@tr)/max((x*x).sum(), 1e-300)
    return float(np.linalg.norm(a*x - tr)/np.linalg.norm(tr))

def out_of(ours, md, a, w=WB):
    return ours + w*(a*md - ours)

tb = pq.read_table(PQ)
NN = min(NN, tb.num_rows)
log("мереж %d · пар %d (⟹ %d семплів) · реплік %d" % (NN, NPAIR, 2*NPAIR, NREP))

REC = []
for j in range(NN):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    tr = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)[15]
    md = closure(W)[15]; den = max((md*md).sum(), 1e-300)
    for r in range(NREP):
        rng = np.random.default_rng(91000 + 100*j + r)
        z = rng.standard_normal((NPAIR, 1024)).astype(np.float32)
        pi = rng.permutation(NPAIR); iA, iB = pi[:NPAIR//2], pi[NPAIR//2:]
        ours, oA, oB = sample3(W, z, iA, iB)
        ai  = (ours @ md)/den
        aA  = (oA @ md)/den; aB = (oB @ md)/den
        # 🔑 дисперсія ПОВНОЇ оцінки = чверть квадрата різниці половин
        vnoise = 0.25*(aA - aB)**2
        REC.append(dict(j=j, r=r, ours=ours, md=md, tr=tr, ai=ai, vn=vnoise,
                        oA=oA, oB=oB))
    log("%2d  ai=%.5f  σ̂шум=%.5f" % (j, REC[-1]["ai"], np.sqrt(REC[-1]["vn"])))

AI = np.array([q["ai"] for q in REC]); VN = np.array([q["vn"] for q in REC])
print("\n🔒 ПЛОМБИ ПРИЛАДУ")
print("  a_i: сер %.5f · sd МІЖ мережами %.5f" % (AI.mean(), AI.std()))
print("  σ̂шум із половин: сер %.5f · sd %.5f" % (np.sqrt(VN).mean(), np.sqrt(VN).std()))
sd_rep = np.mean([np.std([q["ai"] for q in REC if q["j"] == j]) for j in range(NN)])
print("  🔒 sd a_i МІЖ РЕПЛІКАМИ тієї самої мережі: %.5f  (мусить ≈ σ̂шум)" % sd_rep)
print("     ⟹ відношення %.3f — це і є перевірка оцінки шуму з половин" % (sd_rep/max(np.sqrt(VN).mean(),1e-30)))
vb_raw = AI.var() - VN.mean()
print("  σ²між = Var(a_i) − сер σ²шум = %.3e − %.3e = %.3e" % (AI.var(), VN.mean(), vb_raw))
SB = max(vb_raw, 1e-12)
print("  ⟹ ДЖЕЙМС-СТАЙН S = σ²між/(σ²між+σ²шум) сер = %.4f  [у сабміті 0.5]"
      % np.mean(SB/(SB+VN)))

def ev(name, afun, wfun=None):
    ra, ca = [], []
    for q in REC:
        a = afun(q); w = WB if wfun is None else wfun(q)
        o = out_of(q["ours"], q["md"], a, w)
        ra.append(rerr(o, q["tr"])); ca.append(cerr(o, q["tr"]))
    return name, float(np.mean(ra)), float(np.mean(ca))

base = ev("S = 0.5 (САБМІТ)", lambda q: (1-ASHR)*(1+ALPHA) + ASHR*q["ai"])
print("\n── СКАН ГЛОБАЛЬНОГО S ──")
print("   S       сира%     ×база      косинусна%")
best = None
for S in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
    n, ra, ca = ev("S=%.2f" % S, lambda q, S=S: (1-S)*(1+ALPHA) + S*q["ai"])
    mark = "  ← сабміт" if abs(S-0.5) < 1e-9 else ""
    print("  %.2f   %8.4f   ×%.4f    %8.4f%s" % (S, 100*ra, base[1]/ra, 100*ca, mark))
    if best is None or ra < best[1]: best = (S, ra)
print("  ⟹ оптимум S=%.2f  ×%.4f" % (best[0], base[1]/best[1]))

print("\n── ПОМЕРЕЖНЕ S_i (Джеймс-Стайн, шум із половин) ──")
print("   схема                          сира%     ×база      косинусна%")
rows = [base]
rows.append(ev("EB: S_i=σ²b/(σ²b+σ²n,i)",
               lambda q: (lambda S: (1-S)*(1+ALPHA) + S*q["ai"])(SB/(SB+q["vn"]))))
# 🔒 контроль: перемішати оцінки шуму між записами (розподіл той самий, пара зламана)
perm = np.random.RandomState(31337).permutation(len(REC))
VNS = VN[perm]
for k, q in enumerate(REC): q["vns"] = VNS[k]
rows.append(ev("🔒 перемішаний σ²n",
               lambda q: (lambda S: (1-S)*(1+ALPHA) + S*q["ai"])(SB/(SB+q["vns"]))))
# оракульна стеля: S на мережу, що мінімізує СИРУ похибку
for q in REC:
    gs = np.linspace(0.0, 2.0, 201)
    er = [rerr(out_of(q["ours"], q["md"], (1-S)*(1+ALPHA) + S*q["ai"]), q["tr"]) for S in gs]
    q["Sopt"] = float(gs[int(np.argmin(er))])
rows.append(ev("🔒 ОРАКУЛЬНЕ S_i (стеля)", lambda q: (1-q["Sopt"])*(1+ALPHA) + q["Sopt"]*q["ai"]))
for n, ra, ca in rows:
    print("  %-30s %8.4f   ×%.4f    %8.4f" % (n, 100*ra, base[1]/ra, 100*ca))
SO = np.array([q["Sopt"] for q in REC])
print("  оракульне S: сер %.3f · sd %.3f · медіана %.3f" % (SO.mean(), SO.std(), np.median(SO)))
print("  🔒 corr(оракульне S_i , передбачене EB S_i) = %+.4f"
      % np.corrcoef(SO, SB/(SB+VN))[0, 1])

print("\n── ПОМЕРЕЖНИЙ _W_BLEND (той самий пул, безкоштовно) ──")
print("   схема                          сира%     ×база      косинусна%")
afix = lambda q: (1-ASHR)*(1+ALPHA) + ASHR*q["ai"]
rw = [base]
for Wg in [0.70, 0.78, 0.82, 0.86, 0.90]:
    rw.append(ev("W = %.2f" % Wg, afix, lambda q, Wg=Wg: Wg))
for q in REC:
    gw = np.linspace(0.3, 1.2, 181)
    er = [rerr(out_of(q["ours"], q["md"], afix(q), w), q["tr"]) for w in gw]
    q["Wopt"] = float(gw[int(np.argmin(er))])
rw.append(ev("🔒 ОРАКУЛЬНЕ W_i (стеля)", afix, lambda q: q["Wopt"]))
# реалізовне: V_i з половин ⟹ W = E²/(V+E²) з глобальним E²
dv = np.array([float(np.sum((q["oA"]-q["oB"])**2)/4.0) for q in REC])
for k, q in enumerate(REC): q["Vhat"] = dv[k]
E2 = np.median(dv)*(1/WB - 1)      # калібруємо так, щоб медіана дала нинішнє W
rw.append(ev("реалізовне W=V̂/(V̂+E²)", afix, lambda q: q["Vhat"]/(q["Vhat"]+E2)))
print("  🔒 калібровка: при V̂=медіана W = %.4f (мусить = %.2f)"
      % (np.median(dv)/(np.median(dv)+E2), WB))
for n, ra, ca in rw:
    print("  %-30s %8.4f   ×%.4f    %8.4f" % (n, 100*ra, base[1]/ra, 100*ca))
WO = np.array([q["Wopt"] for q in REC])
print("  оракульне W: сер %.3f · sd %.3f   🔒 corr(W_i оракул, V̂_i) = %+.4f"
      % (WO.mean(), WO.std(), np.corrcoef(WO, dv)[0, 1]))
np.savez_compressed("/home/trim/wblogs/xnet3_rec.npz",
    j=np.array([q["j"] for q in REC]), ai=AI, vn=VN, Vhat=dv,
    Sopt=SO, Wopt=WO)
log("ГОТОВО")
