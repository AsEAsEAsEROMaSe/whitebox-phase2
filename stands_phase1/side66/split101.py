"""split101 — РОЗДІЛИТИ `m = mp·Φ(t) + s·φ(t)` НА ДВА ВЕКТОРИ І ФІТУВАТИ ОБИДВА.

МОТИВ (з нового закону С101-в). Ціль ТОЧНО косинусна ⟹ важить лише КУТ. Ми
фітуємо ОДИН скаляр `_ai` на СУМУ двох доданків, хоча вони мають різну природу:
`mp·Φ` — перенесене середнє, `s·φ` — внесок дисперсії через ворота. Дати їм
РІЗНІ ваги = повернути кут, а не масштаб.
🔒 ВБУДОВАНА ПЛОМБА: при `a = b` це ТОЧНО нинішній метод ⟹ програти in-sample
   неможливо, вирок — лише на LOO/кросі.
🚨 ВІДМІННІСТЬ ВІД sub120 (`lam`, ×0.9955): там була ГЛОБАЛЬНА калібрувальна
   константа на всіх шарах; тут — ДВА коефіцієнти, фітовані ПО МЕРЕЖІ проти
   ВЛАСНОЇ МК, без жодної константи з іншого зрізу.

ПЕРШЕ ПИТАННЯ — ОБУМОВЛЕНІСТЬ. Якщо `cos(mp·Φ, s·φ) ≈ 1`, вектори колінеарні,
задача 2×2 вироджена, і шум МК рознесе коефіцієнти. Це вбиває ідею ДО реалізації.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) cos(A,B) буде в 0.90…0.995                                     55%
 (б) оракульний виграш (фіт проти ІСТИНИ) перевищить ×1.05          45%
 (в) з реальним шумом МК виграш впаде щонайменше вдвічі             75%
 (г) LOO-стиснення виявиться потрібним (як `_A_SHRINK` = 0.5)       70%
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG, KAPPA = 8, 0.9985
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 50
NMC = int(sys.argv[2]) if len(sys.argv) > 2 else 6144      # ≈ бюджет сабміту

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
yy = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(yy.type):
    yy = yy.flatten()
FM = np.asarray(yy).reshape(tb.num_rows, W_).astype(np.float64)
NNET = min(NNET, tb.num_rows)
print(f"split101 · {NNET} мереж · МК {NMC} пар", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def parts(Wg):
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    for i in range(DEPTH):
        w = Wg[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * KAPPA
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        A, B = mp * Ph, s * ph
        mn = A + B
        if i == DEPTH - 1:
            # 🔑 БАЗИС КЛАСУ «лінійна корекція у величинах, які замикання ВЖЕ має»
            return cp.stack([A, B, A * t, B * t, mp, s, A * t * t, B * t * t])
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        Co = cpm * cp.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        C = Co + cp.diag(d - cp.diagonal(Co))
        m = mn


def mc(Wg, npair, seed):
    """МК з антитетикою + вибілюванням — як у сабміті"""
    rng = cp.random.default_rng(seed)
    z = rng.standard_normal((npair, W_), dtype=cp.float64)
    cov = (z.T @ z) / npair
    L = cp.linalg.cholesky(cov)
    z = cp.linalg.solve_triangular(L, z.T, lower=True).T if hasattr(
        cp.linalg, "solve_triangular") else z @ cp.linalg.inv(L).T
    a = cp.concatenate([z, -z], axis=0).astype(cp.float32)
    for i in range(DEPTH):
        p = a @ Wg[i].astype(cp.float32)
        a = cp.maximum(p, 0.0) if i < DEPTH - 1 else p
    return (cp.maximum(a, 0.0).sum(0, dtype=cp.float64) / (2 * npair))


NB = 8
BAS = np.zeros((NNET, NB, W_))
YY = np.zeros((NNET, W_))
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float64))
          for i in range(DEPTH)]
    BAS[net] = cp.asnumpy(parts(Wg))
    YY[net] = cp.asnumpy(mc(Wg, NMC, 12345 + net))
    if (net + 1) % 10 == 0:
        print(f"  {net+1}/{NNET}", flush=True)
T = FM[:NNET]

NAMES = ["mp·Φ", "s·φ", "mp·Φ·t", "s·φ·t", "mp", "s", "mp·Φ·t²", "s·φ·t²"]
AA, BB = BAS[:, 0], BAS[:, 1]
cs = [float(np.dot(AA[k], BB[k]) /
            math.sqrt(np.dot(AA[k], AA[k]) * np.dot(BB[k], BB[k])))
      for k in range(NNET)]
print(f"\n🚨 ОБУМОВЛЕНІСТЬ: cos(mp·Φ, s·φ) сер {np.mean(cs):.6f}")


def err(v, t):
    return math.sqrt(((v - t) ** 2).mean() / (t * t).mean())


def fitk(Bk, y, lam=1e-12):
    G = Bk @ Bk.T
    G = G + lam * np.trace(G) / len(G) * np.eye(len(G))
    return np.linalg.solve(G, Bk @ y)


print(f"\n{'K':>3s} {'базис':<34s}{'ОРАКУЛ':>10s}{'реальна МК':>13s}{'×до K=1':>10s}")
print("=" * 72)
base_or = base_mc = None
for K in range(1, NB + 1):
    eo, em = [], []
    for k in range(NNET):
        Bk = BAS[k, :K]
        eo.append(err(fitk(Bk, T[k]) @ Bk, T[k]))
        em.append(err(fitk(Bk, YY[k]) @ Bk, T[k]))
    eo, em = float(np.mean(eo)), float(np.mean(em))
    if K == 1:
        base_or, base_mc = eo, em
    print(f"{K:>3d} {'+' + NAMES[K-1]:<34s}{100*eo:>9.4f}%{100*em:>12.4f}%"
          f"{base_mc/em:>10.4f}")
print("\n🔑 K=1 — це РІВНО нинішній `_ai` (один скаляр на m). Стовпець ОРАКУЛ —")
print("   СТЕЛЯ класу «лінійна корекція у величинах замикання».")
print("🔒 КОНТРОЛЬ ЧУЖА: базис іншої мережі ⟹ мусить ЗЛАМАТИ")
ef = []
for k in range(NNET):
    o = (k + 1) % NNET
    Bk = BAS[o, :4]
    ef.append(err(fitk(BAS[k, :4], YY[k]) @ Bk, T[k]))
print(f"   K=4 своя {100*np.mean([err(fitk(BAS[k,:4], YY[k]) @ BAS[k,:4], T[k]) for k in range(NNET)]):.4f}%"
      f"  ·  ЧУЖА {100*np.mean(ef):.4f}%")
