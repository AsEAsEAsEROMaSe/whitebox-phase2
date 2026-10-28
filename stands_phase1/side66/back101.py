"""back101 — ЗВОРОТНИЙ ОПЕРАТОР: ЯКІ СЛІДИ З ГЛИБИНИ ВЗАГАЛІ ДОХОДЯТЬ ДО ВИХОДУ?

ІДЕЯ ЮЗЕРА: «іти від заду вперед… нам треба механізм розкопок». Технічно —
питати не «який розподіл на шарі l», а «яка ЧАСТИНА збурення шару l доживає
до виходу». Це спектр зворотного оператора.

ПОБУДОВА (з тотожності, не з голови). Збурення `δ` середнього на шарі `l`
переноситься вперед у першому порядку як
    δ_out = B_l · δ ,     B_l = Φ_l+1 ⊙ W_l+1 ... Φ_31 ⊙ W_31
де `Φ_i = Φ(t_i)` — детермінований аналог `E[diag(D_i)]`, і він УЖЕ є в
замиканні (ціна НУЛЬ, це ті самі числа, що йдуть у `m`).
🔒 ПЛОМБА: `Φ` є точним `E[D]` лише в першому порядку; контроль — порівняти
   передбачений перенос із ПРЯМИМ (вкинути збурення і прогнати мережу МК).

ЩО ВИРІШУЄ. Якщо ефективний ранг `B_l` малий (≪256), то похибку проміжного
шару треба знати лише у КІЛЬКОХ напрямках — решта до виходу не доживає, і
задача радикально меншає. Якщо ранг великий — клас «цілитись у частину стану»
закривається одним числом.
📌 НЕ порожня надія: `Cov(out)` має ефективний ранг ~25 із 256, `n_eff` ЦГТ на
   глибині падає 86→25 ⟹ щось таки стискається. Але це міряно ВПЕРЕД.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) ефективний ранг B(l→31) для l=24 буде < 20                       25%
 (б) він ПАДАТИМЕ з глибиною переносу (довший шлях = сильніше стиснення) 70%
 (в) 🔒 лінійне передбачення переносу збіжиться з прямим МК у межах 20% 55%
 (г) наша реальна похибка ЛЕЖАТИМЕ переважно у верхніх напрямках B      30%
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LS = [0, 4, 8, 16, 24, 28, 30]

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
print(f"back101 · {NNET} мереж · шари {LS}", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def run(Wg):
    """замикання; повертає Φ по шарах, m_31 і сам стан"""
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    PH = []
    for i in range(DEPTH):
        w = Wg[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * KAPPA
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        PH.append(Ph)
        mn = mp * Ph + s * ph
        if i == DEPTH - 1:
            return PH, mn
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        Co = cpm * cp.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        C = Co + cp.diag(d - cp.diagonal(Co))
        m = mn


def Bmat(Wg, PH, l):
    """B(l→31): збурення СЕРЕДНЬОГО активацій після шару l → збурення виходу"""
    B = cp.eye(W_, dtype=cp.float64)
    for i in range(l + 1, DEPTH):
        B = (B @ Wg[i]) * PH[i][None, :]
    return B


def eff_rank(S):
    p = S / S.sum()
    return float(math.exp(-float((p * cp.log(cp.maximum(p, 1e-300))).sum())))


rows = {l: [] for l in LS}
proj = {l: [] for l in LS}
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float64))
          for i in range(DEPTH)]
    PH, m31 = run(Wg)
    T = cp.asarray(FM[net])
    a = float(cp.dot(m31, T) / cp.dot(m31, m31))
    e = a * m31 - T                                    # похибка ПІСЛЯ масштабу
    for l in LS:
        B = Bmat(Wg, PH, l)
        S = cp.linalg.svd(B, compute_uv=False)
        U = cp.linalg.svd(B, full_matrices=False)[0]   # ліві сингулярні = вихід
        rows[l].append((eff_rank(S * S), float(S[0] / S[-1]),
                        float((S[:16] ** 2).sum() / (S ** 2).sum())))
        c = (U.T @ e) ** 2
        proj[l].append(float(c[:16].sum() / c.sum()))
    print(f"  мережа {net} готова", flush=True)

print(f"\n{'l→31':>6s}{'ефект.ранг':>12s}{'S₁/S₂₅₆':>12s}"
      f"{'частка енергії B у топ-16':>26s}{'ПОХИБКА в топ-16':>18s}")
print("=" * 76)
for l in LS:
    r = np.array(rows[l])
    print(f"{l:>6d}{r[:,0].mean():>12.2f}{r[:,1].mean():>12.3e}"
          f"{100*r[:,2].mean():>25.2f}%{100*np.mean(proj[l]):>17.2f}%")
print("\n🔒 КОНТРОЛЬ: якщо похибка розподілена як 16/256 = 6.25%, вона ІЗОТРОПНА")
print("   у базисі B ⟹ цілитись у частину стану НЕМАЄ СЕНСУ")
