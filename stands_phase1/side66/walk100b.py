"""walk100b — КОНТРОЛЬ НА КОНЦЕНТРАЦІЮ + ЧИ ПЕРЕДБАЧУВАНІ «ВИННІ» НЕЙРОНИ.

walk100 показала: 8 нейронів із 256 дають 50% енергії локальної похибки на
шарі 31, і в них МАЛИЙ |t| при медіані 3.98. Три питання, без яких це не вимір:

 (1) КОНТРОЛЬ: чи не тривіальна концентрація? Еталон — 256 i.i.d. гаусів
     (для них топ-8 дають ~22%, не 50%). І чи не є це просто розкидом `s`?
 (2) ПЕРЕДБАЧУВАНІСТЬ: якщо взяти топ-k нейронів за |t| (величина, ЯКУ
     ЗАМИКАННЯ ЗНАЄ ДЕТЕРМІНОВАНО), яку частку РЕАЛЬНОЇ енергії похибки
     ВИХОДУ вони покриють? Це і є «чи можна цілитись без істини».
 (3) ЩО САМЕ ЛІКУВАТИ: розклад енергії фінальної похибки `b_31` за |t|
     (пломба до запису С98 «96.4% енергії каналу m при |t|<2»).

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) концентрація ПЕРЕВИЩИТЬ гаусів контроль (топ-8 > 30%)            80%
 (б) топ-25% нейронів за МАЛИМ |t| покриють > 60% енергії b_31        65%
 (в) нормування на `s` НЕ прибере концентрацію (лишиться > 35%)       60%
 (г) картина відтвориться на другій мережі з тим самим порядком       85%
"""
import glob
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16384


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def conc(v, k):
    """частка енергії у топ-k компонентах вектора v"""
    e = np.sort(np.asarray(v, float) ** 2)[::-1]
    return float(e[:k].sum() / max(e.sum(), 1e-300))


def energy_below(v, key, thr):
    """частка енергії v на компонентах, де key < thr"""
    e = np.asarray(v, float) ** 2
    msk = np.asarray(key, float) < thr
    return float(e[msk].sum() / max(e.sum(), 1e-300)), float(msk.mean())


f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
x = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(x.type):
    x = x.flatten()
Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, W_, W_)

# гаусів еталон концентрації (аналітично через розіграші)
rg = np.random.default_rng(1)
gc = np.array([[conc(rg.standard_normal(W_), k) for k in (8, 29, 64)]
               for _ in range(400)]).mean(0)
print(f"walk100b · {NNET} мереж × {2*NS} семплів")
print(f"🔒 ГАУСІВ ЕТАЛОН КОНЦЕНТРАЦІЇ (256 i.i.d.): "
      f"топ-8 {100*gc[0]:.1f}% · топ-29 {100*gc[1]:.1f}% · "
      f"топ-64 {100*gc[2]:.1f}%\n", flush=True)

agg = {k: [] for k in ("c8", "c29", "c8n", "c29n", "cs8",
                       "e_lo", "f_lo", "e_lo1", "f_lo1",
                       "pred25", "pred50", "rnd25", "top_s25")}

for net in range(NNET):
    W = [np.ascontiguousarray(Wall[net][L], dtype=np.float64)
         for L in range(DEPTH)]
    rng = np.random.default_rng(777 + net)
    z = rng.standard_normal((NS, W_))
    A = np.concatenate([z, -z], axis=0)

    m_h, C_h = np.zeros(W_), np.eye(W_)
    for L in range(DEPTH):
        w = W[L]
        P = A @ w
        m_t = P.mean(0)
        Pc = P - m_t
        s_t = np.sqrt((Pc * Pc).mean(0))

        cp = w.T @ C_h @ w
        s_h = np.sqrt(np.maximum(np.diagonal(cp), 1e-20))
        mp = m_h @ w
        t = mp / s_h
        ph, Ph = phi(t), ndtr(t)
        m_new = mp * Ph + s_h * ph
        Areal = np.maximum(P, 0.0)
        ma_t = Areal.mean(0)
        b = m_new - ma_t
        if L == DEPTH - 1:
            break
        A = Areal
        Ca = w.T @ C_h @ w
        Ca = Ca * np.outer(Ph, Ph) + (0.5 * (Ca * Ca)) * np.outer(ph / s_h,
                                                                 ph / s_h)
        d = (mp * mp + s_h * s_h) * Ph + (mp * s_h) * ph - m_new * m_new
        C_h = Ca + np.diag(d - np.diagonal(Ca))
        m_h = m_new

    # ── фінальна похибка ПІСЛЯ зняття масштабу (те, що робить _ALPHA) ──
    a_opt = float((m_new @ ma_t) / max(float(m_new @ m_new), 1e-300))
    bb = a_opt * m_new - ma_t
    tt = np.abs(t)

    agg["c8"].append(conc(bb, 8))
    agg["c29"].append(conc(bb, 29))
    # (в) нормування на s — чи це просто розкид масштабу?
    agg["c8n"].append(conc(bb / np.maximum(s_h, 1e-300), 8))
    agg["c29n"].append(conc(bb / np.maximum(s_h, 1e-300), 29))
    # контроль: концентрація самого сигналу (не похибки)
    agg["cs8"].append(conc(m_new, 8))

    # (3) розклад енергії за |t|
    e2, f2 = energy_below(bb, tt, 2.0)
    agg["e_lo"].append(e2)
    agg["f_lo"].append(f2)
    e1, f1 = energy_below(bb, tt, 1.0)
    agg["e_lo1"].append(e1)
    agg["f_lo1"].append(f1)

    # (2) ПЕРЕДБАЧУВАНІСТЬ: топ-k за МАЛИМ |t| — чисто з замикання
    order = np.argsort(tt)
    en = bb ** 2
    tot = max(en.sum(), 1e-300)
    for frac, key in ((0.25, "pred25"), (0.50, "pred50")):
        k = int(W_ * frac)
        agg[key].append(float(en[order[:k]].sum() / tot))
    # контроль: випадкові 25%
    agg["rnd25"].append(float(np.mean(
        [en[rng.permutation(W_)[:64]].sum() / tot for _ in range(200)])))
    # другий претендент: топ-25% за НАЙБІЛЬШИМ s
    agg["top_s25"].append(float(en[np.argsort(-s_h)[:64]].sum() / tot))

    print(f"  мережа {net}: конц8 {100*agg['c8'][-1]:.1f}% · "
          f"|t|<2 несе {100*e2:.1f}% енергії при {100*f2:.1f}% нейронів · "
          f"топ-25%|t| {100*agg['pred25'][-1]:.1f}%", flush=True)


def st(k):
    v = np.array(agg[k])
    return f"{100*v.mean():6.1f}% ± {100*v.std(ddof=1)/math.sqrt(len(v)):.1f}"


print("\n" + "=" * 74)
print("(1) КОНЦЕНТРАЦІЯ похибки виходу (після зняття масштабу)")
print(f"    топ-8  нейронів:            {st('c8')}   гаус-еталон "
      f"{100*gc[0]:.1f}%")
print(f"    топ-29 нейронів:            {st('c29')}   гаус-еталон "
      f"{100*gc[1]:.1f}%")
print(f"    🔒 контроль — той самий сигнал m, топ-8: {st('cs8')}")
print(f"    (в) після нормування на s, топ-8:  {st('c8n')}  топ-29 "
      f"{st('c29n')}")

print("\n(3) ДЕ ЕНЕРГІЯ ПОХИБКИ ЗА НАСИЧЕНІСТЮ ВОРІТ")
print(f"    |t| < 2 :  несе {st('e_lo')} енергії  при {st('f_lo')} нейронів")
print(f"    |t| < 1 :  несе {st('e_lo1')} енергії  при {st('f_lo1')} нейронів")

print("\n(2) ЧИ МОЖНА ЦІЛИТИСЬ БЕЗ ІСТИНИ (відбір лише за замиканням)")
print(f"    топ-25% за МАЛИМ |t|  ⟹ {st('pred25')} енергії")
print(f"    топ-50% за МАЛИМ |t|  ⟹ {st('pred50')} енергії")
print(f"    🔒 контроль випадкові 25%      ⟹ {st('rnd25')}")
print(f"    🔒 контроль топ-25% за БІЛЬШИМ s ⟹ {st('top_s25')}")
