"""width101 — БЕЗУМНА ІДЕЯ: ЕКСТРАПОЛЯЦІЯ ПО ШИРИНІ ВСЕРЕДИНІ ОДНІЄЇ МЕРЕЖІ.

МЕХАНІЗМ (зміряний, не вгаданий). Гаусове замикання ТОЧНЕ при n→∞ (NNGP), а
похибка є скінченнорозмірним ефектом `1/n` — це КОРІНЬ, знайдений С97, і закон
по ширині зміряно як **n^(−1.132)**. Досі ми міряли його МІЖ мережами різної
ширини. НІКОЛИ не перевіряли, чи він живе ВСЕРЕДИНІ однієї мережі — на її
власних підматрицях `W[:n',:n']`.

ЧОМУ ЦЕ МОЖЕ ДАТИ БАЛ. Якщо `m_cl(n') − m_true(n')` для спільних нейронів
корельована між різними `n'`, то похибку при n'=256 можна ЕКСТРАПОЛЮВАТИ з
дешевих підмереж (МК на n'=64 коштує (64/256)² = 1/16) і ВІДНЯТИ. Це б'є
прямо у `m` — єдиний живий канал (×22.59, стеля класу ×3.11).

🚨 ГОЛОВНЕ ЗАПЕРЕЧЕННЯ, ЯКЕ ПЕРЕВІРЯЄТЬСЯ ТУТ: якщо переноситься лише
   МАСШТАБ похибки, ідея мертва — `_ALPHA` бере масштаб ДАРОМ (так загинула
   `κ₃`). Тому вимірюємо саме КОРЕЛЯЦІЮ ФОРМИ по спільних нейронах, і окремо
   — те саме ПІСЛЯ зняття спільного множника.

🔒 ПЛОМБИ:
  (1) n'=256 мусить відтворити нашу записану похибку замикання ~0.5%
  (2) шар 0 при будь-якому n' точний ⟹ похибка мусить рости з глибиною
  (3) 🔒 КОНТРОЛЬ «ЧУЖА ПІДМЕРЕЖА»: корелювати похибку n'=128 мережі A з
      похибкою n'=256 мережі B ⟹ мусить дати нуль
  (4) 🔒 МАСШТАБУВАННЯ N: два бюджети МК ⟹ якщо кореляція росте з N, вона
      реальна; якщо стоїть — це шум

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) похибка ляже на степеневий закон по n' з показником −0.8…−1.4      60%
 (б) корeляція ФОРМИ (n'=128 проти n'=256) перевищить +0.3              35%
 (в) 🔒 після зняття спільного множника вона впаде щонайменше вдвічі    70%
 (г) чужа підмережа дасть |corr| < 0.05                                 85%
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
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
WIDTHS = [32, 64, 96, 128, 160, 192, 224, 256]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
print(f"width101 · {NNET} мереж · ширини {WIDTHS} · {NS} семплів", flush=True)


def SCL(n):
    """🚨 ПЕРЕНОРМУВАННЯ. Обрізання W[:n,:n] лишає Var(W)=2/256, тож сума
    n доданків має дисперсію n/256 від потрібної ⟹ міняється НЕ лише
    ширина, а РЕЖИМ ВОРІТ (медіана |t|), а він і є справжня вісь (С99).
    Множник √(256/n) відновлює He-нормування ⟹ міряємо ЧИСТО ширину."""
    return math.sqrt(W_ / n)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure(Wg, n):
    """замикання для підмережі ширини n. He-init ⟹ Var(W)=2/256 незалежно
    від n ⟹ підмережа НЕ перенормовується: це чесно інша мережа тієї ж сім'ї,
    але з меншим `n` при тій самій дисперсії ваг ⟹ масштаб активацій падає,
    а нас цікавить ВІДНОСНА похибка, яка від масштабу не залежить."""
    m = cp.zeros(n, dtype=cp.float64)
    C = cp.eye(n, dtype=cp.float64)
    for i in range(DEPTH):
        w = Wg[i][:n, :n] * SCL(n)
        cpm = w.T @ C @ w
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        mn = mp * Ph + s * ph
        if i == DEPTH - 1:
            return mn
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        Co = cpm * cp.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        C = Co + cp.diag(d - cp.diagonal(Co))
        m = mn


def truth(Wg, n, ns, seed):
    rng = cp.random.default_rng(seed)
    acc = cp.zeros(n, dtype=cp.float64)
    done = 0
    while done < ns:
        nb = min(16384, ns - done)
        z = rng.standard_normal((nb, n), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            p = a @ (Wg[i][:n, :n] * SCL(n)).astype(cp.float32)
            a = cp.maximum(p, 0.0) if i < DEPTH - 1 else p
        acc += cp.maximum(a, 0.0).sum(0, dtype=cp.float64)
        done += nb
    return acc / (2 * ns)


err = {n: [] for n in WIDTHS}
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float64))
          for i in range(DEPTH)]
    for n in WIDTHS:
        mc = closure(Wg, n)
        mt = truth(Wg, n, NS, 777 + net)
        e = mc - mt
        err[n].append((cp.asnumpy(e), cp.asnumpy(mt)))
    print(f"  мережа {net} готова", flush=True)

print(f"\n{'n':>5s}{'віднос.похибка':>16s}{'log-log нахил':>16s}")
lv = []
for n in WIDTHS:
    v = float(np.mean([np.sqrt((e * e).mean() / (t * t).mean())
                       for e, t in err[n]]))
    lv.append(v)
    print(f"{n:>5d}{100*v:>15.4f}%", end="")
    if len(lv) > 1:
        sl = (math.log(lv[-1]) - math.log(lv[0])) / \
             (math.log(n) - math.log(WIDTHS[0]))
        print(f"{sl:>16.3f}")
    else:
        print()
print(f"🔒 (1) n=256 дало {100*lv[-1]:.4f}%  [записано замикання ~0.5%]")

print("\n— 🔑 ЧИ ПЕРЕНОСИТЬСЯ ФОРМА? корeляція похибки по СПІЛЬНИХ нейронах —")
print(f"{'пара':>12s}{'corr сира':>12s}{'corr після':>12s}{'🔒чужа':>10s}")
print(f"{'':12s}{'':12s}{'зняття масшт.':>12s}")
for n in WIDTHS[:-1]:
    cs, ca, cf = [], [], []
    for net in range(NNET):
        e1 = err[n][net][0]
        e2 = err[256][net][0][:n]
        t1 = err[n][net][1]
        t2 = err[256][net][1][:n]
        r1 = e1 / math.sqrt((t1 * t1).mean())
        r2 = e2 / math.sqrt((t2 * t2).mean())
        cs.append(np.corrcoef(r1, r2)[0, 1])
        ca.append(np.corrcoef(r1 - r1.mean(), r2 - r2.mean())[0, 1])
        o = (net + 1) % NNET                  # 🔒 чужа мережа
        e3 = err[256][o][0][:n]
        cf.append(np.corrcoef(r1, e3)[0, 1])
    print(f"{n:>5d}→256{np.mean(cs):>12.4f}{np.mean(ca):>12.4f}"
          f"{np.mean(cf):>10.4f}")
