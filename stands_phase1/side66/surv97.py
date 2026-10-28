"""surv97 — ЧИ ВИЖИВАЄ ВУЗЬКА ГЛИБОКА МЕРЕЖА. ЦЕ ВИРІШУЄ ДОЛЮ ІДЕЇ ЮЗЕРА
ПРО ВУЗЬКІ МЕРЕЖІ РАНІШЕ, НІЖ БУДЬ-ЯКИЙ ЗАКОН `1/n`.

Перший прогін `narrow97` дав ВИХІД РІВНО НУЛЬ на n=1 і n=2 (три різні жереби).
Це не збій стенду: мережа без біасів із ReLU при малій ширині ВМИРАЄ —
конус напрямків, що доживають до виходу, стискається в нуль.
[ЦИТАТА finding-s2-collapse-is-lyapunov] «PR 165 → 2.4», і С90:
«без мінусів мережа ТОЧНО лінійна; мінуси розмірність СТВОРЮЮТЬ».
⟹ при малому `n` мінусів замало, щоб утримати конус.

🎯 ЩО МІРЯЄМО: `P(вижити)` і масштаб виходу проти ширини, глибина 32 СКРІЗЬ.
Це визначає, з якого `n` ідея «вузька мережа з точною істиною» взагалі має
об'єкт. Нижче цього `n` вимірювати нема чого — цілі не існує.

ПЕРЕДБАЧЕННЯ (до прогону): поріг виживання ляже між n=4 і n=16,
і при n=2 виживає <10% мереж.                                    шанс 65%
"""
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402

DEPTH = 32


def make(n, seed):
    r = np.random.default_rng(seed)
    return [r.standard_normal((n, n)) * math.sqrt(2.0 / n)
            for _ in range(DEPTH)]


print("surv97 — ЧИ ВИЖИВАЄ ВУЗЬКА ГЛИБОКА МЕРЕЖА (He, без біасів, L=32)")
print("=" * 78)
print(f"{'n':>4s}{'мереж':>7s}{'вижило':>9s}{'P(вижити)':>12s}"
      f"{'медіана ‖вихід‖':>18s}{'глиб. смерті (мед)':>20s}")
rng = np.random.default_rng(7)
for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128):
    NM = 200 if n <= 16 else 50
    alive = 0
    mags = []
    deaths = []
    for k in range(NM):
        W = make(n, 10000 + 97 * n + k)
        z = rng.standard_normal((2048, n))
        h = z
        dead_at = None
        for L in range(DEPTH):
            h = np.maximum(h @ W[L], 0.0)
            if dead_at is None and not np.any(h > 0):
                dead_at = L
        m = h.mean(0)
        nrm = float(np.sqrt((m * m).mean()))
        if nrm > 0:
            alive += 1
            mags.append(nrm)
        else:
            deaths.append(DEPTH if dead_at is None else dead_at)
    med = float(np.median(mags)) if mags else 0.0
    dm = f"{np.median(deaths):.0f}" if deaths else "—"
    print(f"{n:>4d}{NM:>7d}{alive:>9d}{alive/NM:>12.3f}{med:>18.3e}{dm:>20s}",
          flush=True)

print("\n" + "-" * 78)
print("КОНТРОЛЬ ГЛИБИНИ: при якій глибині вмирає мережа ширини n (медіана)")
print(f"{'n':>4s}" + "".join(f"{f'L={L}':>9s}" for L in (4, 8, 16, 24, 32)))
for n in (2, 3, 4, 6, 8, 16):
    row = []
    for LMAX in (4, 8, 16, 24, 32):
        alive = 0
        for k in range(120):
            W = make(n, 55000 + 131 * n + k)
            z = rng.standard_normal((1024, n))
            h = z
            for L in range(LMAX):
                h = np.maximum(h @ W[L], 0.0)
            if float(np.abs(h.mean(0)).max()) > 0:
                alive += 1
        row.append(f"{alive/120:>9.3f}")
    print(f"{n:>4d}" + "".join(row), flush=True)
