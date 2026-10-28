"""mse89 — ПРЯМИЙ ВИМІР БАЛУ: sub97 проти sub99 на 48 мережах із істиною.

КРОС-ЦІЛЬ ОБОВ'ЯЗКОВА: істина шумна (`a_true` 4.84e-07 = 24% робочого MSE),
тож `MSE = mean((pred−T1)·(pred−T2))` — незміщена оцінка БЕЗ шуму істини,
бо `E[(p−T1)(p−T2)] = (p−μ)² коли T1,T2 незалежні й незміщені.

ЩО ПЕРЕВІРЯЄМО:
  1. sub99 має БІЛЬШЕ семплів за той самий `C` ⟹ MSE мусить упасти на ~3%.
  2. Холецьке вибілювання не зіпсувало контрольні змінні (це НЕ бітова заміна).

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
  MSE(99)/MSE(97) = 0.970 ± 0.015 (шанс 70%). Якщо вийде >1.00 — Холецьке
  вибілювання щось ламає, і його треба відкотити, лишивши самі теги.
  🚨 48 мереж — це МЕЖА розсуду (закон: вирок ≥24, сабміти ≥60), тому
  дивимось і на ПАРНУ статистику (та сама мережа обома методами), яка
  прибирає розкид між мережами і бачить набагато менший ефект.
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/win_d/llm/White-Box/whest-starterkit")
import flopscope as flops                                    # noqa: E402
import flopscope.numpy as fnp                                # noqa: E402

B = 2.72e11
W = np.load("fs_w.npy", mmap_mode="r")
seeds = np.load("fs_w_seeds.npy")
T = np.load("/mnt/win_d/llm/White-Box/bank_s53/truth2.npz")
tmap = {int(s): i for i, s in enumerate(T["seeds"])}
idxs = [i for i, s in enumerate(seeds) if int(s) in tmap]
print(f"мереж із істиною: {len(idxs)}")


class MLP:
    def __init__(self, ws):
        self.weights = ws
        self.width = ws[0].shape[0]


def load(i):
    return MLP([fnp.asarray(np.ascontiguousarray(W[i, k]).astype(np.float32))
                for k in range(W.shape[1])])


import estimator_sub97 as E97                                # noqa: E402
import estimator_sub99 as E99                                # noqa: E402

e97 = E97.Estimator.__new__(E97.Estimator)
e99 = E99.Estimator.__new__(E99.Estimator)

rows = []
for j, i in enumerate(idxs):
    mlp = load(i)
    ti = tmap[int(seeds[i])]
    t1, t2 = T["T1"][ti].astype(np.float64), T["T2"][ti].astype(np.float64)
    rec = {}
    for tag, est in (("97", e97), ("99", e99)):
        with flops.BudgetContext(flop_budget=10**13) as c:
            p = np.asarray(est.predict(mlp, int(B)))[-1].astype(np.float64)
        rec[tag] = (float(np.mean((p - t1) * (p - t2))),
                    c.flops_used / B + 1e11 * c.residual_wall_time_s / B)
    rows.append(rec)
    if j % 8 == 0:
        print(f"  {j:>3d}/{len(idxs)}  97={rec['97'][0]:.4e}  99={rec['99'][0]:.4e}"
              f"  ×{rec['99'][0] / rec['97'][0]:.4f}")

m97 = np.array([r["97"][0] for r in rows])
m99 = np.array([r["99"][0] for r in rows])
c97 = np.array([r["97"][1] for r in rows])
c99 = np.array([r["99"][1] for r in rows])

print()
print("=" * 76)
print(f"РЕЗУЛЬТАТ на {len(rows)} мережах (крос-ціль T1×T2)")
print("=" * 76)
print(f"  MSE  sub97 = {m97.mean():.5e}    sub99 = {m99.mean():.5e}"
      f"   ×{m99.mean() / m97.mean():.4f}")
print(f"  C/B  sub97 = {c97.mean():.5f}      sub99 = {c99.mean():.5f}")
s97 = m97.mean() * max(0.1, c97.mean())
s99 = m99.mean() * max(0.1, c99.mean())
print(f"  БАЛ  sub97 = {s97:.4e}      sub99 = {s99:.4e}   ×{s97 / s99:.4f}")

d = m99 - m97
se = d.std(ddof=1) / np.sqrt(len(d))
print(f"\n  ПАРНА статистика: Δ = {d.mean():+.4e} ± {se:.4e}"
      f"   (t = {d.mean() / se:+.2f})")
print(f"  краще у {int((d < 0).sum())}/{len(d)} мереж")
print(f"\n  🎯 прогноз до запуску був ×0.970±0.015 ⟹ "
      f"{'ЗБІГСЯ' if abs(m99.mean() / m97.mean() - 0.970) < 0.02 else 'НЕ ЗБІГСЯ'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/mse89.npz",
         m97=m97, m99=m99, c97=c97, c99=c99)
