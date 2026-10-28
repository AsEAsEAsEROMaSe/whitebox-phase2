"""acct89b — sub97 проти sub99: ФЛОПИ + ЧИСЛОВА ТОТОЖНІСТЬ ВИХОДУ.

Дві пломби, без яких хід не можна нести на грейдер:
  P1  вихід sub99 мусить збігтися з sub97 до fp32 (це та сама математика,
      лише інший запис згорток) — rel < 1e-5.
  P2  `as_symmetric` ВАЛІДУЄ дані; якщо `C` у fp32 виявиться несиметричною
      бодай на 1 ulp, буде SymmetryError і сабміт УПАДЕ. Гоняємо на РЕАЛЬНИХ
      вагах із банку, а не на синтетиці.

ПЕРЕДБАЧЕННЯ (шанс 80%): P1 пройде бітово (0.00e+00), бо einsum із тегом
рахує половину і дзеркалить ⟹ результат бітово симетричний ⟹ і далі теж.
P2 — головний ризик; якщо впаде, весь хід мертвий.
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/win_d/llm/White-Box/whest-starterkit")
import flopscope as flops                                    # noqa: E402
import flopscope.numpy as fnp                                # noqa: E402

B = 2.72e11
W = np.load("fs_w.npy", mmap_mode="r")   # (100, 32, 256, 256) float16


class MLP:
    def __init__(self, ws):
        self.weights = ws
        self.width = ws[0].shape[0]


def load(idx):
    return MLP([fnp.asarray(np.ascontiguousarray(W[idx, i]).astype(np.float32))
                for i in range(W.shape[1])])


import estimator_sub97 as E97                                # noqa: E402
import estimator_sub99 as E99                                # noqa: E402

e97 = E97.Estimator.__new__(E97.Estimator)
e99 = E99.Estimator.__new__(E99.Estimator)

print("=" * 76)
print("sub97 vs sub99 НА РЕАЛЬНИХ МЕРЕЖАХ БАНКУ")
print("=" * 76)
print(f"{'мережа':>8s} {'F(97)':>16s} {'F(99)':>16s} {'×':>7s} {'|Δвихід|/max':>14s}")
tot97 = tot99 = 0
for idx in (0, 1, 2, 3):
    try:
        mlp = load(idx)
    except Exception as ex:                                  # noqa: BLE001
        print(f"  net{idx}: НЕ ЗАВАНТАЖИЛАСЬ — {type(ex).__name__}: {str(ex)[:40]}")
        continue
    with flops.BudgetContext(flop_budget=10**13) as c1:
        o1 = np.asarray(e97.predict(mlp, int(B)))
    try:
        with flops.BudgetContext(flop_budget=10**13) as c2:
            o2 = np.asarray(e99.predict(mlp, int(B)))
    except Exception as ex:                                  # noqa: BLE001
        print(f"  net{idx}: 🚨 sub99 ВПАВ — {type(ex).__name__}: {str(ex)[:60]}")
        continue
    rel = np.abs(o2 - o1).max() / max(np.abs(o1).max(), 1e-30)
    tot97 += c1.flops_used
    tot99 += c2.flops_used
    print(f"{idx:>8d} {c1.flops_used:>16,d} {c2.flops_used:>16,d} "
          f"{c2.flops_used / c1.flops_used:>7.4f} {rel:>14.3e}")

if tot97:
    n = 4
    print(f"\n   середнє F/B:  sub97 {tot97 / n / B:.5f}   sub99 {tot99 / n / B:.5f}")
    save = (tot97 - tot99) / n
    print(f"   економія = {save:,.0f} флопів = {100 * save / B:.4f}% B")
    samp = tot97 / n * 0.915
    g = save / samp
    print(f"   ⟹ приріст семплів +{100 * g:.2f}%")
    mse = 1.542e-06 / (1 + g) + 3.27e-07
    print(f"   ⟹ MSE {mse:.4e} ⟹ бал {mse * 0.1:.4e}  (×{1.873e-07 / (mse * 0.1):.4f})")
    print(f"\n   🎯 _TARGET_UTIL: 0.0875 → {0.0875 + save / B:.5f}")
