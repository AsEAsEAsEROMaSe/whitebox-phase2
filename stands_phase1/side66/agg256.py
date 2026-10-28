"""agg256 — зшиває шарди calib256 і рахує пломбу №1 на всіх мережах разом.
🚨 LOO-α мусить рахуватись на ПОВНОМУ зрізі, а не пошардово (закон: на зрізі з
N мереж α по N−1; при малому N база штучно слабка ×1.618)."""
import glob
import math
import sys

import numpy as np

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "97ba0e9f-c49d-48e8-8d0a-5a2ae0c00729/scratchpad")
from calib99 import loo_alpha_err, VARIANTS                   # noqa: E402

T, S3, SA = [], [], []
P = {v: [] for v, _ in VARIANTS}
for f in sorted(glob.glob(sys.argv[1])):
    d = np.load(f)
    T.append(d["T"])
    S3.append(d["seal3"])
    SA.append(d["seal_align"])
    for j, (v, _) in enumerate(VARIANTS):
        P[v].append(d[f"P{j}"])
T = list(np.concatenate(T))
S3 = np.concatenate(S3)
SA = np.concatenate(SA)
for v in P:
    P[v] = list(np.concatenate(P[v]))

print("=" * 94)
print(f"🔒 ПЛОМБА №1 · n = 256 · {len(T)} мереж · оракул N=1e8 · істина N=1e9")
print("=" * 94)
print(f"🔒P3 повний оракул (мусить бути 0): {S3.max():.3e}")
print(f"🔒 узгодженість двох джерел істини: {100*float(np.median(SA)):.4f}%")
raw = float(np.median([float(np.linalg.norm(p - t) / np.linalg.norm(t))
                       for p, t in zip(P["база"], T)]))
print(f"🔒P5 сира база БЕЗ α: {100*raw:.4f}%   (anchor97: 0.9138%)")
base_med, base_rms, be = loo_alpha_err(P["база"], T)
TARGET = {"оракул m": 0.912, "оракул діаг C": 0.243,
          "оракул позадіаг C": 0.634, "оракул C повна": 0.725,
          "оракул m + C": 20.97}
print(f"\n{'варіант':>20s}{'медіана':>11s}{'RMS':>11s}"
      f"{'k(мед)':>10s}{'k(RMS)':>10s}{'ціль С88':>11s}{'промах':>9s}"
      f"{'знак':>7s}{'парно':>8s}")
for name, chan in VARIANTS:
    med, rms, e = loo_alpha_err(P[name], T)
    km, kr = base_med / max(med, 1e-300), base_rms / max(rms, 1e-300)
    tg = TARGET.get(name)
    s1 = f"{tg:.3f}" if tg else "—"
    s2 = f"×{max(km,tg)/min(km,tg):.2f}" if tg else "—"
    sg = ("—" if not tg else
          ("✅" if (km > 1) == (tg > 1) else "❌"))
    # парний знаковий тест: скільки мереж покращились
    w = int(np.sum(e < be)) if chan else 0
    pr = f"{w}/{len(e)}" if chan else "—"
    print(f"{name:>20s}{100*med:>10.4f}%{100*rms:>10.4f}%"
          f"{km:>10.3f}{kr:>10.3f}{s1:>11s}{s2:>9s}{sg:>7s}{pr:>8s}")
