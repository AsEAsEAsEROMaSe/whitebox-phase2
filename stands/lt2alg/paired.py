"""21.10.2026 b9: парне порівняння двох result.json на 48 відкладених мережах (ratio корекції по мережах)."""
import json, sys, math
a, b = (json.load(open(f)) for f in sys.argv[1:3])
ra = [c / r for r, c in zip(a["mini_raw"], a["mini_corr"])]; rb = [c / r for r, c in zip(b["mini_raw"], b["mini_corr"])]
d = [math.log(y / x) for x, y in zip(ra, rb)]
n = len(d); m = sum(d) / n; s = (sum((x - m) ** 2 for x in d) / (n - 1)) ** .5
win = sum(1 for x in d if x < 0)
print("pooled A %.5f B %.5f | per-net geo B/A x%.5f  t=%.2f  B better on %d/%d" % (a["holdout_ratio_best"], b["holdout_ratio_best"], math.exp(m), m / (s / n ** .5), win, n))
