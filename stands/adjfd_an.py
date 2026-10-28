"""Аналіз adjfd: видимий R² (у метриці виходу, ||J(R − a c)||²) проти поелементного (||R − a c||²), коеф. з TRAIN 0-3, тест 4-7."""
import glob, numpy as np
TRAIN = [0, 1, 2, 3]; TEST = [4, 5, 6, 7]
D = {int(f.split("_")[-1][:-4]): dict(np.load(f)) for f in sorted(glob.glob("/home/trim/wb2data/adjfd/fd_*.npz"))}
print("дампи:", sorted(D))
lays = sorted({int(k[2:]) for d in D.values() for k in d if k.startswith("vR")})
for l in lays:
    have = [i for i in D if "vR%d" % l in D[i]]
    tr = [i for i in TRAIN if i in have]; te = [i for i in TEST if i in have]
    if not tr or not te: print("l=%d мало дампів" % l); continue
    lin = np.mean([D[i]["lin%d" % l] for i in have]); d3r = np.mean([np.linalg.norm(D[i]["v3R%d" % l]) / np.linalg.norm(D[i]["vR%d" % l]) for i in have])
    names = sorted({k.split("|", 1)[1] for k in D[have[0]] if k.startswith("v%d|" % l)})
    rows = []
    for nm in names:
        k = "v%d|%s" % (l, nm); e = "e%d|%s" % (l, nm)
        a = sum(float(D[i][k] @ D[i]["vR%d" % l]) for i in tr) / sum(float(D[i][k] @ D[i][k]) for i in tr)
        num = sum(float(((D[i]["vR%d" % l] - a * D[i][k])**2).sum()) for i in te); den = sum(float((D[i]["vR%d" % l]**2).sum()) for i in te)
        r2v = 1 - num / den
        ae = sum(D[i][e][0] for i in tr) / sum(D[i][e][1] for i in tr)
        r2e = 1 - sum(D[i][e][2] - 2*ae*D[i][e][0] + ae*ae*D[i][e][1] for i in te) / sum(D[i][e][2] for i in te)
        rows.append((r2v, r2e, nm))
    rows.sort(reverse=True)
    print("\nl=%2d  нелінійність %.4f  ||J r3||/||J R|| %.3f  (train %s test %s)" % (l, lin, d3r, tr, te))
    for r2v, r2e, nm in rows[:10]: print("   %-20s видимий R²_test %+.4f | поелементний %+.4f" % (nm, r2v, r2e))
    # жадібний спільний видимий до 5 членів
    chosen = []
    for step in range(5):
        best = None
        for nm in names:
            if nm in chosen: continue
            cols = chosen + [nm]
            A = np.concatenate([np.stack([D[i]["v%d|%s" % (l, c)] for c in cols], 1) for i in tr]); b = np.concatenate([D[i]["vR%d" % l] for i in tr])
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            At = np.concatenate([np.stack([D[i]["v%d|%s" % (l, c)] for c in cols], 1) for i in te]); bt = np.concatenate([D[i]["vR%d" % l] for i in te])
            r = 1 - float(((bt - At @ sol)**2).sum() / (bt**2).sum())
            if best is None or r > best[0]: best = (r, nm)
        chosen.append(best[1]); print("   жадібний +%-18s видимий R²_test %+.4f" % (best[1], best[0]))
