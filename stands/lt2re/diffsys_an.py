import numpy as np
Z = {k: np.load("/home/trim/wblogs/diffsys_n%d.npz" % k) for k in (0, 1, 2, 3)}
LS = range(6, 15)
def get(k, l):
    z = Z[k]; mu, sd, D3, g4, c3, c4 = (z[n % l] for n in ("mu%d", "sd%d", "D3_%d", "g4_%d", "c3_%d", "c4_%d"))
    live = np.abs(D3) > 1e-12
    return dict(t=(mu / sd)[live], g3c=(D3 / sd ** 3)[live], g3t=(c3 / sd ** 3)[live], g4c=(g4 / sd ** 4)[live], g4t=(c4 / sd ** 4)[live],
                r21=(z["e21row%d" % l] / sd ** 3)[live] if "e21row%d" % l in z else None, rel21=float(z["e21rel%d" % l][0]) if "e21rel%d" % l in z else np.nan)
print("(1) рівень, середнє по 4 мережах: шар | D3: rel, нахил ланц/істина, corr | g4: rel, нахил, corr | D21 rel")
for l in LS:
    r = []
    for k in Z:
        d = get(k, l)
        for a, b in (("g3c", "g3t"), ("g4c", "g4t")):
            x, y = d[a], d[b]; r.append((np.sqrt(np.mean((x - y) ** 2) / np.mean(y ** 2)), x @ y / (y @ y), np.corrcoef(x, y)[0, 1]))
        r.append((d["rel21"],))
    R = np.array([q[:3] for i, q in enumerate(r) if i % 3 != 2]).reshape(4, 2, 3).mean(0); r21 = np.nanmean([q[0] for i, q in enumerate(r) if i % 3 == 2])
    print("  L%2d | D3 %.3f %.3f %.4f | g4 %.3f %.3f %.4f | D21 %.3f" % (l, *R[0], *R[1], r21))
print("(2) спільний зсув mean(e)/rms(e) (e = ланц − істина, нормовано σ^k), по мережах:")
for nm, a, b in (("e3", "g3c", "g3t"), ("e4", "g4c", "g4t")):
    print("  %s " % nm + " ".join("L%d %s" % (l, "/".join("%+.2f" % (np.mean(get(k, l)[a] - get(k, l)[b]) / np.sqrt(np.mean((get(k, l)[a] - get(k, l)[b]) ** 2))) for k in Z)) for l in (6, 10, 14)))
print("(3) частка дисперсії e, яку знімає функція (крос між мережами): t-біни | (t, γ_істина) 2D | лінійна по γ_істина | лінійна по γ_ланц")
def frac(feat, tr, te, nb=20, kind="bin"):
    num = den = 0.0
    for kte in Z:
        Xtr = np.concatenate([feat(k)[0] for k in Z if k != kte]); Ytr = np.concatenate([feat(k)[1] for k in Z if k != kte])
        X, Y = feat(kte)
        if kind == "bin":
            q = np.quantile(Xtr, np.linspace(0, 1, nb + 1)); q[0], q[-1] = -1e9, 1e9
            bt = np.clip(np.searchsorted(q, Xtr) - 1, 0, nb - 1); m = np.array([Ytr[bt == i].mean() for i in range(nb)])
            P = m[np.clip(np.searchsorted(q, X) - 1, 0, nb - 1)]
        elif kind == "bin2":
            nb2 = 8
            qa = np.quantile(Xtr[:, 0], np.linspace(0, 1, nb2 + 1)); qa[0], qa[-1] = -1e9, 1e9
            qb = np.quantile(Xtr[:, 1], np.linspace(0, 1, nb2 + 1)); qb[0], qb[-1] = -1e9, 1e9
            ia = lambda v: np.clip(np.searchsorted(qa, v) - 1, 0, nb2 - 1); ib = lambda v: np.clip(np.searchsorted(qb, v) - 1, 0, nb2 - 1)
            cell = ia(Xtr[:, 0]) * nb2 + ib(Xtr[:, 1]); m = np.array([Ytr[cell == i].mean() if (cell == i).any() else 0 for i in range(nb2 * nb2)])
            P = m[ia(X[:, 0]) * nb2 + ib(X[:, 1])]
        else:
            A = np.stack([Xtr, np.ones_like(Xtr)], 1); c, *_ = np.linalg.lstsq(A, Ytr, rcond=None); P = c[0] * X + c[1]
        num += np.sum((Y - P) ** 2); den += np.sum((Y - Y.mean()) ** 2)
    return 1 - num / den
for nm, a, b in (("e3", "g3c", "g3t"), ("e4", "g4c", "g4t")):
    row = []
    for l in (6, 10, 14):
        e = lambda k: get(k, l)[a] - get(k, l)[b]
        row.append("L%d %.3f | %.3f | %.3f | %.3f" % (l, frac(lambda k: (get(k, l)["t"], e(k)), 0, 0),
                   frac(lambda k: (np.stack([get(k, l)["t"], get(k, l)[b]], 1), e(k)), 0, 0, kind="bin2"),
                   frac(lambda k: (get(k, l)[b], e(k)), 0, 0, kind="lin"), frac(lambda k: (get(k, l)[a], e(k)), 0, 0, kind="lin")))
    print("  %s  " % nm + "   ".join(row))
print("(4) звʼязок каналів: corr(e3, e4) і corr(e3, рядкова сума похибки D21 / σ³), по мережах:")
for l in (6, 10, 14):
    s1 = "/".join("%+.2f" % np.corrcoef(get(k, l)["g3c"] - get(k, l)["g3t"], get(k, l)["g4c"] - get(k, l)["g4t"])[0, 1] for k in Z)
    s2 = "/".join("%+.2f" % np.corrcoef(get(k, l)["g3c"] - get(k, l)["g3t"], get(k, l)["r21"])[0, 1] for k in Z if get(k, l)["r21"] is not None)
    print("  L%d corr(e3,e4) %s   corr(e3, ΣE21) %s" % (l, s1, s2))
print("(5) профіль e3 і e4 по t (середнє по 4 мережах, ш14; квантилі t):")
X = np.concatenate([get(k, 14)["t"] for k in Z]); E3 = np.concatenate([get(k, 14)["g3c"] - get(k, 14)["g3t"] for k in Z]); E4 = np.concatenate([get(k, 14)["g4c"] - get(k, 14)["g4t"] for k in Z])
G3 = np.concatenate([get(k, 14)["g3t"] for k in Z]); G4 = np.concatenate([get(k, 14)["g4t"] for k in Z])
q = np.quantile(X, np.linspace(0, 1, 9))
for i in range(8):
    m = (X >= q[i]) & (X <= q[i + 1])
    print("  t %+.2f…%+.2f  e3 %+.4f (γ3 %+.3f, rms e3 %.4f)  e4 %+.4f (γ4 %+.3f)" % (q[i], q[i + 1], E3[m].mean(), G3[m].mean(), np.sqrt(np.mean(E3[m] ** 2)), E4[m].mean(), G4[m].mean()))
