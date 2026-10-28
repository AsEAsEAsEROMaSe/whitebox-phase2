"""Чиста оцінка узагальнення: 400 випадкових розбиттів навпіл ПО МЕРЕЖАХ.
Фіт на половині A (разом із вибором ознаки), вирок на половині B. Дублікатів
немає за побудовою ⟹ на відміну від бутстрепу над LOO, оптимізму немає."""
import numpy as np
d = np.load("/mnt/win_d/llm/White-Box/arena62/logs/wfeat99.npz", allow_pickle=True)
X, y, names = d["X"], d["y"], list(d["names"])
M = X.shape[0]; rng = np.random.default_rng(11)

def split_r2(X, y, tr, te, j=None, sel=True):
    Xt, yt = X[tr], y[tr]
    if sel:
        cc = np.array([abs(np.corrcoef(Xt[:,k], yt)[0,1]) for k in range(X.shape[1])])
        cc = np.nan_to_num(cc); j = int(np.argmax(cc))
    sx = Xt[:, j]
    b = np.cov(sx, yt)[0,1] / max(np.var(sx), 1e-300)
    p = yt.mean() + b * (X[te, j] - sx.mean())
    # R² рахуємо проти середнього ТРЕНУВАЛЬНОЇ половини — чесно позавибірково
    ss = ((y[te]-p)**2).sum(); s0 = ((y[te]-yt.mean())**2).sum()
    return 1.0 - ss/s0, j

J7 = names.index("F7_absT_med")
for tag, kw in (("вибір усередині", dict(sel=True)),
                ("лише F7 (post-hoc)", dict(sel=False, j=J7))):
    r2, picks = [], []
    for _ in range(400):
        p = rng.permutation(M); a, b = p[:M//2], p[M//2:]
        for tr, te in ((a,b), (b,a)):
            v, j = split_r2(X, y, tr, te, **kw); r2.append(v); picks.append(j)
    r2 = np.array(r2)
    print(f"{tag:<20s} 2-fold R²: медіана {np.median(r2):+.4f}  "
          f"середнє {r2.mean():+.4f}  90% [{np.percentile(r2,5):+.4f},"
          f" {np.percentile(r2,95):+.4f}]  P(>0)={float((r2>0).mean()):.3f}")
    if kw.get("sel"):
        from collections import Counter
        c = Counter(picks)
        print("     обрані ознаки:", {names[k]: v for k, v in c.most_common(4)})

# ─── нуль тим самим протоколом (пломба «чужа W»)
null = []
for _ in range(400):
    yp = y[rng.permutation(M)]
    p = rng.permutation(M); a, b = p[:M//2], p[M//2:]
    for tr, te in ((a,b),(b,a)):
        null.append(split_r2(X, yp, tr, te, sel=True)[0])
null = np.array(null)
print(f"{'🔒 ЧУЖА W (нуль)':<20s} 2-fold R²: медіана {np.median(null):+.4f}  "
      f"95-й перц. {np.percentile(null,95):+.4f}  P(>0)={float((null>0).mean()):.3f}")
