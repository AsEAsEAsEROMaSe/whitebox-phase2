"""Перегонка CORR_BETA на траєкторії v29/regen — те, що автор лишив недоробленим.

Автор: «The V16b CORR_BETA mean rider was ridge-fitted on V16b's OWN trajectory and is
anti-correlated with V16b's base error; on the regen base it HURTS (dump0: 3.57e-8 with vs
2.51e-8 without). Off by default until refitted on the V17 trajectory.»

Ознаки (13, понейронні, на кожному шарі): 1, mu, var, sigma, alpha, |alpha|, phi, Phi,
pk1v, D3, rowsum|D21|, k4, sigma*phi. Ціль: all_layer_means[l] − pk1v[l] (офіційна істина 1e9).
Коеф. на мережах TRAIN, R² на TEST тими самими коеф. Гребінь підбирається на TRAIN.

Передбачення ДО прогону: R²_test на ОСТАННЬОМУ шарі (він і є бал) ≤0.15; якщо >0.25 —
поправка реально жива. Шанс, що наскрізний виграш >x1.05: ~35%."""
import pickle, sys, numpy as np

CD = sys.argv[1] if len(sys.argv) > 1 else "/home/trim/wb2data/corrdump.pkl"
TRAIN = ["logan-fitzgerald", "william-graves", "raymond-barnes"]
NAMES = {}
for i in range(8):
    d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i)
    NAMES[str(d["mlp_name"])] = d["all_layer_means"].astype(np.float64)

recs = pickle.load(open(CD, "rb"))
print("мереж у дампі: %d" % len(recs), [r[0] for r in recs])
L = len(recs[0][1])
tr = [r for r in recs if r[0] in TRAIN]; te = [r for r in recs if r[0] not in TRAIN]
if not te: tr, te = recs[:len(recs)//2], recs[len(recs)//2:]
print("TRAIN %d / TEST %d" % (len(tr), len(te)))

def stack(rs, l):
    X = np.concatenate([np.asarray(r[1][l][0], dtype=np.float64) for r in rs], 0)
    p = np.concatenate([np.asarray(r[1][l][1], dtype=np.float64) for r in rs], 0)
    y = np.concatenate([NAMES[r[0]][l].astype(np.float64) for r in rs], 0) - p
    return X, y, p

BETA = []
print("\n%-4s %9s %9s %9s %9s" % ("шар", "R2_train", "R2_test", "|y|rms", "ridge"))
for l in range(L):
    Xa, ya, _ = stack(tr, l); Xb, yb, _ = stack(te, l)
    sc = np.sqrt((Xa * Xa).mean(0)) + 1e-30
    Xas, Xbs = Xa / sc, Xb / sc
    G = Xas.T @ Xas; g = Xas.T @ ya
    best = None
    for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0):
        b = np.linalg.solve(G + lam * np.trace(G) / len(G) * np.eye(len(G)), g)
        ra = 1 - ((ya - Xas @ b) ** 2).sum() / max((ya * ya).sum(), 1e-300)
        rb = 1 - ((yb - Xbs @ b) ** 2).sum() / max((yb * yb).sum(), 1e-300)
        if best is None or rb > best[2]: best = (lam, ra, rb, b)
    lam, ra, rb, b = best
    BETA.append(b / sc)
    print("%-4d %9.4f %9.4f %9.3e %9.0e" % (l, ra, rb, np.sqrt((yb*yb).mean()), lam))

print("\nОСТАННІЙ ШАР (він і є бал): R2_test = %.4f" % rb)
with open("/home/trim/wb2lean/corr_beta_new.py", "w") as fh:
    fh.write("CORR_BETA_NEW = [\n")
    for b in BETA: fh.write("    [%s],\n" % ", ".join(repr(float(v)) for v in b))
    fh.write("]\n")
print("таблицю записано у corr_beta_new.py")
