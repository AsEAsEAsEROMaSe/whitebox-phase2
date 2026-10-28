"""Аналіз kmix: деревні ознаки для κ41, κ32 (півот i та j), коеф. з TRAIN дампів, R² на TEST; стеля = corr(A, B) цілі."""
import glob, numpy as np
FILES = sorted(glob.glob("/home/trim/wb2data/kmix_*.npz")); D = {int(f.split("_")[-1][:-4]): np.load(f) for f in FILES}
dumps = sorted(D); TEST = [dumps[-1]]; TRAIN = dumps[:-1]; print("train", TRAIN, "test", TEST)
def sl(dd, h, l):
    g = lambda k: dd["h%d_l%d_%s" % (h, l, k)].astype(np.float64) if h is not None else 0.5*(dd["h0_l%d_%s" % (l, k)].astype(np.float64) + dd["h1_l%d_%s" % (l, k)].astype(np.float64))
    v = dd["h%d_l%d_v" % (h, l)] if h is not None else 0.5*(dd["h0_l%d_v" % l] + dd["h1_l%d_v" % l])
    m = [v[k]/v[0] for k in range(6)]; m2, m3, m4 = m[2], m[3], m[4]
    E11, E21, E12, E31, E22, E41, E32 = (g(k) for k in ("11", "21", "12", "31", "22", "41", "32"))
    X2 = m2[:, None]; Y2 = m2[None, :]; X3 = m3[:, None]; Y3 = m3[None, :]
    k21 = E21; k31 = E31 - 3*X2*E11; k22 = E22 - X2*Y2 - 2*E11**2
    k41 = E41 - 6*X2*E21 - 4*E11*X3; k32 = E32 - 3*X2*E12 - 6*E11*E21 - Y2*X3
    k3 = m3; k4 = m4 - 3*m2**2; vi = X2; vj = Y2
    K3i = k3[:, None]; K3j = k3[None, :]; K4i = k4[:, None]
    F41 = {"k3i·k31/vi": K3i*k31/vi, "k21·k4i/vi": k21*K4i/vi, "k21·k22/vj": k21*k22/vj, "k21T·k31/vj": k21.T*k31/vj}
    F32 = {"k3i·k22/vi": K3i*k22/vi, "k21·k31/vi": k21*k31/vi, "k21T·k4i/vi": k21.T*K4i/vi,
           "k21·k31T/vj": k21*k31.T/vj, "k21T·k22/vj": k21.T*k22/vj, "k3j·k31/vj": K3j*k31/vj}
    return dict(k41=k41, k32=k32), dict(k41=F41, k32=F32)
off = None
for l in [6, 9, 12, 15]:
    print("\n== шар %d" % l)
    for tg in ("k41", "k32"):
        ys = {}; Xs = {}; ceil = []
        for i in dumps:
            (T, F) = sl(D[i], None, l); TA, _ = sl(D[i], 0, l); TB, _ = sl(D[i], 1, l)
            n = T[tg].shape[0]; m = ~np.eye(n, dtype=bool)
            ys[i] = T[tg][m]; Xs[i] = np.stack([f[m] for f in F[tg].values()], 1); names = list(F[tg])
            ceil.append(np.corrcoef(TA[tg][m], TB[tg][m])[0, 1])
        A = np.concatenate([Xs[i] for i in TRAIN]); b = np.concatenate([ys[i] for i in TRAIN]); sol = np.linalg.lstsq(A, b, rcond=None)[0]
        per = []
        for i in dumps:   # коефіцієнти ПО мережі для стабільності
            per.append(np.linalg.lstsq(Xs[i], ys[i], rcond=None)[0])
        yt = np.concatenate([ys[i] for i in TEST]); Xt = np.concatenate([Xs[i] for i in TEST]); r2 = 1 - ((yt - Xt@sol)**2).sum()/((yt-yt.mean())**2).sum()
        sing = []
        for c in range(len(names)):
            s1 = np.linalg.lstsq(A[:, [c]], b, rcond=None)[0]; sing.append(1 - ((yt - Xt[:, [c]]@s1)**2).sum()/((yt-yt.mean())**2).sum())
        per = np.array(per)
        print("  %s: стеля(corr A↔B) %s | R²_test спільний %.3f | коеф %s | розкид коеф. між мережами %s"
              % (tg, " ".join("%.3f" % c for c in ceil), r2, " ".join("%.2f" % v for v in sol), " ".join("%.0f%%" % (100*s) for s in per.std(0)/np.abs(per.mean(0)))))
        print("     одиночні R²_test: " + " | ".join("%s %.3f" % (nm, s) for nm, s in zip(names, sing)))
