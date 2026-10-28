"""LOO по дампах: на кожному L фіт γ4 ≈ c + a·гаус + b·κ3 на чутливих 25% (одиниці γ4), передбачення для дампу поза фітом.
Вихід k4pred_d{i}.npy [NL,n] у одиницях κ4; рядок L=0 = 0 (p_0 гаусова). Друкує похибку γ4 на чутливих: rms(pred−true)."""
import numpy as np, sys
SUF = sys.argv[1] if len(sys.argv) > 1 else ''   # '' або 'chain_'
D = list(range(8)); R = {i: np.load('/home/trim/wblogs/k4pred_raw_%sd%d.npz' % (SUF, i)) for i in D}
NL, n = R[0]['kt'].shape
def feats(i, L):
    r = R[i]; s, t = r['s'][L], r['t'][L]; v2 = s**4; b = np.abs(t**2-1)*np.exp(-t**2/2)/(24*s**3); sel = np.argsort(-b*v2)[:n//4]
    X = np.stack([np.ones(n), r['gs'][L]/v2, r['k3'][L]/v2], 1); y = r['kt'][L]/v2
    return X, y, sel, v2
out = {i: np.zeros((NL, n)) for i in D}   # L=0: p_0 гаусова, κ4=0 точно
print(' L | rms похибки γ4 на чутл.: гаус+κ3 (LOO)   лише гаус   лише рівень | sd істинного')
for L in range(1, NL):
    e3, eg, e0, sd = [], [], [], []
    for i in D:
        tr = [j for j in D if j != i]
        Xa = np.concatenate([feats(j, L)[0][feats(j, L)[2]] for j in tr]); ya = np.concatenate([feats(j, L)[1][feats(j, L)[2]] for j in tr])
        X, y, sel, v2 = feats(i, L)
        c3 = np.linalg.lstsq(Xa, ya, rcond=None)[0]; cg = np.linalg.lstsq(Xa[:, :2], ya, rcond=None)[0]
        p3 = X @ c3; out[i][L] = p3*v2
        e3.append(np.sqrt(((p3-y)[sel]**2).mean())); eg.append(np.sqrt(((X[:, :2]@cg-y)[sel]**2).mean())); e0.append(np.sqrt(((ya.mean()-y)[sel]**2).mean())); sd.append(y[sel].std())
    print('%2d | %.4f   %.4f   %.4f | %.4f' % (L, np.mean(e3), np.mean(eg), np.mean(e0), np.mean(sd)))
for i in D: np.save('/home/trim/wblogs/k4pred_%sd%d.npy' % (SUF, i), out[i])
