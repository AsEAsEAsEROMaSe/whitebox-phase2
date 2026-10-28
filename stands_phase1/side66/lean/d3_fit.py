"""Чи похибка його D3 (власна κ₃ передактивацій, R² 0.998) систематична: регресія ΔD3 = D3_true − D3_his на понейронні ознаки стану
(його D3, mu, var, κ₄_true≈його, D3·mu/var, D3²/var, |D21|-рядкова сума, ...). Передбачення: R² ≤0.3 (шанс 60%), тоді memoryless-поправка D3 мертва."""
import numpy as np, sys
for i in (0, 3):
    his = np.load("/home/trim/wb2lean/his_d21_dump%d.npz" % i); kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i)
    print("== dump", i)
    for l in range(2, 16):
        a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64); E4 = kp["pre_m4"][l].astype(np.float64)
        v = E2 - a*a; D3t = E3 - 3*a*E2 + 2*a**3; k4 = E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v
        D3h = his["d3_%d" % l].astype(np.float64); H = his["his_%d" % l].astype(np.float64)
        d = D3t - D3h; sd = np.sqrt(v); t = a/sd
        feats = {"D3h": D3h, "mu": a, "var": v, "k4": k4, "D3*t": D3h*t, "D3^2/v": D3h**2/v, "rowsumH": np.abs(H).sum(1), "colsumH": np.abs(H).sum(0),
                 "sd^3": sd**3, "t*sd^3": t*sd**3, "k4*t": k4*t, "D3*t^2": D3h*t*t, "phi*sd3": np.exp(-t*t/2)*sd**3, "rowH2": (H*H).sum(1), "HD3": H @ D3h}
        X = np.stack(list(feats.values()), 1); X = np.concatenate([X, np.ones((len(d), 1))], 1)
        def r2(cols):
            Xc = X[:, cols]; sol = np.linalg.lstsq(Xc, d, rcond=None)[0]; return 1 - ((d - Xc@sol)**2).sum()/((d - d.mean())**2).sum()
        names = list(feats); singles = " ".join("%s %.2f" % (nm, r2([k, len(names)])) for k, nm in enumerate(names))
        # крос: фіт на dump0, застосувати на dump3 — окремо нижче; тут in-sample
        print("l=%2d ε(D3) %.3f rel.bias %+.3f | in-sample: %s | ВСІ %.3f" % (l, np.sqrt((d*d).sum()/(D3t*D3t).sum()), d.sum()/np.abs(D3t).sum(), singles, r2(list(range(len(names)+1)))))
