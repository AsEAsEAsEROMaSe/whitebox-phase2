"""21.10.2026: МНК-коефіцієнти прогнозу c4 (живі нейрони) по шарах з c4fit_n*.npz → /home/trim/wblogs/g4fit_coef.json"""
import numpy as np, json, glob
D = [np.load(f) for f in sorted(glob.glob('/home/trim/wblogs/c4fit_n[0-5].npz'))]
out = {}
for l in range(1, 16):
    Xs, ys = [], []
    for d in D:
        g4, var, D3, mu, ab, d4, s4 = d['F'][l]; t = mu / np.sqrt(var); v2 = var ** 2
        X = np.stack([g4, ab, d4, v2, v2 * t, v2 * t * t, g4 * t, ab * t, v2 * np.exp(-t * t / 2)], 1); m = D3 != 0 if l > 0 else np.ones(1024, bool)
        Xs.append(X[m]); ys.append(d['c4'][l][m])
    out[l] = np.linalg.lstsq(np.concatenate(Xs), np.concatenate(ys), rcond=None)[0].tolist()
json.dump(out, open('/home/trim/wblogs/g4fit_coef.json', 'w'))
print(len(D), 'nets', {l: [round(x, 3) for x in v[:3]] for l, v in out.items() if l in (5, 10, 14)})
