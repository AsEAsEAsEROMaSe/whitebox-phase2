"""С115-ж СТАДІЯ 1: чи фітується `δs/s` з `|t|`, і чи ПЕРЕНОСИТЬСЯ між мережами.

Об'єкт: наша `s` (корінь діагоналі cp) проти ІСТИННОЇ `s` передактивації
        (банк GPU N=2^22, `/home/trim/wblogs/truth115_NN.npz`).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 δ = log(s_true/s_our) СПАДАЄ з |t|  (corr −0.88 уже зміряно)        90%
  P2 фіт 3 параметрами бере >=70% дисперсії ВСЕРЕДИНІ мережі             65%
  P3 коефіцієнти ПЕРЕНОСЯТЬСЯ між мережами (крос-R2 > 0.5 від in-sample) 45%
  P4 форма δ(|t|) МІНЯЄТЬСЯ з глибиною ⟹ потрібен спільний член depth    60%
"""
import numpy as np, pyarrow.parquet as pq, sys, time, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
rows = []
for f in range(8):
    d = np.load("/home/trim/wblogs/truth115_%02d.npz" % f)
    nm = str(d["name"]); k = NAMES.get(nm)
    if k is None: log("НЕМА %s у parquet" % nm); continue
    W = np.asarray(tb["weights"][k].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    MS, SS, TS, VV = closure(W)
    st = d["s"]; mt = d["mp"]
    rows.append(dict(name=nm, s_our=np.array(SS), t_our=np.array(TS),
                     s_tru=st, mp_our=np.array([x for x in (np.array(MS),)][0])*0 ))
    log("%d %s" % (f, nm))

NL = 16
print("\n%-3s %9s %9s %9s %9s" % ("ш", "мед d%", "RMS d%", "corr|t|", "медt"))
D = {}
for i in range(NL):
    dl, tl = [], []
    for r in rows:
        so, st, t = r["s_our"][i], r["s_tru"][i], r["t_our"][i]
        dl.append(np.log(st/so)); tl.append(np.abs(t))
    dl = np.concatenate(dl); tl = np.concatenate(tl)
    D[i] = (dl, tl)
    print("%-3d %9.4f %9.4f %9.4f %9.3f" % (i, np.median(dl)*100,
          np.sqrt((dl*dl).mean())*100, np.corrcoef(dl, tl)[0,1], np.median(tl)))

def basis(t, i):
    """низькопараметрична сім'я з |t|; ТІЛЬКИ величини, відомі даром."""
    return np.stack([np.ones_like(t), t, t*t, np.exp(-0.5*t*t)], 1)

print("\nФІТ по шару: R2 in-sample / LOO по МЕРЕЖАХ")
print("%-3s %8s %8s %8s" % ("ш", "R2 in", "R2 LOO", "коеф"))
COEF = {}
for i in range(NL):
    Xs, Ys = [], []
    for r in rows:
        t = np.abs(r["t_our"][i]); y = np.log(r["s_tru"][i]/r["s_our"][i])
        Xs.append(basis(t, i)); Ys.append(y)
    Xa = np.concatenate(Xs); Ya = np.concatenate(Ys)
    c = np.linalg.lstsq(Xa, Ya, rcond=None)[0]
    r2in = 1.0 - ((Xa@c - Ya)**2).sum()/((Ya-Ya.mean())**2).sum()
    # LOO по мережах
    num = den = 0.0
    for j in range(len(rows)):
        Xtr = np.concatenate([Xs[q] for q in range(len(rows)) if q != j])
        Ytr = np.concatenate([Ys[q] for q in range(len(rows)) if q != j])
        cj = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
        num += ((Xs[j]@cj - Ys[j])**2).sum(); den += ((Ys[j]-Ya.mean())**2).sum()
    COEF[i] = list(map(float, c))
    print("%-3d %8.4f %8.4f  %s" % (i, r2in, 1.0-num/den,
          " ".join("%+.5f" % v for v in c)))
json.dump(COEF, open("/home/trim/wblogs/sig1_coef.json", "w"))
log("ГОТОВО")
