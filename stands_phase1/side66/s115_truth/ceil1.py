"""С115-к: ЧЕСНА СТЕЛЯ ДВОМОМЕНТНОГО КЛАСУ. Оракул із НЕЗАЛЕЖНОЇ вибірки.

ПОМИЛКА, ЯКУ ВИПРАВЛЯЄМО: `neuroC`/`early100` вкидали істинне `m` з
`all_layer_means` І міряли проти `all_layer_means[15]` — ОДИН прогін на 400k
⟹ спільний шум ⟹ вкинута істина частково ВІДТВОРЮВАЛА шум цілі.
🔒 Тепер: вкидаємо з ПОЛОВИНИ A банку (2^22), міряємо проти ПОЛОВИНИ B.
   Половини незалежні за побудовою; шум кожної 0.0144%.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 чесна стеля ГІРША за записану ×11.1 (шум був на користь оракула)   65%
  P2 чесна стеля КРАЩА (шум лише псував)                                20%
  P3 різниця <20% ⟹ вирок про клас стоїть                               30%
  P4 стара «база 0.1269%» теж завищена (шум цілі) ⟹ ~0.118%             80%
"""
import numpy as np, pyarrow.parquet as pq, glob, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def run(W, T, mode, ks=None):
    """mode: base | m (вкидаємо істинне m) — дисперсія ЛИШАЄТЬСЯ нашою (cons)."""
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd0 = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m       # 🔑 дисперсія НАША завжди
        if mode == "m" and (ks is None or i in ks): m = T[i].copy()
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd0 - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
MODES = [("base", None), ("m", None), ("m", tuple(range(8,16))), ("m", tuple(range(0,8)))]
acc = {}; 
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES[nm]].as_py(), dtype=np.float64).reshape(16,1024,1024)
    Tds = np.asarray(tb["all_layer_means"][NAMES[nm]].as_py(), dtype=np.float64).reshape(16,1024)
    A, B = d["m_half"][:, 0, :], d["m_half"][:, 1, :]      # (16,1024) кожна
    for md, ks in MODES:
        key = (md, "усі" if ks is None else ("глиб" if 8 in ks else "мілк"))
        # 🔒 ЧЕСНО: вкидаємо A, міряємо проти B (і навпаки, усереднюємо)
        acc.setdefault(("чесно",)+key, []).append(
            0.5*(cerr(run(W, A, md, ks), B[15]) + cerr(run(W, B, md, ks), A[15])))
        # 🚨 СТАРА СХЕМА: вкидаємо датасет, міряємо проти датасету (спільний шум)
        acc.setdefault(("стара",)+key, []).append(cerr(run(W, Tds, md, ks), Tds[15]))
    log(nm)
print("\n%-7s %-5s %-5s %10s %8s" % ("схема", "режим", "шари", "похибка%", "×база"))
for scheme in ("чесно", "стара"):
    b = np.median(acc[(scheme, "base", "усі")])
    for md, ks in MODES:
        key = (scheme, md, "усі" if ks is None else ("глиб" if 8 in ks else "мілк"))
        v = np.median(acc[key])
        print("%-7s %-5s %-5s %10.4f %8.3f" % (scheme, key[1], key[2], v*100, b/v))
json.dump({str(k): list(map(float,v)) for k,v in acc.items()}, open("/home/trim/wblogs/ceil1.json","w"))
log("ГОТОВО")
