"""chpre_f0L (21.10.2026 f0): D3/g4row ланцюга проти істини ПО ВСІХ шарах (живі) + оракул c3 живих ПО ОДНОМУ шару. d3_live ×1.435 при точному D3 на L12/14 ⟹ виграш на інших шарах.\nПЕРЕДБАЧЕННЯ: corr3 падає на мілких або на останніх шарах; основна частка ×1.435 у 1–3 шарах. Шанс 60%.\nchpre_f0d (21.10.2026 f0): оракул c3 окремо на обрізаних і живих. На живих D3 точний (corr 0.9988, slope 1.015), 99.5% енергії похибки D3 у обрізаних (c3 ≈ −0.02, D3 = 0).\nПЕРЕДБАЧЕННЯ: d3_prun бере більшість з ×1.44. Шанс 55%.\nchpre_f0 (21.10.2026 white-box-f0, на прохання 5f): оракул c4 окремо на обрізаних (індекси >= S_OUT) і необрізаних нейронах.\nПЕРЕДБАЧЕННЯ: g4_live ≈ g4_all (×1.12…1.14), g4_prun ≈ ×1.01, бо обрізаних 96 з 1024 і D3 на них не читається. Шанс 70%.\nchpre2: оракул — справжні c3/c4 передактивацій (МК 2^24) замість D3/g4row ланцюга на наборах шарів; MSE виходу. ПЕРЕДБАЧЕННЯ d3_all ×1.5…×3.\nchpre (20.10.2026 white-box-89). edge14: Еджворт зі СПРАВЖНІМИ γ3,γ4 по нейрону знімає похибку E[ReLU] ×50 (1.0e-3σ → 2e-5σ).
Тут: γ3,γ4 передактивацій, які несе сам ланцюг (D3, g4row з хука), проти справжніх МК 2^24 на шарах 12,14; і Еджворт з γ ланцюга
проти справжнього E[ReLU]. ПЕРЕДБАЧЕННЯ: corr(γ3 ланц, справж) 0.8–0.95, γ4 ланцюга грубий (corr <0.5) ⟹ похибка кроку в γ4. Шанс: 50%.
usage: chpre.py <k> <out.json>"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]; LS = (12, 14); N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_orc", "/home/trim/wb2lean/lt2re/estimator_sb2_orc_f0.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(22000 + k)
S = torch.zeros(16, 4, 1024, dtype=torch.float64, device=dev); sh = {}; n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        p = h @ W[l]
        if l not in sh: sh[l] = p.mean(0)
        x = (p - sh[l]).double()
        S[l, 0] += x.sum(0); S[l, 1] += (x * x).sum(0); S[l, 2] += (x ** 3).sum(0); S[l, 3] += (x ** 4).sum(0)
        h = torch.relu(p)
    n += B
a = (S / n).cpu().numpy(); m1, m2, m3, m4 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
var = m2 - m1 ** 2; c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2
del W, S; torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]
c3t = [cp.asarray(c3[l].astype(np.float32)) for l in range(16)]; c4t = [cp.asarray(c4[l].astype(np.float32)) for l in range(16)]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(D3set=(), G4set=(), part=None, dpart=None):
    mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "D3set": tuple(D3set), "G4set": tuple(G4set), "c3t": c3t, "c4t": c4t, "g4part": part, "d3part": dpart}
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
res = {"k": k, "base": run()}
A = range(1, 16)
mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "D3set": (), "G4set": (), "c3t": c3t, "c4t": c4t, "dump3": {}}
est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41)
dmp = mod.ORAC["dump3"]; mod.ORAC = None
lay = {}
for l in sorted(dmp):
    D3 = cp.asnumpy(cp.asarray(dmp[l][0])).astype(np.float64); nr = int(dmp[l][2])
    g4 = None if dmp[l][1] is None else cp.asnumpy(cp.asarray(dmp[l][1])).astype(np.float64)
    pr = D3 == 0; lv = ~pr; t3 = c3[l]; t4 = c4[l]
    r = {"n_dead": int(pr.sum()), "nr": nr, "corr3": float(np.corrcoef(D3[lv], t3[lv])[0, 1]), "slope3": float(D3[lv] @ t3[lv] / (D3[lv] @ D3[lv])),
         "rel3": float(np.sqrt(np.mean((D3[lv] - t3[lv]) ** 2)) / np.sqrt(np.mean(t3[lv] ** 2)))}
    if g4 is not None:
        r.update({"corr4": float(np.corrcoef(g4[lv], t4[lv])[0, 1]), "slope4": float(g4[lv] @ t4[lv] / (g4[lv] @ g4[lv]))})
    lay[l] = r
    print("net %d L%02d %s" % (k, l, " ".join("%s=%.4g" % kv for kv in r.items())), flush=True)
res["layers"] = lay
for l in range(1, 16):
    res["d3live_L%d" % l] = run((l,), (), None, "live")
    print("net %d d3live_L%02d ×%.4f" % (k, l, res["base"] / res["d3live_L%d" % l]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
