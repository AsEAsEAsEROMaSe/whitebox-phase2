"""21.10.2026 white-box-f9 (на прохання white-box-89): чи псує triple-частину c3 передактивації (D3 ланцюга) обрізання
факторизованого K3. Конфіг ланцюга задається env (ранги/обрізання поза бюджетом); частини справжнього c3 (diag, pair, triple)
беруться з /home/trim/wblogs/c3arr_n<k>_L<l>.npz (МК c3parts3, незалежна від конфігу).
Міра: r = D3 - diag - pair (справжні) регресується на triple: beta_t (база ≈0.73) і rms(r - beta_t*triple)/sd(triple) (шум ν).
Плюс raw MSE виходу ланцюга проти істини корпусу.
ПЕРЕДБАЧЕННЯ: без обрізання (Y_EPS=0, NO_CONFINE, PRUNE=0) beta_t 0.73 -> 0.85…0.95, ν падає на третину; до 1.0 не дійде,
бо частину triple ланцюг не несе за побудовою (K3 факторизований через хаби). Шанс beta_t ≥ 0.95 і ν ≤ 0.3: ~20%.
usage: trip_rank.py <k> <tag> <out.jsonl>"""
import json, os, sys, types, importlib.util, math, time
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1)
k, TAG, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]; LS = (12, 14)
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
Wg = [cp.asarray(x.numpy()) for x in w]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": []}
t0 = time.time()
p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
wall = time.time() - t0
pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
res = {"k": k, "tag": TAG, "raw": float(np.mean((p[-1] - m_true[-1]) ** 2)), "wall": wall,
       "env": {e: os.environ[e] for e in os.environ if e.startswith(("V2", "V3", "V1", "AW_", "SB"))}}
for l in LS:
    a = np.load("/home/trim/wblogs/c3arr_n%d_L%d.npz" % (k, l))
    c3, diag, pair, trip = a["c3"], a["diag"], a["pair"], a["triple"]
    D3 = pre[l][2]
    r = D3 - diag - pair
    bt = float(r @ trip / (trip @ trip))
    nu = float(np.sqrt(np.mean((r - bt * trip) ** 2)) / trip.std())
    X = np.stack([pair, trip, np.ones_like(trip)], 1); beta, *_ = np.linalg.lstsq(X, D3 - diag, rcond=None)
    res[str(l)] = {"corr_D3_c3": float(np.corrcoef(D3, c3)[0, 1]), "beta_t": bt, "nu": nu,
                   "corr_r_trip": float(np.corrcoef(r, trip)[0, 1]), "joint_pair_trip_1": beta.tolist(), "nu_joint": float(np.sqrt(np.mean((D3 - diag - X @ beta) ** 2)) / trip.std()),
                   "chk_D3_vs_saved": float(np.max(np.abs(D3 - a["D3"])) / np.max(np.abs(a["D3"]))) if TAG == "base" else None}
print(json.dumps(res), flush=True)
open(OUT, "a").write(json.dumps(res) + "\n")
