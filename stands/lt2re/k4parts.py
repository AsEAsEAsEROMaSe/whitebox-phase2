"""k4parts (20.10.2026, white-box-89): передатна функція κ4. Яка частина читання κ4 несе виграш K=4 над k3a?
Кожне читання κ4 дає чотири частини: (4,) n чисел, (3,1) і (2,2) n^2, (2,1,1) n^3. Обнуляємо всі, крім K4_PARTS.
Якщо (4,) і (3,1) тримають виграш, κ4 можна читати за n^3 (лише діагональ B і B·dA), а не n^4.
usage: K4_PARTS=4,31,22,211 k4parts.py <n> <L> <seed> <N_truth>
Пломба: K4_PARTS=4,31,22,211 = повний 2.965e-7; K4_PARTS=none ≈ k3 без κ4 (не k3a: аугментації нема).
Передбачення (до прогону): (4,)+(3,1) тримає ≥ 60% log-виграшу, сам (4,) ≥ 30%; шанс 20%."""

import sys, os, math, time
import numpy as np, torch
torch.set_grad_enabled(False)
sys.path.insert(0, "/home/trim/ext/fc/repo/lab")
import k4full as K4
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import linear_kprop, coerce_input, Kind
from mlp_kprop.wick import WICK_COEF_D
from mlp_kprop.diagslice import zero_repeated

n, L, seed, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4]))
dev, dt = "cuda", torch.float32
rng = np.random.default_rng(seed)
W = [(rng.standard_normal((n, n)) * math.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
z = np.load(os.environ.get("AW_DATA", "data") + f"/kscan_{n}_{L}_{seed}_{N}.npz")
gt = z["gt"]
TYPES = ("path", "star", "three2", "slice4", "slice31", "slice22", "slice211")
KEYS = [(t, a) for t in TYPES for a in ("y", "o")]
birth, CUR, REC = {}, [0], []


PARTS_ON = {"4": (4,), "31": (3, 1), "22": (2, 2), "211": (2, 1, 1)}
KEEP = {PARTS_ON[x] for x in os.environ.get("K4_PARTS", "4,31,22,211").split(",") if x in PARTS_ON}
_orig = K4.SK4._dslices


def _dslices(self, groups):
    out = _orig(self, groups)
    return {p: (t if p in KEEP else torch.zeros_like(t)) for p, t in out.items()}


K4.SK4._dslices = _dslices
mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).to(device=dev, dtype=dt)
for i in range(L):
    mlp.Ws[i].weight.data = torch.from_numpy(W[i].T.copy()).to(device=dev, dtype=dt).contiguous()
Kc = coerce_input({1: torch.zeros(n, device=dev, dtype=dt), 2: torch.eye(n, device=dev, dtype=dt)}, k_max=4, kind=Kind.SIMPLE)
wick = WICK_COEF_D["relu"]
means, t0 = [], time.time()
for l in range(L):
    K4.LAST_LAYER = l == L - 1
    CUR[0] = l
    Kc = linear_kprop(Kc, mlp.Ws[l].weight, k_max=4, set_metric=mlp.init_scale[l])
    Kc = K4.nonlin(Kc, nonlin_wick_coef=wick)
    for gr in getattr(Kc[4], "groups", []):
        birth.setdefault(id(gr.Bb), l)
    means.append(Kc[1].to_tensor().double().cpu().numpy())
    pass  # print(f"layer {l} done t={time.time() - t0:.0f}s mem {torch.cuda.memory_allocated() / 2**30:.2f}G", flush=True)
per = np.mean((np.stack(means) - gt) ** 2, axis=1)
print(f"k4parts {os.environ.get('K4_PARTS', 'all')} n={n} L={L} seed={seed}: final {per[-1]:.4e} t={time.time() - t0:.0f}s", flush=True)
