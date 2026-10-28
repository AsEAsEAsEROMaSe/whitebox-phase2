"""k4pair (20.10.2026, white-box-89, ідея юзера «трюк Гаусса 1+100»): чи гасяться діаграми K=4 парами?
Кожен тип народження (path, star, three2, slice4, slice31, slice22, slice211) іде окремою групою (AW_SPEC).
На кожному повному читанні κ4 (SK4._dslices по всіх групах) внесок кожного ключа (тип, молода вік<=1 | стара) у
кожну частину (4,), (3,1), (2,2), (2,1,1) рахується окремо; пишемо |сума| / сума|внесків| і матрицю косинусів.
Результат ланцюга не змінюється (сума та сама).
usage: k4pair.py <n> <L> <seed> <N_truth> <out.npz>
Пломба: final MSE = повний K=4 (2.964e-7 на мережі 1).
Передбачення (до прогону): гасіння є (ratio < 0.5) у частинах (2,1,1) і (2,2) між three2 і зрізами, косинус
стабільної пари < -0.7 по шарах; у (4,) гасіння нема. Шанс, що є стабільна пара з cos < -0.7: 30%."""
import sys, os, math, time
import numpy as np, torch
torch.set_grad_enabled(False)
sys.path.insert(0, "/home/trim/ext/fc/repo/lab")
os.environ.setdefault("AW_SPEC", "path,star,three2,slice4,slice31,slice22,slice211")
import k4full as K4
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import linear_kprop, coerce_input, Kind
from mlp_kprop.wick import WICK_COEF_D
from mlp_kprop.diagslice import zero_repeated

n, L, seed, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4]))
OUT = sys.argv[5]
dev, dt = "cuda", torch.float32
rng = np.random.default_rng(seed)
W = [(rng.standard_normal((n, n)) * math.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
z = np.load(os.environ.get("AW_DATA", "data") + f"/kscan_{n}_{L}_{seed}_{N}.npz")
gt = z["gt"]
TYPES = ("path", "star", "three2", "slice4", "slice31", "slice22", "slice211")
KEYS = [(t, a) for t in TYPES for a in ("y", "o")]
birth, CUR, REC = {}, [0], []


def _dslices(self, groups):
    full = groups is self.groups
    kw = dict(device=self.device, dtype=self.dtype)
    tot = {p: torch.zeros((n,) * len(p), **kw) for p in K4.PARTS}
    if not full:
        for g in groups:
            K4.group_dslices(g, tot)
        return {p: zero_repeated(t) for p, t in tot.items()}
    per = {}
    for g in groups:
        b = birth.get(id(g.Bb), CUR[0])
        key = (g.types[0], "y" if CUR[0] - b <= 1 else "o")
        if key not in per:
            per[key] = {p: torch.zeros((n,) * len(p), **kw) for p in K4.PARTS}
        K4.group_dslices(g, per[key])
    ks = [k for k in KEYS if k in per]
    for p in K4.PARTS:
        V = [zero_repeated(per[k][p]).reshape(-1) for k in ks]   # f32 views, Gram accumulated in f64
        G = torch.zeros(len(ks), len(ks), dtype=torch.float64, device=dev)
        for i in range(len(ks)):
            for j in range(i, len(ks)):
                G[i, j] = G[j, i] = torch.dot(V[i].double(), V[j].double()) if V[i].numel() < 2**22 else \
                    sum(torch.dot(V[i][c:c + 2**22].double(), V[j][c:c + 2**22].double()) for c in range(0, V[i].numel(), 2**22))
        nrm = torch.sqrt(torch.diagonal(G))
        ratio = float(torch.sqrt(G.sum().clamp_min(0)) / nrm.sum().clamp_min(1e-300))
        s = torch.sqrt(G.sum().clamp_min(0))
        Gm = np.full((len(KEYS), len(KEYS)), np.nan)
        idx = [KEYS.index(k) for k in ks]
        Gm[np.ix_(idx, idx)] = G.cpu().numpy()
        REC.append((CUR[0], len(p), p, ratio, float(s), float(nrm.sum()), Gm))
        cs = (G / torch.outer(nrm, nrm).clamp_min(1e-300)).cpu().numpy()
        w = nrm.cpu().numpy() / nrm.sum().item()
        iu = [(i, j) for i in range(len(ks)) for j in range(i + 1, len(ks))]
        worst = sorted(iu, key=lambda ij: cs[ij] * min(w[ij[0]], w[ij[1]]))[:3]
        en = float(G.sum() / torch.diagonal(G).sum().clamp_min(1e-300))
        print(f"l={CUR[0]:2d} part={p} ratio |sum|/sum|v| {ratio:.3f} energy |sum|^2/sum|v|^2 {en:.3f} (1 = no cancel)  top weights " +
              " ".join(f"{ks[i][0]}:{ks[i][1]}={w[i]:.2f}" for i in np.argsort(-w)[:4]) + "  most-cancelling " +
              " ".join(f"{ks[i][0]}:{ks[i][1]}~{ks[j][0]}:{ks[j][1]} cos {cs[i, j]:+.2f}" for i, j in worst), flush=True)
        del V, G
    for p in K4.PARTS:
        for k in ks:
            tot[p] += per[k][p]
    del per
    return {p: zero_repeated(t) for p, t in tot.items()}


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
    print(f"layer {l} done t={time.time() - t0:.0f}s mem {torch.cuda.memory_allocated() / 2**30:.2f}G", flush=True)
per = np.mean((np.stack(means) - gt) ** 2, axis=1)
np.savez(OUT, layer=np.array([r[0] for r in REC]), part=np.array([str(r[2]) for r in REC]),
         ratio=np.array([r[3] for r in REC]), snorm=np.array([r[4] for r in REC]), sumnorm=np.array([r[5] for r in REC]),
         gram=np.stack([r[6] for r in REC]), keys=np.array([f"{t}:{a}" for t, a in KEYS]), per=per)
print(f"k4pair n={n} L={L} seed={seed}: final {per[-1]:.4e} t={time.time() - t0:.0f}s", flush=True)
