"""k4proj (20.10.2026, white-box-89): чи можна нести стан K=4 лише в r напрямках?
Повний factored K=4 Susupro (lab/k4full.py, MIT, поверх ARC mlp_kprop) з гаком: після нелінійності шару l кожна
група κ4 віком >= A0 проєктується по рядках (усі чотири ноги в одному просторі нейронів): Q <- U U^T Q.
Напрямки U (n x r):
  adj  — праві сингулярні вектори лінеаризованого відображення post-act шару l -> post-act виходу
         J_l = D_{L-1} W_{L-1} ... D_{l+1} W_{l+1}, D = P(p > 0) з МК (65536 семплів);
  cov  — провідні власні вектори коваріації K[2] шару l (стан самого ланцюга);
  rand — випадковий r-вимірний підпростір (контроль).
usage: k4proj.py <n> <L> <seed> <N_truth> <mode> <r> <A0>    (mode: none|adj|cov|rand|hosvd; AW_SPEC як у k4full)
Пломби: mode=none = k4full kscan; r=n дає те саме, що none (проєктор = I).
Передбачення (до прогону, n=256): adj r=32 тримає >= 60% log-виграшу K=4 над k3a, rand r=32 < 30%; шанс 25%."""
import sys, os, math, time
import numpy as np, torch
torch.set_grad_enabled(False)
sys.path.insert(0, "/home/trim/ext/fc/repo/lab")
import k4full as K4
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import linear_kprop, coerce_input, Kind
from mlp_kprop.wick import WICK_COEF_D

n, L, seed, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4]))
mode, r, A0 = sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
dev = "cuda"
dt = torch.float32 if os.environ.get("AW_F32", "1") == "1" else torch.float64
rng = np.random.default_rng(seed)
W = [(rng.standard_normal((n, n)) * math.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
z = np.load(os.environ.get("AW_DATA", "data") + f"/kscan_{n}_{L}_{seed}_{N}.npz")
gt, var_final = z["gt"], z["var_final"]

# slopes D_l = P(p_l > 0) by MC (x @ W convention, as kscan)
Wt = [torch.from_numpy(w).to(dev) for w in W]
g = torch.Generator(device=dev).manual_seed(seed + 777)
cnt = [torch.zeros(n, device=dev) for _ in range(L)]
for _ in range(4):
    h = torch.randn(16384, n, generator=g, device=dev)
    for l in range(L):
        p = h @ Wt[l]
        cnt[l] += (p > 0).float().sum(0)
        h = torch.relu(p)
D = [c / 65536.0 for c in cnt]


def adj_dirs(l):
    """right singular vectors of J_l = d h_{L-1} / d h_l (h row vectors: h_{k+1} = relu(h_k @ W_{k+1}))."""
    J = torch.eye(n, device=dev, dtype=torch.float64)
    for k in range(L - 1, l, -1):   # J <- J @ (d h_k / d h_{k-1})^T in column form: dh_k = D_k * (dh_{k-1} @ W_k)
        J = (J * D[k].double()[None, :]) @ Wt[k].double().T
    _, _, vh = torch.linalg.svd(J, full_matrices=False)
    return vh[:r].T.to(dt)        # (n, r) directions in the post-act space of layer l


def cov_dirs(C):
    w, V = torch.linalg.eigh(C.double())
    return V[:, -r:].to(dt)


grng = torch.Generator(device=dev).manual_seed(seed + 999)


def rand_dirs():
    Q, _ = torch.linalg.qr(torch.randn(n, r, generator=grng, device=dev, dtype=torch.float64))
    return Q.to(dt)


mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).to(device=dev, dtype=dt)
for i in range(L):
    mlp.Ws[i].weight.data = torch.from_numpy(W[i].T.copy()).to(device=dev, dtype=dt).contiguous()
Kc = coerce_input({1: torch.zeros(n, device=dev, dtype=dt), 2: torch.eye(n, device=dev, dtype=dt)}, k_max=4, kind=Kind.SIMPLE)
wick = WICK_COEF_D["relu"]
birth = {}   # id(group.Bb) -> birth layer
ATOM_SPEC = os.environ.get("K4_ATOM_SPEC", "0") == "1"   # print per-atom kernel spectra of newborn groups
ATOM_K = int(os.environ.get("K4_ATOM_RANK", "0"))         # truncate each atom kernel at birth to rank ATOM_K (0: off)
ATOM_TYPES = os.environ.get("K4_ATOM_TYPES", "all")       # which birth types to truncate (path,star,three2,slice4,...)
SKIP = int(os.environ.get("K4_SKIP_YOUNG_READS", "0"))   # mute reads of groups with age <= SKIP (state still transported)
CUR = [0]
_orig_gd = K4.group_dslices


def _gd(g, out):
    b = birth.get(id(g.Bb))
    if SKIP > 0 and b is not None and CUR[0] - b <= SKIP:
        return
    return _orig_gd(g, out)


K4.group_dslices = _gd
means = []
t0 = time.time()
for l in range(L):
    K4.LAST_LAYER = l == L - 1
    CUR[0] = l
    Kc = linear_kprop(Kc, mlp.Ws[l].weight, k_max=4, set_metric=mlp.init_scale[l])
    Kc = K4.nonlin(Kc, nonlin_wick_coef=wick)
    k4 = Kc[4]
    for gr in getattr(k4, "groups", []):
        if id(gr.Bb) in birth:
            continue
        birth[id(gr.Bb)] = l
        if ATOM_SPEC or ATOM_K > 0:
            # «розширення»: окремий простір кожного атома r — ядро Bb[:, :, r] (n x n); спектр і обрізання до рангу ATOM_K
            Bd = gr.Bb.to(dev, dtype=torch.float64)
            ker = Bd.permute(2, 0, 1)                       # (R, n, n), atom-major
            U_, S_, Vh_ = torch.linalg.svd(ker, full_matrices=False)
            e = (S_ ** 2)
            tot = e.sum(1, keepdim=True).clamp_min(1e-300)
            cum = torch.cumsum(e, 1) / tot
            w = tot[:, 0] / tot.sum()                        # atom weight in the group's energy
            fr = [float((cum[:, k - 1] * w).sum()) for k in (1, 2, 4, 8, 16, 32)]
            print(f"atoms l={l} types={'+'.join(gr.types)} R={ker.shape[0]} energy@k(1,2,4,8,16,32) " +
                  " ".join(f"{x:.4f}" for x in fr), flush=True)
            if ATOM_K > 0 and (ATOM_TYPES == "all" or any(t in ATOM_TYPES.split(",") for t in gr.types)):
                kk = ATOM_K
                trunc = (U_[:, :, :kk] * S_[:, None, :kk]) @ Vh_[:, :kk, :]
                gr.Bb.copy_(trunc.permute(1, 2, 0).to(gr.Bb.dtype).to(gr.Bb.device))
                k4.clear_repeated()
    if mode == "hosvd" and r < n and l < L - 1:
        # "кубик Рубика": власний базис кожної групи — провідні напрямки першої розгортки її тензора,
        # G = Q (Bb_flat Bb_flat^T + X X^T) Q^T, де Bb_flat = Bb.reshape(n, n R), X = xb (двоногі групи)
        hit = False
        for gr in k4.groups:
            if l - birth[id(gr.Bb)] >= A0:
                Bb = gr.Bb.to(dev, dtype=torch.float64)
                Bf = Bb.reshape(n, -1)
                M = Bf @ Bf.T
                if gr.xb is not None:
                    xb = gr.xb.to(torch.float64)
                    M = M + (xb @ xb.T) * (torch.linalg.matrix_norm(M) / torch.linalg.matrix_norm(xb @ xb.T))
                Qd = gr.Q.to(torch.float64)
                G = Qd @ M @ Qd.T
                w_, V_ = torch.linalg.eigh(G)
                U = V_[:, -r:]
                gr.Q = ((U @ U.T) @ Qd).to(dt)
                hit = True
        if hit:
            k4.clear_repeated()
    elif mode != "none" and r < n and l < L - 1:
        U = None
        for gr in k4.groups:
            if l - birth[id(gr.Bb)] >= A0:
                if U is None:
                    U = adj_dirs(l) if mode == "adj" else cov_dirs(Kc[2].to_tensor()) if mode == "cov" else rand_dirs()
                    P = U @ U.T
                gr.Q = P @ gr.Q
        if U is not None:
            k4.clear_repeated()
    means.append(Kc[1].to_tensor().double().cpu().numpy())
pred = np.stack(means)
per = np.mean((pred - gt) ** 2, axis=1)
print(f"k4proj atomk={ATOM_K}:{ATOM_TYPES} skip={SKIP} spec={os.environ.get('AW_SPEC', 'full')} n={n} L={L} seed={seed} mode={mode} r={r} A0={A0}: "
      f"final {per[-1]:.4e} t={time.time() - t0:.0f}s | " + " ".join(f"{x:.1e}" for x in per[::3]), flush=True)
