"""Референс: k_max=3 + ((1,1,2),) з κ₄ = носій (sea3) / носій + n²-море (carrier2) / + n²-зрізи; похибка народження моря κ₃."""
import sys, os, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import mlp_kprop.kprop_ds as kds
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
T3 = GATI(3); T4 = GATI(4)
i, ls = int(sys.argv[1]), int(sys.argv[2])
D = np.load("/home/trim/wb2data/brc_%d_%d.npz" % (i, ls)); Nh = float(D["Nh"]); a = torch.tensor(D["a"]); m = a.shape[0]
g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m)
C2 = R2 - torch.outer(e, e)
C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
mean = a + e
car = torch.tensor(np.load("/home/trim/wb2data/seacarry_%d_%d.npz" % (i, ls))["carrier"])
c2 = torch.load("/home/trim/wb2data/carrier2_%d_%d.pt" % (i, ls)); c2 = {k: v.double() for k, v in c2.items()}
h1 = (torch.tensor(D["0_h1"]) + torch.tensor(D["1_h1"])) / (2 * Nh); h2 = (torch.tensor(D["0_h2"]) + torch.tensor(D["1_h2"])) / (2 * Nh); h3 = (torch.tensor(D["0_h3"]) + torch.tensor(D["1_h3"])).reshape(m, m, m) / (2 * Nh)
tru3 = h3 - (h2[:, :, None]*h1[None, None, :] + h2[:, None, :]*h1[None, :, None] + h2[None, :, :]*h1[:, None, None]) + 2*h1[:, None, None]*h1[None, :, None]*h1[None, None, :]
ar = torch.arange(m); A_, B_, C_ = torch.meshgrid(ar, ar, ar, indexing="ij"); sea = (A_ != B_) & (A_ != C_) & (B_ != C_); tru = tru3[sea]
def run(K4):
    keep = lambda vp: vp in T3[(1, 1, 1)] or vp in {((1, 1, 2),), ((1, 2, 1),), ((2, 1, 1),)}
    def patched(k_max, d_max=None, use_mean_var=False):
        return {ip: ({vp: (c if keep(vp) else 0.0) for vp, c in vd.items()} if ip == (1, 1, 1) else dict(vd)) for ip, vd in T4.items()}
    kds.get_all_terms_iso = patched
    return relu_kprop(DSTower.from_tower({1: mean, 2: C2, 3: C3, 4: K4}), k_max=4)[3].to_tensor()[sea]
for nm, K in (("κ₄=0", torch.zeros_like(k4)), ("істинна", k4), ("sea3", car), ("sea3 + n²-море", car + c2["sea"]), ("sea3 + n²-море + n²-зрізи", car + c2["sea"] + c2["n2"]), ("sea3 + n²-море, ×1.1", 1.1 * (car + c2["sea"]))):
    x = run(K); print("  %-30s ‖ref−tru‖/‖tru‖ %.4f" % (nm, (x - tru).norm() / tru.norm()), flush=True)
print("DONE_TRI4B", flush=True)
