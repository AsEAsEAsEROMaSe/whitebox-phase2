"""30.09: похибка народження моря κ₃(h) на референсі (32 нейрони, істинні входи) для: k_max=3; k_max=4 з істинною κ₄;
k_max=4 з κ₄=0 (лише n²+κ₃ діаграми = d4); k_max=4 з κ₄ = наш носій (seacarry npz); лише 3-осьові κ₄-діаграми ((1,1,2)-тип) з істинною/носієм."""
import sys, os, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import mlp_kprop.kprop_ds as kds
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
torch.set_default_dtype(torch.float64)
T3 = GATI(3); T4 = GATI(4)
def has3ax(vp): return any(sum(v) >= 4 and sum(1 for x in v if x > 0) == 3 for v in vp)
def has_k4(vp): return any(sum(v) >= 4 for v in vp)
for fn, cf in [(sys.argv[1], sys.argv[2])]:
    D = np.load(fn); Nh = float(D["Nh"]); a = torch.tensor(D["a"]); m = a.shape[0]
    g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
    e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m)
    C2 = R2 - torch.outer(e, e)
    C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
    k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
    mean = a + e
    Z = np.load(cf); carrier = torch.tensor(Z["carrier"]); assert carrier.shape == k4.shape
    h1 = (torch.tensor(D["0_h1"]) + torch.tensor(D["1_h1"])) / (2 * Nh); h2 = (torch.tensor(D["0_h2"]) + torch.tensor(D["1_h2"])) / (2 * Nh); h3 = (torch.tensor(D["0_h3"]) + torch.tensor(D["1_h3"])).reshape(m, m, m) / (2 * Nh)
    tru3 = h3 - (h2[:, :, None]*h1[None, None, :] + h2[:, None, :]*h1[None, :, None] + h2[None, :, :]*h1[:, None, None]) + 2*h1[:, None, None]*h1[None, :, None]*h1[None, None, :]
    ar = torch.arange(m); A_, B_, C_ = torch.meshgrid(ar, ar, ar, indexing="ij"); sea = (A_ != B_) & (A_ != C_) & (B_ != C_)
    tru = tru3[sea]
    def run(kmax, keep, K4):
        def patched(k_max, d_max=None, use_mean_var=False):
            T = T4 if kmax == 4 else T3
            return {ip: ({vp: (c if keep(vp) else 0.0) for vp, c in vd.items()} if ip == (1, 1, 1) else dict(vd)) for ip, vd in T.items()}
        kds.get_all_terms_iso = patched
        tower = {1: mean, 2: C2, 3: C3}
        if kmax == 4: tower[4] = K4
        return relu_kprop(DSTower.from_tower(tower), k_max=kmax)[3].to_tensor()[sea]
    def rep(name, x): print("  %-46s ‖ref−tru‖/‖tru‖ %.4f" % (name, (x - tru).norm() / tru.norm()), flush=True)
    print(os.path.basename(fn), "‖tru‖ %.3e" % tru.norm(), flush=True)
    rep("k_max=3 (хаби)", run(3, lambda vp: True, None))
    rep("k_max=4, κ₄=0 (n²+κ₃ діаграми, d4)", run(4, lambda vp: True, torch.zeros_like(k4)))
    rep("k_max=4, істинна κ₄", run(4, lambda vp: True, k4))
    rep("k_max=4, κ₄ = носій", run(4, lambda vp: True, carrier))
    rep("k_max=3 + лише 3-осьові κ₄-діаграми, істинна κ₄", run(4, lambda vp: (not has_k4(vp)) and vp in T3[(1, 1, 1)] or has3ax(vp), k4))
    rep("k_max=3 + лише ((1,1,2),), істинна κ₄", run(4, lambda vp: vp in T3[(1, 1, 1)] or vp in {((1, 1, 2),), ((1, 2, 1),), ((2, 1, 1),)}, k4))
    rep("k_max=3 + лише ((1,1,2),), носій", run(4, lambda vp: vp in T3[(1, 1, 1)] or vp in {((1, 1, 2),), ((1, 2, 1),), ((2, 1, 1),)}, carrier))
print("DONE_TRI4", flush=True)
