"""Пломба конвенцій 30.09: моя формула sea3-діаграм κ₄(h) (κ₃-гіперребро + одне ребро) проти референсної kprop k_max=4
на 32 нейронах з ІСТИННИМИ входами (brc_*.npz). Референс: клас лише з цих vp (patched keep). Очікування: відношення 1 до 1e-10."""
import sys, os, itertools, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import mlp_kprop.kprop_ds as kds
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
from mlp_kprop.partitions import vec_part_coef
sys.argv = sys.argv[:1] + []
_a = sys.argv; sys.argv = sys.argv[:1]; import lean_k3_aug_gpu as lk; sys.argv = _a
torch.set_default_dtype(torch.float64)
T4 = GATI(4)
fn = "/home/trim/wb2data/brc_0_12.npz"
D = np.load(fn); Nh = float(D["Nh"]); a = torch.tensor(D["a"]); m = a.shape[0]
g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m)
C2 = R2 - torch.outer(e, e)
C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
mean = a + e
SEA3 = [vp for vp in T4[(1, 1, 1, 1)] if any(sum(v) == 3 and sum(1 for x in v if x > 0) == 3 for v in vp)
        and not any(sum(v) >= 4 and sum(1 for x in v if x > 0) >= 3 for v in vp)]
print("sea3 vp:", [(vp, T4[(1,1,1,1)][vp], vec_part_coef(vp, divide_fac=True)) for vp in SEA3])
def run(keep):
    def patched(k_max, d_max=None, use_mean_var=False):
        return {ip: ({vp: (c if vp in keep else 0.0) for vp, c in vd.items()} if ip == (1, 1, 1, 1) else dict(vd)) for ip, vd in T4.items()}
    kds.get_all_terms_iso = patched
    return relu_kprop(DSTower.from_tower({1: mean, 2: C2, 3: C3, 4: k4}), k_max=4)[4].to_tensor()
zero = run(set())
ar = torch.arange(m); A_, B_, C_, D_ = torch.meshgrid(ar, ar, ar, ar, indexing="ij")
sea = (A_ != B_) & (A_ != C_) & (A_ != D_) & (B_ != C_) & (B_ != D_) & (C_ != D_)
wk = lk.WickCache(mean, torch.diagonal(C2))
w = {k: wk(k, 1) for k in range(1, 5)}
Coff = C2.clone(); Coff.fill_diagonal_(0.0)
D21 = torch.einsum("aac->ac", C3).clone(); D21.fill_diagonal_(0.0)      # D21[a,c] = κ₃_{aac}
K22 = torch.einsum("aabb->ab", k4).clone(); K22.fill_diagonal_(0.0)
def edge(kc, kd):
    """блок ребра між c (порядок kc) і d (порядок kd): M[c,d]"""
    if (kc, kd) == (1, 1): return Coff
    if (kc, kd) == (2, 1): return D21          # κ_{ccd}
    if (kc, kd) == (1, 2): return D21.T        # κ_{cdd}
    if (kc, kd) == (2, 2): return K22
    raise ValueError((kc, kd))
def mine(vp):
    """T[a,b,c,d] = vpc · (cnt/24) · Σ_{24 перм} Π w(deg) · κ₃_{hyper} · edge; hyperedge = вектор із трьома ненулями."""
    vpc = vec_part_coef(vp, divide_fac=True); cnt = T4[(1, 1, 1, 1)][vp]
    h = [v for v in vp if sum(1 for x in v if x > 0) == 3][0]; ed = [v for v in vp if v != h][0]
    hv = [i for i in range(4) if h[i] > 0]; ev = [i for i in range(4) if ed[i] > 0]
    deg = [h[i] + ed[i] for i in range(4)]
    base = torch.ones(m, m, m, m)
    for i in range(4):
        shp = [1, 1, 1, 1]; shp[i] = m; base = base * w[deg[i]].reshape(shp)
    perm = hv + [i for i in range(4) if i not in hv]
    t = C3.reshape(m, m, m, 1).expand(m, m, m, m)
    inv = [perm.index(i) for i in range(4)]
    base = base * t.permute(*inv)
    E = edge(ed[ev[0]], ed[ev[1]])
    perm = ev + [i for i in range(4) if i not in ev]
    t = E.reshape(m, m, 1, 1).expand(m, m, m, m)
    inv = [perm.index(i) for i in range(4)]
    base = base * t.permute(*inv)
    tot = torch.zeros(m, m, m, m)
    for p in itertools.permutations(range(4)):
        tot = tot + base.permute(*p)
    return vpc * (cnt / 24.0) * tot
for vp in SEA3:
    ref = (run({vp}) - zero)
    my = mine(vp)
    r, y = ref[sea], my[sea]
    print("%-32s ‖ref‖ %.3e  ‖my‖ %.3e  cos %+.6f  ‖my−ref‖/‖ref‖ %.2e" % (vp, r.norm(), y.norm(), (r*y).sum()/r.norm()/y.norm(), (y-r).norm()/r.norm()), flush=True)
print("DONE_SEA3SEAL")
