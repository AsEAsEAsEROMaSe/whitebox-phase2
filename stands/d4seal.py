"""Пломба d4terms проти референсної kprop (32 нейрони, істинні кумулянти з brc_*.npz).
base = kprop з термами k_max=3 (у машинерії k_max=4); test = base + мої (1,1,1)-діаграми. Δ_ref = test − base на (1,1,1).
Очікування: Δ_mine/Δ_ref = 1 до 1e-5 (тоді конвертація pK→K на (1,1,1) тотожна для доданого)."""
import sys, os, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import mlp_kprop.kprop_ds as kds
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
from mlp_kprop.wick import relu_wick_coef
import d4terms
torch.set_default_dtype(torch.float64)
fn = sys.argv[1] if len(sys.argv) > 1 else "/home/trim/wb2data/brc_0_12.npz"
D = np.load(fn); Nh = float(D["Nh"]); a = torch.tensor(D["a"]); m = a.shape[0]
g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m)
C2 = R2 - torch.outer(e, e)
C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
mean = a + e
T3 = GATI(3); T4 = GATI(4)
mine = d4terms.select_diagrams(T3, T4, with_k4=os.environ.get("K4", "0") == "1")
print("діаграм:", len(mine), [vp for vp, c in mine][:6], "...")
def run(extra):
    def patched(k_max, d_max=None, use_mean_var=False):
        out = {}
        for ip, vd in T4.items():
            out[ip] = {vp: c for vp, c in vd.items() if vp in T3.get(ip, {}) or (ip == (1, 1, 1) and vp in extra)}
        return out
    kds.get_all_terms_iso = patched
    return relu_kprop(DSTower.from_tower({1: mean, 2: C2, 3: C3, 4: k4}), k_max=4)[3].to_tensor()
P_base = run(set()); P_test = run({vp for vp, c in mine})
iu = torch.tensor([[x, y, z] for x in range(m) for y in range(m) for z in range(m) if x < y < z])
sel = lambda T: T[iu[:, 0], iu[:, 1], iu[:, 2]]
dref = sel(P_test - P_base)
# моє
var = torch.diagonal(C2).clone(); Coff = C2.clone(); Coff.fill_diagonal_(0.0)
D3 = torch.einsum("aaa->a", C3).clone(); D21 = torch.einsum("aac->ac", C3).clone(); D21.fill_diagonal_(0.0)
blocks = dict(var=var, C=Coff, D3=D3, D21=D21, K4=torch.einsum("aaaa->a", k4).clone(),
              K22=torch.einsum("aabb->ab", k4).clone(), K31=torch.einsum("aaab->ab", k4).clone())
wick = lambda k, p=1: relu_wick_coef(mean, var, k, p)
dm = d4terms.delta_slab(mine, blocks, wick, C3, slice(0, m), m, "cpu", torch.float64)
dmine = sel(dm)
ratio = float((dmine * dref).sum() / (dref * dref).sum()); rel = float(((dmine - dref) ** 2).sum().sqrt() / (dref ** 2).sum().sqrt())
print("Δ_ref норма %.4e  Δ_mine норма %.4e  нахил %.6f  віднос. похибка %.2e" % (float(dref.norm()), float(dmine.norm()), ratio, rel))
# по класах: кожна діаграма окремо
for vp, c in mine:
    P1 = run({vp}); d1 = sel(P1 - P_base)
    m1 = sel(d4terms.delta_slab([(vp, c)], blocks, wick, C3, slice(0, m), m, "cpu", torch.float64))
    print("  %-40s count %d |Δ_ref| %.3e нахил %.5f" % (vp, c, float(d1.norm()), float((m1 * d1).sum() / (d1 * d1).sum()) if float(d1.norm()) > 0 else 0))
comp = d4terms.compile_diagrams(mine, blocks, wick, m)
df = sel(d4terms.delta_slab_fast(comp, blocks, C3, slice(0, m), m, "cpu"))
print("швидка форма: класів %d, T разом %d, віднос. похибка проти повільної %.2e" % (len(comp), sum(len(v) for v in comp.values()), float((df - dmine).norm() / dmine.norm())))
for k, v in comp.items(): print("   ", k, len(v))
