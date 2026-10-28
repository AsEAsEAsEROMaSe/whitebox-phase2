"""Ранг носія κ₄ за класами діаграм (стенд 26.09 пізно). Референс kprop k_max=4 на 32 нейронах з ІСТИННИМИ кумулянтами
p_l (brc_*.npz): народжена κ₄(h) розкладається за класами діаграм ключів (1,1,1,1) і (2,1,1):
  star = лише n²-блоки, зіркове дерево (CP-ранг n після переносу W) · n2 = n²-блоки з шляхом/циклом/трикутником (ранг n²)
  sea3 = має κ₃-гіперребро входу (ранг n·n) · mem = має κ₄-гіперребро входу (пам'ять, ранг носія).
Виходи лінійні за списком членів ⟹ сума класів = усе (пломба). Міряємо частку норми і cos кожного класу у морі (1,1,1,1)
і у тривісному зрізі (2,1,1) з масками різних індексів. Передбачення ДО прогону: частка n2+sea3 у народженні ≥30%
(шанс 60%) ⟹ зірковий носій рангу n не існує; mem ≥ 50% моря (пам'ять)."""
import sys, os, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import mlp_kprop.kprop_ds as kds
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
torch.set_default_dtype(torch.float64)
T4 = GATI(4)
def cls(vp):
    edges, h3, h4 = set(), 0, 0
    for v in vp:
        nz = tuple(i for i, x in enumerate(v) if x > 0)
        if len(nz) == 2: edges.add(nz)
        elif len(nz) >= 3 and sum(v) == 3: h3 += 1
        elif len(nz) >= 3: h4 += 1
    if h4: return "mem"
    if h3: return "sea3"
    nv = len(vp[0]); verts = set(i for e in edges for i in e)
    # зв'язність графа з ребер
    comp = {u: u for u in range(nv)}
    def f(x):
        while comp[x] != x: x = comp[x]
        return x
    for u, v in edges: comp[f(u)] = f(v)
    if len(set(f(u) for u in range(nv))) > 1: return "disc"
    if len(edges) == nv - 1 and any(all(u in e for e in edges) for u in verts): return "star"
    return "n2"
KEYS = [(1, 1, 1, 1), (2, 1, 1)]
CL = ["disc", "star", "n2", "sea3", "mem"]
for ip in KEYS:
    cnt = {c: 0 for c in CL}
    for vp in T4[ip]: cnt[cls(vp)] += 1
    print("ключ %s: діаграм за класами %s" % (ip, cnt), flush=True)
def run(keep):
    def patched(k_max, d_max=None, use_mean_var=False):
        return {ip: ({vp: (c if keep(cls(vp)) else 0.0) for vp, c in vd.items()} if ip in KEYS else dict(vd)) for ip, vd in T4.items()}
    kds.get_all_terms_iso = patched
    return relu_kprop(DSTower.from_tower({1: mean, 2: C2, 3: C3, 4: k4}), k_max=4)[4].to_tensor()
for fn in sys.argv[1:]:
    D = np.load(fn); Nh = float(D["Nh"]); a = torch.tensor(D["a"]); m = a.shape[0]
    g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
    e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m)
    C2 = R2 - torch.outer(e, e)
    C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
    k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
    mean = a + e
    ar = torch.arange(m); A_, B_, C_, D_ = torch.meshgrid(ar, ar, ar, ar, indexing="ij")
    sea = (A_ != B_) & (A_ != C_) & (A_ != D_) & (B_ != C_) & (B_ != D_) & (C_ != D_)
    tri3 = (A_ != B_) & (A_ != C_) & (B_ != C_)
    full = run(lambda c: True); zero = run(lambda c: False); parts = {c: run(lambda x, c=c: x == c) - zero for c in CL}
    tot = sum(parts.values()) + zero
    print("%s: пломба ‖zero + Σ(клас−zero) − усе‖/‖усе‖ = %.2e" % (os.path.basename(fn), float((tot - full).norm() / full.norm())), flush=True)
    parts["conv(zero)"] = zero
    for nm, msk, get in (("море (1,1,1,1)", sea, lambda T: T[msk]), ("тривісний [a,a,b,c]", tri3[0], lambda T: T[ar, ar][tri3[0]])):
        y = get(full); print("  %-22s ‖усе‖ %.3e" % (nm, float(y.norm())))
        for c in CL + ["conv(zero)"]:
            x = get(parts[c]); print("     %-10s частка норми %.3f  cos з усім %+.3f  ‖усе−клас‖²/‖усе‖² %.3f" % (
                c, float(x.norm() / y.norm()), float((x * y).sum() / x.norm() / y.norm()) if x.norm() > 0 else 0, float(((y - x) ** 2).sum() / (y ** 2).sum())))
print("DONE_K4CLASS")
