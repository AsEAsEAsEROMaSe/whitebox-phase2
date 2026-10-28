"""30.09: частка НЕ-хабових діаграм (трикутник C·C·C, κ₃-гіперребро + ребро) у народженні моря κ₃(h) (1,1,1) при k_max=3
на референсі з істинними входами (brc, 32 нейрони). Хабова граматика K=3 несе лише діаграми з вершиною, що торкається всіх ребер."""
import sys, os, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import mlp_kprop.kprop_ds as kds
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
torch.set_default_dtype(torch.float64)
T3 = GATI(3)
def hub_ok(vp):
    """чи є вершина, що торкається всіх блоків (тоді Σ_j δ_bj Π блоків = хаб); гіперребро (1,1,1) торкається всіх вершин,
    але разом із парним ребром дає два хаби ⟹ не-хаб."""
    vs = [tuple(v) for v in vp]
    if any(sum(1 for x in v if x > 0) == 3 for v in vs):
        return len(vs) == 1
    for u in range(3):
        if all(v[u] > 0 for v in vs):
            return True
    return False
for fn in sys.argv[1:]:
    D = np.load(fn); Nh = float(D["Nh"]); a = torch.tensor(D["a"]); m = a.shape[0]
    g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
    e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m)
    C2 = R2 - torch.outer(e, e)
    C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
    mean = a + e
    hh = {h: None for h in (0, 1)}
    for h in (0, 1):
        gh = lambda k: torch.tensor(D["%d_%s" % (h, k)]) / Nh
        eh = gh("q1"); R2h = gh("q2"); R3h = gh("q3").reshape(m, m, m); C2h = R2h - torch.outer(eh, eh)
        hh[h] = (a + eh, C2h, R3h - (R2h[:, :, None]*eh[None, None, :] + R2h[:, None, :]*eh[None, :, None] + R2h[None, :, :]*eh[:, None, None]) + 2*eh[:, None, None]*eh[None, :, None]*eh[None, None, :], gh("h3").reshape(m, m, m), gh("h1"), gh("h2"))
    def run(keep, mean_=mean, C2_=C2, C3_=C3):
        def patched(k_max, d_max=None, use_mean_var=False):
            return {ip: ({vp: (c if keep(vp) else 0.0) for vp, c in vd.items()} if ip == (1, 1, 1) else dict(vd)) for ip, vd in T3.items()}
        kds.get_all_terms_iso = patched
        return relu_kprop(DSTower.from_tower({1: mean_, 2: C2_, 3: C3_}), k_max=3)[3].to_tensor()
    ar = torch.arange(m); A_, B_, C_ = torch.meshgrid(ar, ar, ar, indexing="ij")
    sea = (A_ != B_) & (A_ != C_) & (B_ != C_)
    full = run(lambda vp: True)[sea]; zero = run(lambda vp: False)[sea]
    # істинна κ₃(h) з моментів h (h1,h2,h3): пломба референсу і похибка обрізання
    def k3h(h1, h2, h3):
        return h3 - (h2[:, :, None]*h1[None, None, :] + h2[:, None, :]*h1[None, :, None] + h2[None, :, :]*h1[:, None, None]) + 2*h1[:, None, None]*h1[None, :, None]*h1[None, None, :]
    h1 = (hh[0][4] + hh[1][4]) / 2; h2 = (hh[0][5] + hh[1][5]) / 2; h3 = (hh[0][3] + hh[1][3]) / 2
    tru = k3h(h1, h2, h3)[sea]; noise = ((k3h(*[hh[0][k] for k in (4, 5, 3)]) - k3h(*[hh[1][k] for k in (4, 5, 3)]))[sea] / 2).norm() / tru.norm()
    print("%s: ‖істина моря κ₃(h)‖ %.3e шум %.3f; референс k_max=3 повний: ‖ref−tru‖/‖tru‖ %.4f" % (os.path.basename(fn), tru.norm(), noise, (full - tru).norm() / tru.norm()), flush=True)
    hubs = run(hub_ok)[sea]; nonhub = full - hubs
    print("   хабові діаграми: ‖ref_hub−tru‖/‖tru‖ %.4f; не-хабові: частка норми %.4f, cos з (tru−hub) %+.3f, ‖tru−hub−nonhub‖/‖tru‖ %.4f" % (
        (hubs - tru).norm() / tru.norm(), nonhub.norm() / full.norm(), ((tru - hubs) * nonhub).sum() / (tru - hubs).norm() / nonhub.norm(), (tru - hubs - nonhub).norm() / tru.norm()), flush=True)
    for vp, c in T3[(1, 1, 1)].items():
        part = run(lambda x, vp=vp: x == vp)[sea] - zero
        if part.norm() / full.norm() > 0.003:
            print("     %-40s hub=%-5s частка норми %.4f  cos з повним %+.3f" % (vp, hub_ok(vp), part.norm() / full.norm(), (part * full).sum() / part.norm() / full.norm()), flush=True)
print("DONE_TRI3", flush=True)
