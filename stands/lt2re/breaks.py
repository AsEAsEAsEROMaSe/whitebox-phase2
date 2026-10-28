"""breaks (20.10.2026 white-box-89, ідеї юзера про переломи ReLU): три виміри на мережі корпусу 1024×16, шари LS.
(1) Колективне перемикання: на семпл рахуємо N_near = #{j: |p_j| < EPS·σ_j}; надрозкид D = Var/E реальний проти гаусового
    контролю p ~ N(μ, C_pre) з ТИМИ САМИМИ моментами. D_real >> D_gauss ⟹ існують «критичні точки», де перемикаються групи.
(2) Умовні моменти на переломі: для 32 найкритичніших нейронів j (мінімальний |μ/σ|) E[p_k | |p_j|<EPS σ_j] і Var проти
    гаусової формули μ_k + C_kj/C_jj(0−μ_j), C_kk − C_kj²/C_jj. Відносна похибка = похибка гаусового замикання на переломі.
(3) Кластери: ефективні ваги E[∇_z p_l] = E[p_l z] (Стайн) — косинуси між нейронами; і кореляції індикаторів перелому
    64 критичних нейронів (реальні проти гаусових).
ПЕРЕДБАЧЕННЯ: D_real/D_gauss ∈ [1.0, 1.5]; умовне середнє на переломі відхиляється від гауса на 5–20%; ефективні ваги на
шарі 14 злипаються (медіана |cos| > 0.5). Шанс, що (1) дасть ×2+: 15%.
usage: breaks.py <k> <out.json>"""
import json, os, sys, time, math
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]
LS = [int(x) for x in os.environ.get("LS", "10,12,14").split(",")]
N1 = N2 = int(float(os.environ.get("NS", str(2 ** 21)))); B = 32768; EPS = float(os.environ.get("EPS", "0.1"))
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(770000 + k)
W = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)).to(dev)
gg = torch.Generator(device=dev).manual_seed(55000 + k)


def forward(z):
    out, h = {}, z
    for l in range(16):
        p = h @ W[l]
        if l in LS:
            out[l] = p
        h = torch.relu(p)
    return out


t0 = time.time()
# pass 1: pre-activation moments and Stein effective weights E[p z]
S1 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in LS}
S2 = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in LS}
SZ = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in LS}
sh = {}
done = 0
while done < N1:
    z = torch.randn(B, 1024, generator=gg, device=dev)
    P = forward(z)
    for l in LS:
        if l not in sh:
            sh[l] = P[l].mean(0)
        pc = P[l] - sh[l]
        S1[l] += pc.sum(0, dtype=torch.float64)
        S2[l] += (pc.T @ pc).double()
        SZ[l] += (z.T @ P[l]).double()
    done += B
mu, C, J = {}, {}, {}
for l in LS:
    m = S1[l] / done
    C[l] = S2[l] / done - torch.outer(m, m)
    mu[l] = m + sh[l].double()
    J[l] = SZ[l] / done          # (in 1024, neuron 1024): column j = E[grad_z p_j]
res = {"k": k, "eps": EPS, "layers": {}}
for l in LS:
    sd = torch.sqrt(torch.diagonal(C[l]))
    t = mu[l] / sd
    crit = torch.argsort(t.abs())[:64]
    ev, V = torch.linalg.eigh(C[l])
    R = (V * torch.sqrt(ev.clamp_min(0))).float()
    th = (EPS * sd).float()
    # (1) counts and (2) conditionals and (3b) indicator correlations, real and Gaussian
    stats = {}
    for kind in ("real", "gauss"):
        cnt1 = cnt2 = 0.0
        nsum = torch.zeros(32, dtype=torch.float64, device=dev)
        msum = torch.zeros(32, 1024, dtype=torch.float64, device=dev)
        qsum = torch.zeros(32, 1024, dtype=torch.float64, device=dev)
        bsum = torch.zeros(64, dtype=torch.float64, device=dev)
        bbsum = torch.zeros(64, 64, dtype=torch.float64, device=dev)
        n = 0
        gg2 = torch.Generator(device=dev).manual_seed(77000 + 10 * k + l)
        while n < N2:
            if kind == "real":
                p = forward(torch.randn(B, 1024, generator=gg2, device=dev))[l]
            else:
                p = mu[l].float() + torch.randn(B, 1024, generator=gg2, device=dev) @ R.T
            near = (p.abs() < th)
            c = near.sum(1).double()
            cnt1 += float(c.sum()); cnt2 += float((c * c).sum())
            sel = near[:, crit[:32]].double()                  # (B, 32)
            nsum += sel.sum(0)
            pd = (p - mu[l].float()).double()
            msum += sel.T @ pd
            qsum += sel.T @ (pd * pd)
            b = (p[:, crit].abs() < 2.5 * th[crit]).double()
            bsum += b.sum(0); bbsum += b.T @ b
            n += B
        Em = cnt1 / n
        stats[kind] = {"E_near": Em, "D": (cnt2 / n - Em ** 2) / max(Em, 1e-12)}
        cm = msum / nsum[:, None].clamp_min(1)                 # E[p - mu | p_j ~ 0]
        cq = qsum / nsum[:, None].clamp_min(1) - cm ** 2       # Var[p | p_j ~ 0]
        stats[kind]["cm"] = cm; stats[kind]["cq"] = cq; stats[kind]["nsel"] = nsum
        pb = bsum / n
        cov = bbsum / n - torch.outer(pb, pb)
        cor = cov / torch.sqrt(torch.outer(torch.diagonal(cov), torch.diagonal(cov))).clamp_min(1e-30)
        off = cor[~torch.eye(64, dtype=torch.bool, device=dev)]
        stats[kind]["ind_cor_meanabs"] = float(off.abs().mean())
        stats[kind]["ind_cor_top_eig"] = float(torch.linalg.eigvalsh(cor)[-1])
    # Gaussian formula for the conditionals at p_j = 0
    j32 = crit[:32]
    Cj = C[l][:, j32].T                                         # (32, 1024)
    gm = Cj / torch.diagonal(C[l])[j32][:, None] * (-mu[l][j32])[:, None]
    gq = torch.diagonal(C[l])[None, :] - Cj ** 2 / torch.diagonal(C[l])[j32][:, None]
    mask = torch.ones(32, 1024, dtype=torch.bool, device=dev)
    mask[torch.arange(32), j32] = False

    def relerr(a, b, ref):
        return float(((a - b)[mask].norm() / ref[mask].norm()).item())
    shift = gm
    lr = {"D_real": stats["real"]["D"], "D_gauss": stats["gauss"]["D"], "E_near_real": stats["real"]["E_near"],
          "E_near_gauss": stats["gauss"]["E_near"],
          "cond_mean_err_real_vs_formula": relerr(stats["real"]["cm"], gm, shift),
          "cond_mean_err_gaussMC_vs_formula": relerr(stats["gauss"]["cm"], gm, shift),
          "cond_var_err_real_vs_formula": relerr(stats["real"]["cq"], gq, gq),
          "cond_var_err_gaussMC_vs_formula": relerr(stats["gauss"]["cq"], gq, gq),
          "nsel_min_real": float(stats["real"]["nsel"].min()),
          "ind_cor_meanabs_real": stats["real"]["ind_cor_meanabs"], "ind_cor_meanabs_gauss": stats["gauss"]["ind_cor_meanabs"],
          "ind_cor_top_eig_real": stats["real"]["ind_cor_top_eig"], "ind_cor_top_eig_gauss": stats["gauss"]["ind_cor_top_eig"],
          "frac_abs_t_lt1": float((t.abs() < 1).float().mean()), "median_abs_t": float(t.abs().median())}
    U = J[l] / J[l].norm(dim=0, keepdim=True)
    Gc = (U.T @ U).abs(); Gc.fill_diagonal_(0)
    lr["effw_cos_median"] = float(Gc.flatten().median()); lr["effw_cos_max"] = float(Gc.max())
    Uc = U[:, crit]; Gk = (Uc.T @ Uc).abs(); Gk.fill_diagonal_(0)
    lr["effw_cos_median_crit64"] = float(Gk.flatten().median())
    sv = torch.linalg.svdvals(J[l]); e = sv ** 2
    lr["effw_effrank"] = float(e.sum() ** 2 / (e ** 2).sum())
    res["layers"][str(l)] = lr
    print(k, l, " ".join("%s=%.4g" % (a, b) for a, b in lr.items()), flush=True)
res["t"] = time.time() - t0
json.dump(res, open(OUT, "w"), indent=1)
