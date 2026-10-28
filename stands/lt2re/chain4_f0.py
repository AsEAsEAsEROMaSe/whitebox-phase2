"""chain4_f0 (21.10.2026 white-box-f0, ідея 5f): член c1-c2-c2-c1 зв'язного c4 передактивацій з Ерміт-розкладу ReLU,
зі СПРАВЖНІХ (m, C) передактивацій попереднього шару (МК), проти rest = c4 - diag - aabb - aaab (c4arr_n1_L12/14).
x_a = ReLU(g_a) = s_a (c1 He1(u_a) + c2 He2(u_a) + ...), c1 = Phi(t), c2 = phi(t)/2, t = m/s.
F_j = 48 * sum_{c!=d} B_c C_cd B_d,  B = W_j * c2 * (C (W_j*c1)) / s   (2 матмули n^3 на шар).
ПЕРЕДБАЧЕННЯ: corr(F, rest) 0.3…0.6, нахил ≠ 1 (вищі діаграми). Шанс corr ≥ 0.6: 30%.
usage: chain4_f0.py <k> <out.json>"""
import json, os, sys, math
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]; LS = (12, 14); N = int(float(os.environ.get("NS", str(2 ** 22)))); Bt = 8192
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(33000 + k)
PL = [l - 1 for l in LS]                       # pre-activation layer feeding layer l
sh = {}; S1 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in PL}
S2 = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in PL}
n = 0
while n < N:
    h = torch.randn(Bt, 1024, generator=gg, device=dev)
    for l in range(max(PL) + 1):
        p = h @ W[l]
        if l in S1:
            if l not in sh: sh[l] = p.mean(0)
            x = p - sh[l]
            S1[l] += x.double().sum(0); S2[l] += (x.T @ x).double()
        h = torch.relu(p)
    n += Bt
res = {"k": k, "N": n}
phi = lambda t: np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
Phi = lambda t: 0.5 * (1 + torch.erf(torch.tensor(t / math.sqrt(2))).numpy())
for l in LS:
    pl = l - 1
    m1 = (S1[pl] / n).cpu().numpy(); C = (S2[pl] / n).cpu().numpy() - np.outer(m1, m1)
    m = m1 + sh[pl].double().cpu().numpy(); s = np.sqrt(np.diag(C)); t = m / s
    c1, c2 = Phi(t), phi(t) / 2
    Wl = w[l].numpy().astype(np.float64)       # (a, j)
    V = (C @ (Wl * c1[:, None])) / s[:, None]  # v_c for every j
    Bm = Wl * c2[:, None] * V
    F = 48 * ((Bm * (C @ Bm)).sum(0) - (Bm ** 2 * np.diag(C)[:, None]).sum(0))
    a = np.load("/home/trim/wblogs/c4arr_n%d_L%d.npz" % (k, l)); rest, c4 = a["rest"], a["c4"]
    lv = np.load("/home/trim/wblogs/c3arr_n%d_L%d.npz" % (k, l))["D3"] != 0
    r = {}
    for nm, msk in (("all", np.ones(1024, bool)), ("live", lv)):
        r[nm] = {"corr_rest": float(np.corrcoef(F[msk], rest[msk])[0, 1]), "slope_rest": float(F[msk] @ rest[msk] / (F[msk] @ F[msk])),
                 "corr_c4": float(np.corrcoef(F[msk], c4[msk])[0, 1]), "rms_F_over_rest": float(np.sqrt((F[msk] ** 2).mean() / (rest[msk] ** 2).mean())),
                 "mean_F": float(F[msk].mean()), "mean_rest": float(rest[msk].mean())}
    r["corr_rest_g4"] = float(np.corrcoef(rest[lv], a["g4"][lv])[0, 1])
    res[l] = r
    print("net %d L%d %s" % (k, l, json.dumps(r)), flush=True)
    np.save("/home/trim/wblogs/chain4F_n%d_L%d.npy" % (k, l), F)
json.dump(res, open(OUT, "w"), indent=1)
