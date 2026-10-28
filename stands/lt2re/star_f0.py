"""star_f0 (21.10.2026 f0, ідея 5f): діаграма «зірка» c3·c1³ для чотириіндексної частини c4 передактивацій (abcd з aabc_all_f0).
h_a = ReLU(p_a) = Σ_k c_k He_k(u_a), c1 = sΦ(t), c2 = sφ/2, c3 = −s t φ/6. Дерева з трьома ребрами ρ:
  зірка  κ(a;b,c,d) = 6 c3_a c1_b c1_c c1_d ρ_ab ρ_ac ρ_ad  ⟹  S_j = 24 Σ_a W_aj c3_a u_aj³,  u_aj = Σ_{b≠a} ρ_ab c1_b W_bj
  ланцюг (як chain4_f0)  F_j = 48 Σ_{c≠d} B_c C_cd B_d,  B = W∘c2∘(C(W∘c1))/s
(m, C) передактивацій шару l−1 справжні (МК 2^21). Порівняння з abcd_true, aabc_true, c4_true по шарах 1..15.
Пломба: на шарі 1 вхід гаусів точно ⟹ дерева мусять давати знак і порядок величини abcd.
ПЕРЕДБАЧЕННЯ: corr(S, abcd) 0.2…0.5 на ш8..15, нижче порогу 0.6 (ланцюг дав 0.07…0.17 проти rest). Шанс ≥ 0.6: 20%.
usage: star_f0.py <k> <out.json>"""
import json, os, sys, math
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]; LS = tuple(range(1, 16)); N = int(float(os.environ.get("NS", str(2 ** 21)))); Bt = 4096
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(88000 + k)
PL = [l - 1 for l in LS]; sh = {}
S1 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in PL}; S2 = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in PL}
n = 0
while n < N:
    h = torch.randn(Bt, 1024, generator=gg, device=dev)
    for l in range(max(PL) + 1):
        p = h @ W[l]
        if l not in sh: sh[l] = p.mean(0)
        x = p - sh[l]; S1[l] += x.double().sum(0); S2[l] += (x.T @ x).double()
        h = torch.relu(p)
    n += Bt
ms = {l: ((S1[l] / n).cpu().numpy(), (S2[l] / n).cpu().numpy(), sh[l].double().cpu().numpy()) for l in PL}
del S1, S2, W; torch.cuda.empty_cache()
ab = np.load("/home/trim/wblogs/abcd_all_n%d.npz" % k)
phi = lambda t: np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
Phi = lambda t: 0.5 * (1 + torch.erf(torch.tensor(t / math.sqrt(2))).numpy())
def cc(a, b): return float(np.corrcoef(a, b)[0, 1])
def sl(a, b): return float(a @ b / (a @ a))
res = {"k": k, "N": n}
for l in LS:
    m1, M2, s0 = ms[l - 1]; C = M2 - np.outer(m1, m1); m = m1 + s0; s = np.sqrt(np.diag(C)); t = m / s
    c1, c2, c3 = s * Phi(t), s * phi(t) / 2, -s * t * phi(t) / 6
    Wl = w[l].numpy().astype(np.float64)
    R = C / np.outer(s, s); np.fill_diagonal(R, 0.0)
    u = R @ (Wl * c1[:, None])                          # u_aj = Σ_{b≠a} ρ_ab c1_b W_bj
    S = 24 * (Wl * c3[:, None] * u ** 3).sum(0)
    V = (C @ (Wl * (c1 / s)[:, None])) / s[:, None]     # = C(W∘Φ)/s, як у chain4_f0
    Bm = Wl * (c2 / s)[:, None] * V
    F = 48 * ((Bm * (C @ Bm)).sum(0) - (Bm ** 2 * np.diag(C)[:, None]).sum(0))
    abcd, aabc, c4 = ab["abcd"][l], ab["aabc"][l], ab["c4"][l]
    X = np.stack([S, F], 1); bet = np.linalg.lstsq(X, abcd, rcond=None)[0]; fit = X @ bet
    r = {"corr_S_abcd": cc(S, abcd), "slope_S_abcd": sl(S, abcd), "rms_S_over_abcd": float(np.sqrt((S ** 2).mean() / (abcd ** 2).mean())),
         "corr_F_abcd": cc(F, abcd), "corr_SF_abcd": cc(fit, abcd), "beta_SF": bet.tolist(),
         "corr_S_aabc": cc(S, aabc), "corr_S_c4": cc(S, c4), "mean_S": float(S.mean()), "mean_abcd": float(abcd.mean()),
         "var_abcd_over_c4": float(abcd.var() / c4.var()), "med_t": float(np.median(t))}
    res[l] = r
    print("net %d L%d %s" % (k, l, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()})), flush=True)
    np.savez("/home/trim/wblogs/star_n%d_L%d.npz" % (k, l), S=S, F=F)
json.dump(res, open(OUT, "w"), indent=1)
