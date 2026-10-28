"""k4pat (c2, 22.10.2026): розклад справжніх c4/c3 ПЕРЕДАКТИВАЦІЙ шарів 12..14 за структурою індексів κ4/κ3 попередніх
постактивацій x (МК, n×n-накопичувачі): c4(p_i)=Σ W W W W κ4(x): diag · (3,1) · (2,2) · (2,1,1) · (1,1,1,1);
c3(p_i): diag · (2,1) · (1,1,1). Порівняння із залишками ланцюга R4 = c4t − g4row, R3 = c3t − D3 (chpair дампи).
ПЕРЕДБАЧЕННЯ: R4 ≈ P211 + P1111 (corr > 0.8), P211 ≫ P1111; R3 ≈ P111 (corr > 0.7). usage: k4pat.py <k> <out.npz>"""
import os, sys, math, json
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]; LS = (12, 13, 14); N = int(float(os.environ.get("NS", str(2 ** 22)))); B = 8192
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(31000 + k)
zc = np.load("/home/trim/wblogs/chpair_n%d.npz" % k)
n = 1024
acc = {l: dict(m2=torch.zeros(n, dtype=torch.float64, device=dev), m3=torch.zeros(n, dtype=torch.float64, device=dev), m4=torch.zeros(n, dtype=torch.float64, device=dev),
               C=torch.zeros(n, n, dtype=torch.float64, device=dev), M21=torch.zeros(n, n, dtype=torch.float64, device=dev), M31=torch.zeros(n, n, dtype=torch.float64, device=dev),
               M22=torch.zeros(n, n, dtype=torch.float64, device=dev), V=torch.zeros(n, n, dtype=torch.float64, device=dev), U=torch.zeros(n, n, dtype=torch.float64, device=dev),
               p2=torch.zeros(n, dtype=torch.float64, device=dev), p3=torch.zeros(n, dtype=torch.float64, device=dev), p4=torch.zeros(n, dtype=torch.float64, device=dev)) for l in LS}
sh = {}; done = 0
while done < N:
    h = torch.randn(B, n, generator=gg, device=dev)
    for l in range(16):
        if l in LS:
            if l not in sh: sh[l] = h.mean(0)
            x = h - sh[l]; p = x @ W[l]; a = acc[l]
            x2 = x * x; p2 = p * p
            a["m2"] += x2.sum(0, dtype=torch.float64); a["m3"] += (x2 * x).sum(0, dtype=torch.float64); a["m4"] += (x2 * x2).sum(0, dtype=torch.float64)
            a["C"] += (x.T @ x).double(); a["M21"] += (x2.T @ x).double(); a["M31"] += ((x2 * x).T @ x).double(); a["M22"] += (x2.T @ x2).double()
            a["V"] += (p2.T @ x).double(); a["U"] += (p2.T @ x2).double()
            a["p2"] += p2.sum(0, dtype=torch.float64); a["p3"] += (p2 * p).sum(0, dtype=torch.float64); a["p4"] += (p2 * p2).sum(0, dtype=torch.float64)
        h = torch.relu(h @ W[l])
    done += B
res = {"k": k, "N": done}
for l in LS:
    a = {kk: (v / done) for kk, v in acc[l].items()}; Wl = W[l].double()   # p_i = Σ_j x_j W[j,i]
    m2, m3, m4 = a["m2"], a["m3"], a["m4"]; C, M21, M31, M22, V, U = a["C"], a["M21"], a["M31"], a["M22"], a["V"], a["U"]
    # cumulants of centered x
    K4d = m4 - 3 * m2 * m2; K31 = M31 - 3 * m2[:, None] * C; K22 = M22 - m2[:, None] * m2[None, :] - 2 * C * C
    K31.fill_diagonal_(0); K22.fill_diagonal_(0)
    W2, W3, W4 = Wl * Wl, Wl ** 3, Wl ** 4
    # c4 parts (index i = next-layer neuron: columns of W)
    P_diag = W4.T @ K4d                                   # Σ_j W_ji^4 κ4_jjjj
    P_31 = 4 * ((W3.T @ K31) * Wl.T).sum(1)               # 4 Σ_{j≠k} W_ji^3 W_ki κ4_jjjk  -> Σ_j W3[j,i] Σ_k K31[j,k] W[k,i]
    P_22 = 3 * ((W2.T @ K22) * W2.T).sum(1)               # 3 Σ_{j≠k} W_ji^2 W_ki^2 κ4_jjkk
    varp = (Wl.T @ C @ Wl).diagonal()                      # Σ_kl W_ki W_li C_kl
    CW = C @ Wl                                            # (C W)[j,i]
    Uc = U - varp[:, None] * m2[None, :] - 2 * (CW.T) ** 2  # Uc[i,j] = Σ_kl W_ki W_li κ4_jjkl (all k,l)
    K22f = K22 + torch.diag(K4d)                           # κ4_jjkk incl. k=j
    S1 = (K22f @ W2).T                                     # S1[i,j] = Σ_k W_ki^2 κ4_jjkk
    K31f = K31 + torch.diag(K4d)
    S2 = 2 * Wl.T * ((K31f @ Wl).T - Wl.T * K4d[None, :]) # 2 W_ji (Σ_{l≠j} W_li κ4_jjjl)
    D4 = Uc - S1 - S2
    P_211 = 6 * (W2.T * D4).sum(1)
    c4_direct = a["p4"] - 3 * a["p2"] ** 2
    P_1111 = c4_direct - (P_diag + P_31 + P_22 + P_211)
    seal4 = ((W2.T * Uc).sum(1)) - (P_diag + P_22 / 3 + P_31 / 2 + P_211 / 6)   # identity on bookkeeping
    # c3 parts
    K3d = m3
    P_d3 = W3.T @ K3d
    P_21 = 3 * ((W2.T @ M21) * Wl.T).sum(1)              # 3 Σ_{j≠k} W_ji^2 W_ki κ3_jjk ; M21 diag = E[x^3] must be excluded
    P_21 = P_21 - 3 * (W3.T @ K3d)
    D3m = V - (M21.T @ W2).T.T if False else None
    # D3[i,j] = V[i,j] − Σ_k W_ki^2 M21[k,j] − 2 W_ji ((M21 W)[j,i] − W_ji m3_j)
    T1 = (M21.T @ W2).T                                   # T1[i,j] = Σ_k W_ki^2 M21[k,j]
    T2 = 2 * Wl.T * ((M21 @ Wl).T - Wl.T * m3[None, :])
    D3 = V - T1 - T2
    P_111 = (Wl.T * D3).sum(1)
    c3_direct = a["p3"]
    seal3 = c3_direct - (P_d3 + P_21 + P_111)
    out = {kk: v.cpu().numpy() for kk, v in dict(P_diag=P_diag, P_31=P_31, P_22=P_22, P_211=P_211, P_1111=P_1111, c4_direct=c4_direct,
                                                   P_d3=P_d3, P_21=P_21, P_111=P_111, c3_direct=c3_direct, varp=varp).items()}
    c4t, c3t, g4, D3c = zc["c4t"][l].astype(np.float64), zc["c3t"][l].astype(np.float64), zc["g4_%d" % l].astype(np.float64), zc["D3_%d" % l].astype(np.float64)
    R4, R3 = c4t - g4, c3t - D3c
    def cr(x, y): return float(np.corrcoef(x, y)[0, 1])
    def lsq(y, cols):
        X = np.stack(cols, 1); b = np.linalg.lstsq(X, y, rcond=None)[0]; return b, 1 - np.mean((y - X @ b) ** 2) / np.var(y)
    rep = dict(layer=l, seal3_rel=float(np.linalg.norm(seal3.cpu().numpy()) / np.linalg.norm(out["c3_direct"])), seal4_rel=float(np.linalg.norm(seal4.cpu().numpy()) / np.linalg.norm(out["c4_direct"])),
               c4_direct_vs_chpair=cr(out["c4_direct"], c4t), c3_direct_vs_chpair=cr(out["c3_direct"], c3t),
               norms4={kk: float(np.linalg.norm(out[kk])) for kk in ("P_diag", "P_31", "P_22", "P_211", "P_1111", "c4_direct")}, normR4=float(np.linalg.norm(R4)),
               norms3={kk: float(np.linalg.norm(out[kk])) for kk in ("P_d3", "P_21", "P_111", "c3_direct")}, normR3=float(np.linalg.norm(R3)),
               corrR4={kk: cr(R4, out[kk]) for kk in ("P_diag", "P_31", "P_22", "P_211", "P_1111")}, corrR3={kk: cr(R3, out[kk]) for kk in ("P_d3", "P_21", "P_111")},
               fitR4_211_1111=lsq(R4, [out["P_211"], out["P_1111"]])[1], fitR4_all=lsq(R4, [out[kk] for kk in ("P_diag", "P_31", "P_22", "P_211", "P_1111")])[1],
               coefR4_211_1111=lsq(R4, [out["P_211"], out["P_1111"]])[0].tolist(),
               fitR3_111=lsq(R3, [out["P_111"]])[1], fitR3_all=lsq(R3, [out[kk] for kk in ("P_d3", "P_21", "P_111")])[1], coefR3_111=lsq(R3, [out["P_111"]])[0].tolist(),
               g4_vs_tree=cr(g4, out["P_diag"] + out["P_31"] + out["P_22"]), g4_vs_c4=cr(g4, c4t), D3_vs_c3=cr(D3c, c3t))
    print(json.dumps(rep), flush=True)
    for kk, v in out.items(): res["%s_%d" % (kk, l)] = v.astype(np.float32)
    res["rep_%d" % l] = json.dumps(rep)
np.savez(OUT, **res); print("saved", OUT, flush=True)
