"""aabc_f0 (21.10.2026 white-box-f0, ідея 5f): точна частка тріїндексних κ(a,a,b,c) у rest = c4 - diag - aabb - aaab
передактивацій, прямим МК без тензора. Те саме МК дає c4, rest і aabc ⟹ шум спільний.
aabc_j = 6 Σ_{a,b,c різні} W_a² W_b W_c κ(a,a,b,c),  κ = E[x_a² x_b x_c] − v_a C_bc − 2 C_ab C_ac  (x центр. на справжньому m)
різні = усі − (b=c) − (a=b) − (a=c) + 2(a=b=c);  усі для M = E[(x²@W²)·(x@W)²].
ПЕРЕДБАЧЕННЯ: aabc бере 40…70% дисперсії rest. Шанс ≥ 50%: 55%.
usage: aabc_f0.py <k> <out.json>"""
import json, os, sys, math
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]; LS = (12, 14); N = int(float(os.environ.get("NS", str(2 ** 23)))); Bt = 8192
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
dev = "cuda"; W = w.to(dev); W2 = W * W; gg = torch.Generator(device=dev).manual_seed(44000 + k)
mt = {l: torch.tensor(m_true[l - 1], dtype=torch.float32, device=dev) for l in LS}
Z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
acc = {l: {"x2": Z(1024), "x4": Z(1024), "Q11": Z(1024, 1024), "Q22": Z(1024, 1024), "Q31": Z(1024, 1024),
           "p1": Z(1024), "p2": Z(1024), "p3": Z(1024), "p4": Z(1024), "sq2": Z(1024)} for l in LS}
n = 0
while n < N:
    h = torch.randn(Bt, 1024, generator=gg, device=dev)
    for l in range(max(LS) + 1):
        if l in acc:
            a = acc[l]; x = h - mt[l]; x2 = x * x; q = (x @ W[l]).double(); s = (x2 @ W2[l]).double()
            a["x2"] += x2.double().sum(0); a["x4"] += (x2 * x2).double().sum(0)
            a["Q11"] += (x.T @ x).double(); a["Q22"] += (x2.T @ x2).double(); a["Q31"] += ((x2 * x).T @ x).double()
            a["p1"] += q.sum(0); a["p2"] += (q ** 2).sum(0); a["p3"] += (q ** 3).sum(0); a["p4"] += (q ** 4).sum(0)
            a["sq2"] += (s * q * q).sum(0)
        if l < max(LS): h = torch.relu(h @ W[l])
    n += Bt
res = {"k": k, "N": n}
for l in LS:
    a = {kk: (v / n).cpu().numpy() for kk, v in acc[l].items()}
    Wl = w[l].numpy().astype(np.float64); Wl2 = Wl ** 2
    q1, q2, q3, q4 = a["p1"], a["p2"], a["p3"], a["p4"]; v2 = q2 - q1 ** 2
    c4 = q4 - 4 * q1 * q3 + 6 * q1 ** 2 * q2 - 3 * q1 ** 4 - 3 * v2 ** 2
    # x центроване на справжньому m ⟹ E[x] ≈ 0 до шуму; моменти x беремо як є
    C = a["Q11"]; vx = np.diag(C).copy(); x4 = a["x4"]; Q22, Q31 = a["Q22"], a["Q31"]
    off = ~np.eye(1024, dtype=bool)
    k4d = x4 - 3 * vx ** 2; kaabb = Q22 - np.outer(vx, vx) - 2 * C ** 2; kaaab = Q31 - 3 * vx[:, None] * C
    p4d = (Wl ** 4 * k4d[:, None]).sum(0)
    p22 = 3 * (Wl2 * (np.where(off, kaabb, 0) @ Wl2)).sum(0)
    p31 = 4 * ((Wl ** 3) * (np.where(off, kaaab, 0) @ Wl)).sum(0)
    rest = c4 - p4d - p22 - p31
    # M_dist = Σ_{різні} W_a²W_bW_c E[x_a² x_b x_c]
    Mall = a["sq2"]
    Mbc = (Wl2 * (Q22 @ Wl2)).sum(0)                 # b=c: Σ_{a,b} W_a²W_b² E[x_a² x_b²]
    Mab = ((Wl ** 3) * (Q31 @ Wl)).sum(0)            # a=b: Σ_{a,c} W_a³W_c E[x_a³ x_c]
    Mabc = (Wl ** 4 * x4[:, None]).sum(0)
    Mdist = Mall - Mbc - 2 * Mab + 2 * Mabc
    # P1 = Σ_{різні} W_a² v_a W_b W_c C_bc
    A = (Wl2 * vx[:, None]).sum(0); u = C @ Wl; WCW = (Wl * u).sum(0)
    Eab = ((Wl ** 3) * vx[:, None] * u).sum(0)        # Σ_{a,c} W_a³ v_a W_c C_ac
    E4 = (Wl ** 4 * (vx ** 2)[:, None]).sum(0)
    P1 = A * WCW - (Wl2 * vx[:, None]).sum(0) ** 2 - 2 * Eab + 2 * E4
    # P2 = Σ_{різні} W_a² W_b W_c C_ab C_ac
    P2 = (Wl2 * u ** 2).sum(0) - (Wl2 * ((C ** 2) @ Wl2)).sum(0) - 2 * Eab + 2 * E4
    aabc = 6 * (Mdist - P1 - 2 * P2)
    abcd = rest - aabc
    lv = np.load("/home/trim/wblogs/c3arr_n%d_L%d.npz" % (k, l))["D3"] != 0
    old = np.load("/home/trim/wblogs/c4arr_n%d_L%d.npz" % (k, l))
    r = {}
    for nm, msk in (("all", np.ones(1024, bool)), ("live", lv)):
        vr = rest[msk].var()
        r[nm] = {"var_rest_over_c4": float(vr / c4[msk].var()), "share_aabc": float(aabc[msk].var() / vr), "share_abcd": float(abcd[msk].var() / vr),
                 "corr_aabc_rest": float(np.corrcoef(aabc[msk], rest[msk])[0, 1]), "corr_aabc_abcd": float(np.corrcoef(aabc[msk], abcd[msk])[0, 1]),
                 "corr_g4_aabc": float(np.corrcoef(old["g4"][msk], aabc[msk])[0, 1]), "corr_g4_abcd": float(np.corrcoef(old["g4"][msk], abcd[msk])[0, 1]),
                 "corr_g4_rest": float(np.corrcoef(old["g4"][msk], rest[msk])[0, 1]),
                 "seal_c4_vs_old": float(np.corrcoef(c4[msk], old["c4"][msk])[0, 1]), "seal_rest_vs_old": float(np.corrcoef(rest[msk], old["rest"][msk])[0, 1])}
    res[l] = r
    print("net %d L%d %s" % (k, l, json.dumps(r)), flush=True)
    np.savez("/home/trim/wblogs/aabc_n%d_L%d.npz" % (k, l), c4=c4, rest=rest, aabc=aabc, abcd=abcd)
json.dump(res, open(OUT, "w"), indent=1)
