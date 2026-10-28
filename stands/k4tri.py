"""Тривісний блок κ₄[a,a,b,c] (a,b,c різні) на 32 нейронах S (brc, істинні кумулянти): чи є він деревною функцією
n²-об'єктів і κ₃ (стенд 26.09). Кандидати (безрозмірно узгоджені, κ₄ ~ κ₂²): t1 = D21[a,b]·D21[a,c]/var_a (κ₃κ₃/κ₂ з центром a),
t2 = κ₃[a,b,c]·D3[a]/var_a, t3 = C_ab·C_ac, t4 = var_a·C_bc, t5 = D21[b,a]·C_ac + D21[c,a]·C_ab (κ₃·C), t6 = κ₃[a,b,c]·(D21[b,a]+D21[c,a])/... ні.
R² одиночно і спільно (МНК), коефіцієнти по дампах. Потім референс k_max=3 з κ₄ = n² + деревне наближення тривісного проти істини.
Передбачення ДО: найкращий одиночний t1 або t2 з R² 0.5–0.8 (шанс 50%); референс з наближенням 4–6% проти 2.95/10.8."""
import sys, os, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
torch.set_default_dtype(torch.float64)
m = 32
def load(fn):
    D = np.load(fn); Nh = float(D["Nh"]); a = torch.tensor(D["a"])
    g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
    gh = lambda h, k: torch.tensor(D["%d_%s" % (h, k)]) / Nh
    def cums(g_):
        e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m)
        C2 = R2 - torch.outer(e, e)
        C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
        k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
        hm = g_("h1"); H2 = g_("h2"); H3 = g_("h3").reshape(m, m, m)
        K3h = H3 - (H2[:, :, None]*hm[None, None, :] + H2[:, None, :]*hm[None, :, None] + H2[None, :, :]*hm[:, None, None]) + 2*hm[:, None, None]*hm[None, :, None]*hm[None, None, :]
        return a + e, C2, C3, k4, K3h
    return cums(g_), cums(lambda k: gh(0, k)), cums(lambda k: gh(1, k))
ar = torch.arange(m)
A, B, Cc = torch.meshgrid(ar, ar, ar, indexing="ij"); dist = (A != B) & (A != Cc) & (B != Cc)
iu = torch.tensor([[x, y, z] for x in range(m) for y in range(m) for z in range(m) if x < y < z]); sel = lambda T: T[iu[:, 0], iu[:, 1], iu[:, 2]]
def cands(mean, C2, C3, k4):
    var = torch.diagonal(C2); D21 = torch.einsum("aac->ac", C3).clone(); D3 = torch.einsum("aaa->a", C3).clone(); Coff = C2.clone(); Coff.fill_diagonal_(0.0)
    K22 = torch.einsum("aabb->ab", k4); K31 = torch.einsum("aaab->ab", k4); K4 = torch.einsum("aaaa->a", k4)
    t = {}
    t["t1 D21ab·D21ac/var_a"] = D21[:, :, None] * D21[:, None, :] / var[:, None, None]
    t["t2 κ3abc·D3a/var_a"] = C3 * (D3 / var)[:, None, None]
    t["t3 Cab·Cac"] = Coff[:, :, None] * Coff[:, None, :]
    t["t4 var_a·Cbc"] = var[:, None, None] * Coff[None, :, :]
    t["t5 D21ba·Cac+D21ca·Cab"] = D21.T[:, :, None] * Coff[:, None, :] + D21.T[:, None, :] * Coff[:, :, None]
    t["t6 K31ab·Cac+K31ac·Cab /var_a"] = (K31[:, :, None] * Coff[:, None, :] + K31[:, None, :] * Coff[:, :, None]) / var[:, None, None]
    t["t7 K22ab·Cac+K22ac·Cab /var_a"] = (K22[:, :, None] * Coff[:, None, :] + K22[:, None, :] * Coff[:, :, None]) / var[:, None, None]
    t["t8 κ3abc·K4a/D3a"] = C3 * (K4 / D3)[:, None, None]
    t["t9 κ3abc·(K31ab+K31ac)/(D21ab+D21ac)"] = C3 * ((K31[:, :, None] + K31[:, None, :]) / (D21[:, :, None] + D21[:, None, :] + 1e-30))
    return t
def r2(x, y): a = float((x*y).sum()/(x*x).sum()); return 1 - float(((y - a*x)**2).sum()/(y*y).sum()), a
files = sys.argv[1:] or ["/home/trim/wb2data/brc_0_12.npz", "/home/trim/wb2data/brc_1_12.npz", "/home/trim/wb2data/brc_2_14.npz"]
for fn in files:
    (mean, C2, C3, k4, K3h), H0, H1 = load(fn)
    y = torch.einsum("aabc->abc", k4)[dist]           # тривісний блок, центр a
    ynoise = float(((torch.einsum("aabc->abc", H0[3]) - torch.einsum("aabc->abc", H1[3]))[dist]).norm() / 2 / y.norm())
    t = cands(mean, C2, C3, k4)
    print("%s: ‖κ4 tri‖ %.3e шум %.3f" % (os.path.basename(fn), float(y.norm()), ynoise))
    X = []
    for nm, T in t.items():
        x = T[dist]; r, a = r2(x, y); X.append(x); print("   %-42s R² %.3f coef %.3f" % (nm, r, a))
    Xm = torch.stack(X, 1); coef = torch.linalg.lstsq(Xm, y[:, None]).solution[:, 0]; pred = Xm @ coef
    print("   спільно R² %.3f coef %s" % (1 - float(((y - pred)**2).sum()/(y*y).sum()), " ".join("%.2f" % c for c in coef)))
    # референс із наближеним тривісним блоком
    ndistinct = None
    a4 = torch.arange(m); A4, B4, C4, D4 = torch.meshgrid(a4, a4, a4, a4, indexing="ij")
    nd = 1 + (B4 != A4).int() + ((C4 != A4) & (C4 != B4)).int() + ((D4 != A4) & (D4 != B4) & (D4 != C4)).int()
    k4n2 = k4 * (nd <= 2)
    def tri_from(T3):   # тензор T3[a,b,c] (центр a) -> 4-тензор з блоками aabc у всіх 6 положеннях пари
        T = torch.zeros_like(k4); Tm = T3 * dist
        for pa in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 2, 3, 1), (1, 2, 0, 3), (1, 2, 3, 0), (2, 3, 0, 1)]:
            pass
        # явно: κ[a,a,b,c], [a,b,a,c], [a,b,c,a], [b,a,a,c], [b,a,c,a], [b,c,a,a]
        idx = torch.nonzero(dist); a_, b_, c_ = idx[:, 0], idx[:, 1], idx[:, 2]; v = Tm[a_, b_, c_]
        for pos in [(a_, a_, b_, c_), (a_, b_, a_, c_), (a_, b_, c_, a_), (b_, a_, a_, c_), (b_, a_, c_, a_), (b_, c_, a_, a_)]:
            T[pos] = v
        return T
    assert float((tri_from(torch.einsum("aabc->abc", k4)) - k4 * (nd == 3)).norm()) < 1e-9 * float(k4.norm()), "пломба tri_from"
    def ref(k4v):
        return relu_kprop(DSTower.from_tower({1: mean, 2: C2, 3: C3, 4: k4v}), k_max=3)[3].to_tensor()
    e = lambda P: float((sel(P) - sel(K3h)).norm() / sel(K3h).norm())
    print("   референс (1,1,1): κ4 повна %.4f · n² %.4f · n²+tri істинний %.4f" % (e(ref(k4)), e(ref(k4n2)), e(ref(k4n2 + k4 * (nd == 3)))))
    for nm, T in list(t.items())[:2] + [("спільно", None)]:
        Ta = (Xm @ coef).new_zeros(m, m, m) if T is None else T
        if T is None:
            Ta = torch.zeros(m, m, m); Ta[dist] = pred
        else:
            r, a = r2(T[dist], y); Ta = a * T
        print("   референс (1,1,1) з κ4 = n² + tri≈%-28s %.4f" % (nm, e(ref(k4n2 + tri_from(Ta)))))
