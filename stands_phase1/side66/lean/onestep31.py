"""Одношаговий розклад (3,1)-зрізу κ₄ передактивацій: κ₃₁[i,j] = κ(p_i,p_i,p_i,p_j). s = x + y (x = лінійна частина c1·pc@W, y = r@W).
Чиста x-частина = точний лінійний перенос κ₄(p_l) (потребує κ₄-стеку); решта = народження. Пломба: full ~ банк κ₃₁ R²>0.98 на глибині.
Питання: κ₃₁ переноситься (частка pure-x ≥0.8 ⟹ носій = κ₄-стек ⟹ ⛔ ціною) чи народжується (тоді шукати n²-замикання народження)?
Передбачення: pure-x 0.6–0.8 (шанс 50%); народження замикається n²-кандидатами на R² ≤0.5."""
import os, sys, time, numpy as np, torch
dev = "cuda"; i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N = int(os.environ.get("N", str(2**20))); CH = 2**15; seed = 1
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); W = torch.tensor(d["weights"], device=dev)
kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i); L, n, _ = W.shape
def fwd(gen):
    z = torch.randn(CH, n, device=dev, generator=gen); p = z @ W[0]
    for l in range(L):
        yield l, p
        if l < L-1: p = torch.relu(p) @ W[l+1]
g = torch.Generator(device=dev); g.manual_seed(seed)
S = {k: torch.zeros(L, n, dtype=torch.float64, device=dev) for k in ("p", "p2", "h", "hp")}
for c in range(N // CH):
    for l, p in fwd(g):
        h = torch.relu(p); pd = p.double(); hd = h.double()
        S["p"][l] += pd.sum(0); S["p2"][l] += (pd*pd).sum(0); S["h"][l] += hd.sum(0); S["hp"][l] += (hd*pd).sum(0)
a = S["p"]/N; var = S["p2"]/N - a*a; c0 = S["h"]/N; c1 = (S["hp"]/N - c0*a)/var
acc = lambda u, w: (u.T @ w).double()
keys = ("s3s", "x3x", "ss", "xx", "s2s2", "x2x2")
P = {k: torch.zeros(L, n, n, dtype=torch.float64, device=dev) for k in keys}
V = {k: torch.zeros(L, n, dtype=torch.float64, device=dev) for k in ("s2", "x2")}
CS = torch.zeros(L, n, n, dtype=torch.float64, device=dev); KS = torch.zeros(L, n, n, dtype=torch.float64, device=dev); D3S = torch.zeros(L, n, dtype=torch.float64, device=dev)
g = torch.Generator(device=dev); g.manual_seed(seed); t0 = time.perf_counter()
for c in range(N // CH):
    for l, p in fwd(g):
        pc = p - a[l].float(); CS[l] += acc(pc, pc); KS[l] += acc(pc*pc, pc); D3S[l] += (pc*pc*pc).double().sum(0)
        if l == L-1: break
        h = torch.relu(p); r = h - c0[l].float() - c1[l].float()*pc
        x = (c1[l].float()*pc) @ W[l+1]; y = r @ W[l+1]; s = x + y
        P["s3s"][l+1] += acc(s*s*s, s); P["x3x"][l+1] += acc(x*x*x, x); P["ss"][l+1] += acc(s, s); P["xx"][l+1] += acc(x, x)
        P["s2s2"][l+1] += acc(s*s, s*s); P["x2x2"][l+1] += acc(x*x, x*x); V["s2"][l+1] += (s*s).double().sum(0); V["x2"][l+1] += (x*x).double().sum(0)
for k in P: P[k] /= N
for k in V: V[k] /= N
CS /= N; KS /= N; D3S /= N
print("pass %.0fs" % (time.perf_counter()-t0), flush=True)
offm = ~torch.eye(n, dtype=torch.bool, device=dev)
def r2(x, y):
    a_ = float((x*y).sum()/(x*x).sum()); return 1 - float(((y - a_*x)**2).sum()/(y*y).sum()), a_
def nrm(M): return float(M[offm].pow(2).sum().sqrt())
def k31(P3, P1, v): return P3 - 3*v[:, None]*P1
def k22(P22, P1, v): return P22 - v[:, None]*v[None, :] - 2*P1*P1
for l in range(2, L):
    K31 = k31(P["s3s"][l], P["ss"][l], V["s2"][l]); K31x = k31(P["x3x"][l], P["xx"][l], V["x2"][l]); B31 = K31 - K31x
    K22 = k22(P["s2s2"][l], P["ss"][l], V["s2"][l]); K22x = k22(P["x2x2"][l], P["xx"][l], V["x2"][l]); B22 = K22 - K22x
    ab = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64)
    M11 = kp["pre_M11"][l].astype(np.float64); M21 = kp["pre_M21"][l].astype(np.float64); M31 = kp["pre_M31"][l].astype(np.float64)
    C11 = M11 - np.outer(ab, ab); vb = E2 - ab*ab
    mu31 = M31 - 3*ab[:, None]*M21 + 3*(ab*ab)[:, None]*M11 - ab[None, :]*E3[:, None] + 3*ab[None, :]*(ab[:, None]*E2[:, None]) - 3*np.outer(ab**3, ab)
    k31b = torch.tensor(mu31 - 3*vb[:, None]*C11, device=dev)
    rb, sb = r2(K31[offm], k31b[offm]); nT = nrm(K31)
    rx, sx = r2(K31x[offm], K31[offm])
    line = "l=%2d пломба κ₃₁~банк R² %.4f нахил %.3f | ‖pure-x‖/‖κ₃₁‖ %.3f (R² %.3f), ‖народж.‖/‖κ₃₁‖ %.3f | κ₂₂: pure-x %.3f народж. %.3f" % (
        l, rb, sb, nrm(K31x)/nT, rx, nrm(B31)/nT, nrm(K22x)/nrm(K22), nrm(B22)/nrm(K22))
    # n²-замикання народження κ₃₁ через об'єкти шару l−1 (істинні з семплів): C, K=D21, D3, κ₄diag
    lp = l-1; Wl = W[l].double(); Cl = CS[lp]; Kl = KS[lp].clone(); Kl.fill_diagonal_(0.0); D3l = D3S[lp]; vl = var[lp]; c = c1[lp]
    Ct = c[:, None]*Cl*c[None, :]; Kt = (c*c)[:, None]*c[None, :]*Kl; Cn = Wl.T @ Ct @ Wl; Kn = Wl.T @ Kt @ Wl; kap = c**3*D3l; kW = kap @ Wl
    cands = {"Kn*Cn": Kn*Cn, "KnT*Cn": Kn.T*Cn, "Kn*Kn": Kn*Kn, "Kn*KnT": Kn*Kn.T, "dKn*Cn": Kn.diagonal()[:, None]*Cn, "Cn*dKnj": Cn*Kn.diagonal()[None, :],
             "Cn*kWi": Cn*kW[:, None], "Cn*kWj": Cn*kW[None, :], "Cn*Cn*Cn": Cn*Cn*Cn, "(WW)K(W)": (Wl*Wl).T @ (Kt @ Wl), "W(KW)W": (Wl*(Kt@Wl)).T @ Wl,
             "B31x": K31x, "Cn": Cn, "Cn*Cn": Cn*Cn}
    y = B31[offm]; X = torch.stack([v[offm] for v in cands.values()], 1); ss_ = float((y*y).sum())
    def fit(cols):
        Xc = X[:, cols]; sol = torch.linalg.lstsq(Xc, y[:, None]).solution[:, 0]; return 1 - float(((y - Xc@sol)**2).sum()/ss_)
    nm = list(cands); line += " | ЗАМИКАННЯ народж.: " + " ".join("%s %.2f" % (k, fit([j])) for j, k in enumerate(nm)) + " | ВСІ %.3f" % fit(list(range(len(nm))))
    # і повний κ₃₁ на тих самих кандидатах (memoryless як у нього)
    y = K31[offm]; ss_ = float((y*y).sum()); line += " | ПОВНИЙ κ₃₁ ВСІ %.3f" % fit(list(range(len(nm))))
    print(line, flush=True)
