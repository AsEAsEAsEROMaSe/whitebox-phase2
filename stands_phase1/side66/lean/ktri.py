"""Тріваріантний κ₄(p)_{aabc} у народженні κ₂₁: T1 = κ(u_i,x_i,x_j), T2 = κ(x_i,x_i,u_j), u=(w2⊙q²)@W, x=(w1⊙q)@W, q = p_ls − a (центр.),
w1=Φ(t), w2=φ(t)/σ; мінус гаусова частина G1 = 2Σ_a W_ai w2_a M_ai M_aj, M = C(w1⊙W) (та сама для T2 з обміном). Ціль Rb_lt (lt=ls+1).
🔒 половини вибірки (стеля шуму ознаки) і перемішані w2. Критерій ДО: R²(Rb | T1−G1, T2−G2) ≥ 0.3 на lt 13..15, контроль ≈0; шанс 20%."""
import sys, math, numpy as np, torch
dev = "cuda"; i = int(sys.argv[1]); N = 2**22; CH = 2**14; LTS = [13, 14, 15]
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); W = torch.tensor(d["weights"], device=dev); L, n, _ = W.shape
kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i); P = np.load("/home/trim/wb2data/pieces_useall_%d.npz" % i)
LS = [l-1 for l in LTS]
a = {l: torch.tensor(kp["pre_mean"][l], device=dev, dtype=torch.float32) for l in LS}
var = {l: torch.tensor(kp["pre_m2"][l] - kp["pre_mean"][l]**2, device=dev, dtype=torch.float64) for l in LS}
w1, w2 = {}, {}
for l in LS:
    s = var[l].sqrt(); t = a[l].double()/s; phi = torch.exp(-0.5*t*t)/math.sqrt(2*math.pi)
    w1[l] = (0.5*(1+torch.erf(t/math.sqrt(2)))).float(); w2[l] = (phi/s).float()
perm = torch.randperm(n, generator=torch.Generator().manual_seed(3)).to(dev)
Z = lambda: torch.zeros(n, n, dtype=torch.float64, device=dev)
acc = {l: {k: [Z(), Z()] for k in ("uxx", "xxu", "xx", "uxxP", "xxuP")} for l in LS}
vec = {l: {k: [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(2)] for k in ("u", "uP", "x2")} for l in LS}
C = {l: Z() for l in LS}
g = torch.Generator(device=dev); g.manual_seed(2024 + i)
for c in range(N // CH):
    h = c % 2; z = torch.randn(CH, n, device=dev, generator=g); p = z @ W[0]
    for l in range(L):
        if l in acc:
            q = p - a[l]; Wn = W[l+1]; x = (w1[l]*q) @ Wn; u = (w2[l]*q*q) @ Wn; uP = (w2[l][perm]*q*q) @ Wn
            A = acc[l]; xx = x*x
            A["uxx"][h] += ((u*x).T @ x).double(); A["xxu"][h] += (xx.T @ u).double(); A["xx"][h] += (x.T @ x).double()
            A["uxxP"][h] += ((uP*x).T @ x).double(); A["xxuP"][h] += (xx.T @ uP).double()
            V = vec[l]; V["u"][h] += u.sum(0).double(); V["uP"][h] += uP.sum(0).double(); V["x2"][h] += xx.sum(0).double()
            C[l] += (q.T @ q).double()
        if l >= max(LS): break
        p = torch.relu(p) @ W[l+1]
off = ~torch.eye(n, dtype=torch.bool, device=dev); Nh = N/2
def r2(y, X): sol = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]; return 1 - float(((y - X@sol)**2).sum()/(y*y).sum())
for l in LS:
    A = acc[l]; V = vec[l]; Wn = W[l+1].double(); Cl = C[l]/N
    M = Cl @ (w1[l].double()[:, None]*Wn); G1 = 2*((Wn*w2[l].double()[:, None]*M).T @ M)
    MP = M; G1P = 2*((Wn*w2[l][perm].double()[:, None]*M).T @ M)
    def feat(h):
        xx = A["xx"][h]/Nh
        T1 = A["uxx"][h]/Nh - (V["u"][h]/Nh)[:, None]*xx; T2 = A["xxu"][h]/Nh - (V["x2"][h]/Nh)[:, None]*(V["u"][h]/Nh)[None, :]
        T1P = A["uxxP"][h]/Nh - (V["uP"][h]/Nh)[:, None]*xx; T2P = A["xxuP"][h]/Nh - (V["x2"][h]/Nh)[:, None]*(V["uP"][h]/Nh)[None, :]
        return T1 - G1, T2 - G1.T, T1P - G1P, T2P - G1P.T
    F0 = feat(0); F1 = feat(1); F = [0.5*(u_+v_) for u_, v_ in zip(F0, F1)]
    Rb = torch.tensor(P["Rb_%d" % (l+1)], device=dev, dtype=torch.float64)[off]
    ceil = float(torch.corrcoef(torch.stack([F0[0][off], F1[0][off]]))[0, 1])
    print("dump%d Rb_%d ‖Rb‖ %.3e | стеля шуму T1 (A↔B corr) %.3f | R²: T1 %.3f T2 %.3f оба %.3f | G1 сама %.3f | 🔒перемішані w2: оба %.3f"
          % (i, l+1, float(Rb.norm()), ceil, r2(Rb, F[0][off][:, None]), r2(Rb, F[1][off][:, None]), r2(Rb, torch.stack([F[0][off], F[1][off]], 1)),
             r2(Rb, G1[off][:, None]), r2(Rb, torch.stack([F[2][off], F[3][off]], 1))), flush=True)
    np.savez("/home/trim/wb2data/ktri_%d_%d.npz" % (i, l+1), T1=F[0].float().cpu().numpy(), T2=F[1].float().cpu().numpy())
print("DONE_KTRI", i)
