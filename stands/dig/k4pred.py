"""16.10: предиктор κ4(p_L) для всіх L=1..15 через телескоп народжень уздовж напрямків лінії назад:
κ4(p_L,j) ≈ Σ_{l<L} [гаус(l) + член κ3(l)],  гаус = own+star+path+dbl (як backline_k4.py),
член κ3 = 12·Σ_a g_a f''_a κ3(p_a, L_j, L_j)·(C(Φ⊙g))_aj,  L_j = (Φ⊙g_j)·δp_l.
C, t з банку (1e8 семплів), κ3(p_a, L, L) з 2^K семплів (ІСТИННІ входи: міряємо стелю предиктора, не ланцюг).
Вихід: /home/trim/wblogs/k4pred_raw_d{i}.npz  gs[L,n], k3[L,n], k4true[L,n] (банк).
Передбачення: у ланцюгу на чутливих 25% бере 50–75% від виграшу точного κ4 (x1.43); шанс > 75%: 20%."""
import numpy as np, torch, sys
from math import sqrt, pi
dev = 'cuda'; i = int(sys.argv[1]); N = 2**int(sys.argv[2]) if len(sys.argv) > 2 else 2**20; CH = 2**13
d = np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz' % i); kp = np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz' % i)
W = torch.tensor(d['weights'], dtype=torch.float32, device=dev); NL, n = W.shape[0], W.shape[1]
T = lambda k, l: torch.tensor(kp[k][l].astype(np.float64), device=dev)
M = [T('pre_mean', l) for l in range(NL)]; C = [T('pre_M11', l) - torch.outer(M[l], M[l]) for l in range(NL)]
s = [torch.diagonal(C[l]).sqrt() for l in range(NL)]; t = [M[l]/s[l] for l in range(NL)]
Ph = [0.5*(1+torch.erf(t[l]/sqrt(2))) for l in range(NL)]; ph = [torch.exp(-t[l]**2/2)/sqrt(2*pi) for l in range(NL)]
def k4true(l):
    a = M[l]; E2, E3, E4 = T('pre_m2', l), T('pre_m3', l), T('pre_m4', l); v = E2 - a*a
    return E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v
# напрямки G_L[l] для l<L: G_L[L-1] = W_L, G_L[l-1] = W_l diag(Φ_l) G_L[l]
G = {}; VG = {}
for L in range(1, NL):
    g_ = W[L].double()
    for l in range(L-1, -1, -1):
        G[(L, l)] = g_.float(); VG[(L, l)] = (Ph[l][:, None]*g_).float()
        if l > 0: g_ = W[l].double() @ (Ph[l][:, None]*g_)
K3A = {k: torch.zeros(n, n, dtype=torch.float32, device=dev) for k in G}
Mf = [m.float() for m in M]
gen = torch.Generator(device=dev).manual_seed(11 + i)
for c in range(N//CH):
    h = torch.randn(CH, n, generator=gen, device=dev)
    for l in range(NL-1):
        p = h @ W[l]; dp = p - Mf[l]
        for L in range(l+1, NL):
            Lc = dp @ VG[(L, l)]; K3A[(L, l)] += dp.T @ (Lc*Lc)
        h = torch.relu(p)
x = torch.linspace(-14, 14, 4001, dtype=torch.float64, device=dev); wx = torch.exp(-x*x/2); wx /= wx.sum()
gs = torch.zeros(NL, n, dtype=torch.float64, device=dev); k3 = torch.zeros_like(gs)
for l in range(NL-1):
    m_, s_, t_ = M[l], s[l], t[l]; f2 = (ph[l]/s_).float(); f3 = (-t_*ph[l]/s_**2).float()
    Cf = C[l].float(); Co = Cf.clone(); Co.fill_diagonal_(0); Co2 = Co*Co
    r = torch.relu(m_[:, None]+s_[:, None]*x[None, :]); e1 = (r*wx).sum(1); rc = r-e1[:, None]; k4own = ((rc**4*wx).sum(1)-3*((rc**2*wx).sum(1))**2).float()
    for L in range(l+1, NL):
        Gl = G[(L, l)]; V = Co @ VG[(L, l)]; Vf = Cf @ VG[(L, l)]
        own = ((Gl**4)*k4own[:, None]).sum(0); star = 4*(Gl*f3[:, None]*V**3).sum(0)
        U = Gl*f2[:, None]*V; path = 12*(U*(Co@U)).sum(0); G2 = Gl*Gl*f2[:, None]; dbl = 3*(G2*(Co2@G2)).sum(0)
        gs[L] += (own+star+path+dbl).double(); k3[L] += (12*(Gl*f2[:, None]*(K3A[(L, l)]/N)*Vf).sum(0)).double()
kt = torch.stack([k4true(l) for l in range(NL)])
cc = lambda a, b: float(torch.corrcoef(torch.stack([a, b]))[0, 1])
for L in (4, 8, 12, 15):
    v2 = torch.diagonal(C[L])**2; b = (t[L]**2-1).abs()*torch.exp(-t[L]**2/2)/(24*s[L]**3); sel = torch.topk(b*v2, n//4).indices
    print('d%d L=%2d чутл.25%%: γ4 істинний %.4f±%.4f · corr гаус %.3f · гаус+κ3 %.3f' % (i, L, float((kt[L]/v2)[sel].mean()), float((kt[L]/v2)[sel].std()), cc(gs[L][sel], kt[L][sel]), cc((gs[L]+k3[L])[sel], kt[L][sel])), flush=True)
np.savez('/home/trim/wblogs/k4pred_raw_d%d.npz' % i, gs=gs.cpu().numpy(), k3=k3.cpu().numpy(), kt=kt.cpu().numpy(),
         t=torch.stack(t).cpu().numpy(), s=torch.stack(s).cpu().numpy())
