"""16.10: точні народження κ4 уздовж напрямків лінії назад (телескоп).
G^{(L-1)} = W_L, G^{(l-1)} = W_l·diag(Φ_l)·G^{(l)};  Y_l = h_l @ G^{(l)}  (h_l = ReLU(p_l)), Y_{-1} = z @ G^{(-1)} гаусова.
Y_{l-1} = (Φ_l ⊙ p_l) @ G^{(l)}  ⟹  B_l := κ4(Y_l) − κ4(Y_{l-1}) = точне народження на шарі l (нелінійність ReLU проти лінійної частини Φ),
Σ_l B_l = κ4(p_L) точно. Порівняння: гаусове народження лінії назад (own+star+path+dbl з ІСТИННИМИ C, t) на тому ж шарі.
Спільні семпли ⟹ різниця κ4 сусідніх Y має малий шум; шум міряю половинами. Зважування: чутливість останнього кроку b=|t²−1|φ/(24s³) нейронів шару L, топ-25%.
Передбачення до запуску: гаусові народження ловлять середній рівень B_l (нахил 0.7–1.0), але понейронна corr на глибоких шарах ≤0.6;
нестача росте з l (бо p_l стає негаусовим: κ3 народжує κ4). Шанс, що нестача сидить в 1–2 класах і її можна дописати формулою: 30%."""
import numpy as np, torch, sys
from math import sqrt, pi
dev = 'cuda'; i = int(sys.argv[1]) if len(sys.argv) > 1 else 0; L = int(sys.argv[2]) if len(sys.argv) > 2 else 12
N = 2**int(sys.argv[3]) if len(sys.argv) > 3 else 2**21; CH = 2**14
W = torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz' % i)['weights'], dtype=torch.float32, device=dev)
n = W.shape[1]
# прохід 1: m, C передактивацій p_0..p_L
S1 = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(L+1)]; S2 = [torch.zeros(n, n, dtype=torch.float64, device=dev) for _ in range(L+1)]
P4 = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(4)]
g = torch.Generator(device=dev).manual_seed(1)
for c in range(N//CH):
    h = torch.randn(CH, n, generator=g, device=dev)
    for l in range(L+1):
        p = h @ W[l]; pd = p.double(); S1[l] += pd.sum(0); S2[l] += pd.T @ pd; h = torch.relu(p)
M = [S1[l]/N for l in range(L+1)]; C = [S2[l]/N - torch.outer(M[l], M[l]) for l in range(L+1)]
s = [torch.diagonal(C[l]).sqrt() for l in range(L+1)]; t = [M[l]/s[l] for l in range(L+1)]
Ph = [0.5*(1+torch.erf(t[l]/sqrt(2))) for l in range(L+1)]; ph = [torch.exp(-t[l]**2/2)/sqrt(2*pi) for l in range(L+1)]
# напрямки
G = [None]*(L+1); G[L-1] = W[L].double()
for l in range(L-1, 0, -1): G[l-1] = W[l].double() @ (Ph[l][:, None]*G[l])
Gm1 = W[0].double() @ (Ph[0][:, None]*G[0])
Gf = [x.float() for x in G[:L]]; Gm1f = Gm1.float()
# прохід 2: ті самі семпли; центровані степеневі суми Y_{-1..L-1} по половинах
mu = [M[l].clamp(min=0)*0 for l in range(L)]  # заповнюємо точними середніми Y нижче
EY = [None]*(L+1)
EY[0] = torch.zeros(n, dtype=torch.float64, device=dev)            # Y_{-1}: E[z]=0
for l in range(L):
    # E[h_l] з гаусового наближення неточне ⟹ беремо емпіричне середнє: окремий накопичувач
    pass
Mf = [M[l].float() for l in range(L)]; VGf = [(Ph[l][:, None]*G[l]).float() for l in range(L)]; K3A = [torch.zeros(n, n, dtype=torch.float32, device=dev) for _ in range(L)]
SY = [[torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(4)] for _ in range(2*(L+1))]
g = torch.Generator(device=dev).manual_seed(1)
H1 = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(L)]
for c in range(N//CH):
    z = torch.randn(CH, n, generator=g, device=dev); h = z; half = c % 2
    Ys = [z @ Gm1f]
    for l in range(L):
        p = h @ W[l]
        dp = p - Mf[l]; Lc = dp @ VGf[l]; K3A[l] += dp.T @ (Lc*Lc)       # κ3(p_a, L_j, L_j), L_j = (Φ⊙g_j)·δp
        h = torch.relu(p); Ys.append(h @ Gf[l])
    for k, Y in enumerate(Ys):
        Yd = Y.double(); T = SY[2*k+half]
        for q in range(4): T[q] += (Yd**(q+1)).sum(0)
def k4of(T, m):
    m1, m2, m3, m4 = [x/m for x in T]; v = m2 - m1**2
    return m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4 - 3*v*v
K = [[k4of(SY[2*k+hf], N/2) for hf in (0, 1)] for k in range(L+1)]   # k=0 ↔ Y_{-1}, k=l+1 ↔ Y_l
Kt = [0.5*(a+b) for a, b in K]
# точний κ4(p_L) для контролю телескопа
# гаусові народження (як backline_k4.py), на шарі l уздовж G[l], з C, t шару l
x = torch.linspace(-14, 14, 4001, dtype=torch.float64, device=dev); wx = torch.exp(-x*x/2); wx /= wx.sum()
sL, tL = s[L], t[L]; bL = ((tL**2-1).abs()*torch.exp(-tL**2/2)/(24*sL**3)); sel = torch.topk(bL*torch.diagonal(C[L])**2, n//4).indices
vL2 = torch.diagonal(C[L])**2
cc = lambda a, b: float(torch.corrcoef(torch.stack([a, b]))[0, 1])
print('дамп %d, шар цілі L=%d, N=2^%d' % (i, L, int(np.log2(N))))
print('🔒 Y_{-1} гаусова: κ4/var² сер %.2e (шум)' % float((Kt[0]/vL2).abs().mean()))
tot_ex = Kt[L] - Kt[0]
print('κ4(p_L) = κ4(Y_{L-1}); γ4 на чутливих 25%%: сер %.4f, повторюваність половин corr %.3f' % (float((Kt[L]/vL2)[sel].mean()), cc(K[L][0][sel], K[L][1][sel])))
print(' l | точне B_l: сер γ4  повтор.corr | гаус: сер γ4  нахил  corr(гаус,B) на чутл. | own star path dbl (сер γ4)')
acc_g = torch.zeros(n, dtype=torch.float64, device=dev)
for l in range(L):
    B = Kt[l+1] - Kt[l]; B0 = K[l+1][0]-K[l][0]; B1 = K[l+1][1]-K[l][1]
    Gl = G[l]; m_, C_, s_, t_ = M[l], C[l], s[l], t[l]
    f2 = ph[l]/s_; f3 = -t_*ph[l]/s_**2; Co = C_.clone(); Co.fill_diagonal_(0)
    r = torch.relu(m_[:, None]+s_[:, None]*x[None, :]); e1 = (r*wx).sum(1); rc = r-e1[:, None]; k4own = (rc**4*wx).sum(1)-3*((rc**2*wx).sum(1))**2
    V = Co @ (Ph[l][:, None]*Gl)
    own = ((Gl**4)*k4own[:, None]).sum(0); star = 4*(Gl*f3[:, None]*V**3).sum(0)
    U = Gl*f2[:, None]*V; path = 12*(U*(Co@U)).sum(0); G2 = Gl*Gl*f2[:, None]; dbl = 3*(G2*((Co*Co)@G2)).sum(0)
    K3 = K3A[l].double()/N; Vf = C_ @ (Ph[l][:, None]*Gl)
    k3t = 12*(Gl*f2[:, None]*K3*Vf).sum(0)
    gb0 = own+star+path+dbl; gb = gb0 + k3t; acc_g += gb
    print('   l=%2d член κ3: сер γ4 %+.5f · corr(гаус,B) %.3f → (гаус+κ3,B) %.3f · нахил %.3f · CV-фіт a·гаус+b·κ3: corr %.3f' % (l, float((k3t/vL2)[sel].mean()), cc(gb0[sel], (Kt[l+1]-Kt[l])[sel]), cc(gb[sel], (Kt[l+1]-Kt[l])[sel]), float((gb[sel]*(Kt[l+1]-Kt[l])[sel]).sum()/(gb[sel]**2).sum()), cc(torch.linalg.lstsq(torch.stack([gb0[sel], k3t[sel]], 1), (Kt[l+1]-Kt[l])[sel][:, None]).solution.squeeze() @ torch.stack([gb0[sel], k3t[sel]]), (Kt[l+1]-Kt[l])[sel])))
    q = lambda v_: float((v_/vL2)[sel].mean())
    sl = float((gb[sel]*B[sel]).sum()/(gb[sel]**2).sum())
    print('%2d | %+.5f  %.3f | %+.5f  %.3f  %.3f | %+.5f %+.5f %+.5f %+.5f' % (l, q(B), cc(B0[sel], B1[sel]), q(gb), sl, cc(gb[sel], B[sel]), q(own), q(star), q(path), q(dbl)))
print('сума: точне %.5f · гаус %.5f · corr(сума гаус, κ4) на чутл. %.3f' % (float((tot_ex/vL2)[sel].mean()), float((acc_g/vL2)[sel].mean()), cc(acc_g[sel], Kt[L][sel])))
np.savez('/home/trim/wblogs/birth_exact_d%d_L%d.npz' % (i, L), K=torch.stack([torch.stack(k) for k in K]).cpu().numpy(), sel=sel.cpu().numpy(), vL2=vL2.cpu().numpy())
