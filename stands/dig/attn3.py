"""Чи «море» κ4(p_q) (члени з різними входами: κ211, κ1111) у головному порядку = ДЕРЕВО-шлях, що факторизується через C і
згортається за n³: path_q = 12·uᵀ C u, u_b = (C (Φ⊙w))_b · w_b φ_b / σ_b  (діаграма Ерміта a1 a2 a2 a1, кінці f'=Φ, середина f''=φ/σ).
Шар L: входи h_{L-1} = ReLU(p_{L-1}); C, t, σ — передактивації p_{L-1} (з семплів). «Решта» = κ4(p_q) − діаг − 3·пари κ22 − 4·пари κ31.
Пломба: на шарі 1 p_0 ТОЧНО гаусів ⟹ дерево має пояснити решту майже цілком (кінцеві поправки: цикли ρ⁴, перетини індексів).
Передбачення до запуску: шар 1 corr(path, решта) > 0.8, нахил ~1; шар 12 corr 0.3..0.6 (там p не гаус, є перенесений κ4).
Шанс, що на шарі 12 дерево піднімає corr повного предиктора κ4 з 0.33 до > 0.7: 20%."""
import sys, numpy as np, torch
from math import sqrt, pi
dev='cuda'
i=int(sys.argv[1]); LAYERS=[int(x) for x in sys.argv[2].split(',')]
W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],dtype=torch.float32,device=dev)
N=2**20; CH=2**15
def gen(Lm1,seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    for _ in range(N//CH):
        h=torch.randn(CH,1024,generator=g,device=dev)
        for l in range(Lm1): h=torch.relu(h@W[l])
        yield h@W[Lm1]
def run(L,seed):
    S1=torch.zeros(1024,dtype=torch.float64,device=dev); Sp=torch.zeros_like(S1); Spp=torch.zeros(1024,1024,dtype=torch.float64,device=dev)
    for p in gen(L-1,seed): S1+=torch.relu(p).double().sum(0); Sp+=p.double().sum(0); Spp+=p.double().T@p.double()
    mu=(S1/N).float(); mp=Sp/N; Cp=Spp/N-torch.outer(mp,mp)
    S2=torch.zeros_like(Spp); S22=torch.zeros_like(Spp); S31=torch.zeros_like(Spp); S2d=torch.zeros_like(S1)
    P=[torch.zeros_like(S1) for _ in range(4)]
    for p in gen(L-1,seed):
        h=torch.relu(p); x=h-mu; x2=x*x
        S2+=(x.T@x).double(); S22+=(x2.T@x2).double(); S31+=((x2*x).T@x).double(); S2d+=x2.double().sum(0)
        q=(h@W[L]).double()
        for k in range(4): P[k]+=(q**(k+1)).sum(0)
    C=S2/N; v=S2d/N; K22=S22/N-torch.outer(v,v)-2*C*C; K31=S31/N-3*v[:,None]*C
    m1,m2,m3,m4=[x/N for x in P]; k4=m4-4*m1*m3+6*m1**2*m2-3*m1**4-3*(m2-m1**2)**2
    return Cp,mp,K22,K31,k4
for L in LAYERS:
    Cp,mp,K22,K31,k4a=run(L,1); _,_,_,_,k4b=run(L,2); k4=0.5*(k4a+k4b)
    Wd=W[L].double(); W2=Wd**2
    d=torch.diagonal(K22).clone(); K22o=K22.clone(); K22o.fill_diagonal_(0); K31o=K31.clone(); K31o.fill_diagonal_(0)
    pair=(W2*W2*d[:,None]).sum(0)+3*((K22o@W2)*W2).sum(0)+4*(((Wd**3).T@K31o)*Wd.T).sum(1)
    rest=k4-pair
    s=torch.diagonal(Cp).sqrt(); t=mp/s
    Ph=0.5*(1+torch.erf(t/sqrt(2))); ph=torch.exp(-t*t/2)/sqrt(2*pi)
    Co=Cp.clone(); Co.fill_diagonal_(0)
    A=Co@(Ph[:,None]*Wd)                       # (C (Φ⊙w))_b для всіх q
    U=A*Wd*(ph/s)[:,None]
    path=12*((Co@U)*U).sum(0)
    perm=torch.randperm(1024,device=dev)
    cc=lambda a,b: float(torch.corrcoef(torch.stack([a,b]))[0,1])
    sl=lambda p_,y: float((p_*y).sum()/(p_*p_).sum())
    print('\nдамп %d шар %d: повторюваність κ4 між половинами corr %.3f'%(i,L,cc(k4a,k4b)))
    print('  частки середнього: пари %.3f  решта %.3f  (від κ4)'%(float(pair.mean()/k4.mean()),float(rest.mean()/k4.mean())))
    print('  дерево-шлях проти решти: corr %.3f нахил %.3f · 🔒перемішаний шлях corr %.3f'%(cc(path,rest),sl(path,rest),cc(path[perm],rest)))
    print('  повний κ4: лише пари corr %.3f · пари+дерево corr %.3f · пари+нахил·дерево corr ~%.3f'%(cc(pair,k4),cc(pair+path,k4),cc(pair+sl(path,rest)*path,k4)))
