"""Гаусова частина парної уваги: чи K22(h) входів = K22 від ReLU(гаусового p з ІСТИННИМИ m,C)?
K22t: істинна (семпли мережі); K22g: семпли N(m,C) передактивацій шару L-1 з тими ж m,C, далі ReLU.
Перевірки: (1) corr позадіагоналей K22g~K22t і відношення норм; (2) κ4(p_j) для ВСІХ 1024 нейронів шару L:
 предиктор 3 w²ᵀK22off w² + Σ w⁴ κ4diag з K22t (стеля) і з K22g; R² і corr по нейронах; 🔒 контроль: K22g з перемішаними індексами.
Передбачення до запуску: corr(K22g,K22t) 0.6..0.9 (C несе більшість), κ4(p_j) з K22g R² 0.5..0.8, але перемішаний контроль
близько (масштаб), тоді носій марний (як 13.10). Шанс, що K22g бʼє контроль на ≥0.3 у corr по нейронах: 25%."""
import sys, numpy as np, torch
dev='cuda'
i=int(sys.argv[1]); LAYERS=[int(x) for x in sys.argv[2].split(',')]
W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],dtype=torch.float32,device=dev)
N=2**19; CH=2**15
def gen_true(Lm1,seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    for _ in range(N//CH):
        h=torch.randn(CH,1024,generator=g,device=dev)
        for l in range(Lm1): h=torch.relu(h@W[l])
        yield h@W[Lm1]                       # передактивація шару L-1
def gen_gauss(m,Lc,seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    for _ in range(N//CH): yield m+torch.randn(CH,1024,generator=g,device=dev)@Lc.T
def moments(gen, L):
    hs=[]; S1=torch.zeros(1024,dtype=torch.float64,device=dev)
    # двопрохідно через збереження не можна (пам'ять) ⟹ середнє з першого проходу
    return None
def k22_of(genf, L):
    S1=torch.zeros(1024,dtype=torch.float64,device=dev)
    for p in genf(): S1+=torch.relu(p).double().sum(0)
    mu=(S1/N).float(); S2=torch.zeros(1024,1024,dtype=torch.float64,device=dev); S22=torch.zeros_like(S2); S2d=torch.zeros(1024,dtype=torch.float64,device=dev); S31=torch.zeros_like(S2)
    P1=torch.zeros(1024,dtype=torch.float64,device=dev); P2=torch.zeros_like(P1); P3=torch.zeros_like(P1); P4=torch.zeros_like(P1)
    for p in genf():
        x=torch.relu(p)-mu; x2=x*x
        S2+=(x.T@x).double(); S22+=(x2.T@x2).double(); S2d+=x2.double().sum(0); S31+=((x2*x).T@x).double()
        q=(torch.relu(p)@W[L]).double(); P1+=q.sum(0); P2+=(q*q).sum(0); P3+=(q**3).sum(0); P4+=(q**4).sum(0)
    C=S2/N; v=S2d/N; K=S22/N-torch.outer(v,v)-2*C*C; K31=S31/N-3*v[:,None]*C; K31.fill_diagonal_(0); LAST['K31']=K31
    m1=P1/N; m2=P2/N-m1**2; m4=P4/N-4*m1*P3/N+6*m1**2*P2/N-3*m1**4
    return K, m4-3*m2**2
LAST={}
def pred31(K31,w): return 4*(((w**3).T@K31.double())*w.T).sum(1)
def pred(K,W2):
    d=torch.diagonal(K).clone(); Ko=K.clone(); Ko.fill_diagonal_(0)
    return 3*((Ko@W2)*W2).sum(0)+(W2*W2*d[:,None]).sum(0)
def r2(p,t): a=float((p*t).sum()/(p*p).sum()); return 1-float(((t-a*p)**2).sum()/(t*t).sum()), float(torch.corrcoef(torch.stack([p,t]))[0,1]), a
for L in LAYERS:
    S1=torch.zeros(1024,dtype=torch.float64,device=dev); S2=torch.zeros(1024,1024,dtype=torch.float64,device=dev)
    for p in gen_true(L-1,11): pd=p.double(); S1+=pd.sum(0); S2+=pd.T@pd
    m=S1/N; Cp=S2/N-torch.outer(m,m)
    ev,U=torch.linalg.eigh(Cp); Lc=(U*ev.clamp(min=0).sqrt()).float(); m=m.float()
    Kt,k4t=k22_of(lambda: gen_true(L-1,1),L); K31t=LAST['K31']; Kt2,k4t2=k22_of(lambda: gen_true(L-1,2),L)
    Kg,_=k22_of(lambda: gen_gauss(m,Lc,3),L)
    off=~torch.eye(1024,dtype=bool,device=dev)
    c_tt=float(torch.corrcoef(torch.stack([Kt[off],Kt2[off]]))[0,1]); c_gt=float(torch.corrcoef(torch.stack([Kg[off],Kt[off]]))[0,1])
    nr=float(Kg[off].norm()/Kt[off].norm()); dr=float(torch.diagonal(Kg).sum()/torch.diagonal(Kt).sum())
    W2=(W[L].double())**2; k4=0.5*(k4t+k4t2); live=k4.abs()>0
    perm=torch.randperm(1024,device=dev); Ks=Kg[perm][:,perm]
    res={nm:r2(pred(K,W2)[live],k4[live]) for nm,K in [('K22 істинна (стеля)',Kt),('K22 гаусова',Kg),('🔒 гаусова перемішана',Ks)]}
    wd=W[L].double(); p31=pred31(K31t,wd); pall=pred(Kt,W2)+p31
    res['κ22+κ31 істинні (вся пара)']=r2(pall[live],k4[live]); res['лише κ31']=r2(p31[live],k4[live])
    nt=r2(k4t[live],k4t2[live])
    print('\nдамп %d шар %d (входи h шару %d): K22 повторюваність %.3f · corr(гаус, істина) %.3f · норма гаус/істина позадіаг %.3f діаг %.3f'%(i,L,L-1,c_tt,c_gt,nr,dr))
    print('  κ4(p_j), %d живих нейронів; повторюваність κ4 між половинами R² %.3f corr %.3f'%(int(live.sum()),nt[0],nt[1]))
    for nm,(a,c,s) in res.items(): print('  %-24s R² %.3f  corr %.3f  нахил %.3f'%(nm,a,c,s))
