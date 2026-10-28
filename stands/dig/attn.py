"""«Увага між входами нейрона» (ідея юзера 15.10): для нейрона j шару L його 1024 входи h_a (шар L-1).
A2_ab = W_aj W_bj C_ab        внесок пари входів (a,b) у Var(p_j)          (аналог QK: «наскільки a і b звучать разом для j»)
A4_ab = W_aj^2 W_bj^2 κ22_ab  внесок пари у κ4(p_j) (парна частина, якої бракує ланцюгу; оракул x1.47..x20)
Метрики на нейрон: частка перехресних пар у сумі; скорочення |Σ|/Σ|·|; партнерів на вхід (PR рядка |A|); частка топ-1% пар;
контроль: стовпець W_j замінено випадковим тієї ж норми (чи структура від узгодження W і C, чи загальна).
Передбачення до запуску: увага ДИФУЗНА (партнерів ~сотні, топ-1% пар ≤10% маси), сильне скорочення знаків у A2 перехресних;
контроль дає те саме (обмінність). Шанс на розріджену увагу (топ-1% пар >30% маси): 10%."""
import sys, numpy as np, torch
dev='cuda'; torch.manual_seed(0)
i=int(sys.argv[1]) if len(sys.argv)>1 else 0
LAYERS=[int(x) for x in (sys.argv[2] if len(sys.argv)>2 else '3,12').split(',')]
W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],dtype=torch.float32,device=dev)
N=2**19; CH=2**15
def stats(Lm1, seed, mean=None):
    g=torch.Generator(device=dev).manual_seed(seed); S1=torch.zeros(1024,dtype=torch.float64,device=dev)
    S2=torch.zeros(1024,1024,dtype=torch.float64,device=dev); S22=torch.zeros_like(S2); S2d=torch.zeros(1024,dtype=torch.float64,device=dev); P2=torch.zeros(1024,dtype=torch.float64,device=dev); P4=torch.zeros_like(P2)
    for _ in range(N//CH):
        h=torch.randn(CH,1024,generator=g,device=dev)
        for l in range(Lm1+1): h=torch.relu(h@W[l])
        if mean is None: S1+=h.double().sum(0); continue
        x=h-mean.float(); x2=x*x
        S2+=(x.T@x).double(); S22+=(x2.T@x2).double(); S2d+=x2.double().sum(0)
        q=(x@W[Lm1+1]).double(); P2+=(q*q).sum(0); P4+=(q**4).sum(0)
    if mean is None: return S1/N
    C=S2/N; E22=S22/N; v=S2d/N
    k22=E22-torch.outer(v,v)-2*C*C
    k4p=P4/N-3*(P2/N)**2
    return C,k22,k4p
def metrics(A):
    A=A.clone(); d=torch.diagonal(A).clone(); A.fill_diagonal_(0); tot=d.sum()+A.sum()
    ab=A.abs(); rs=ab.sum(1); nz=rs>0; pr=(rs[nz]**2/(ab*ab).sum(1)[nz]).mean()
    v=ab[ab>0]; k=max(1,int(0.01*v.numel())); top=torch.topk(v,k).values.sum()/v.sum()
    return dict(off=float(A.sum()/tot), cancel=float(A.sum().abs()/ab.sum()), partners=float(pr), top1=float(top))
rng=np.random.default_rng(i)
for L in LAYERS:
    mean=stats(L-1,1); C,k22,k4p=stats(L-1,1,mean); Cb,k22b,k4pb=stats(L-1,2,mean)
    off=~torch.eye(1024,dtype=bool,device=dev)
    noise=float(torch.corrcoef(torch.stack([k22[off],k22b[off]]))[0,1])
    js=rng.choice(1024,8,replace=False); rows={'A2':[],'A4':[],'A2ctl':[],'A4ctl':[]}; sh=[]
    for j in js:
        w=W[L][:,j].double(); r=torch.randn(1024,dtype=torch.float64,device=dev); r*=w.norm()/r.norm()
        k=0.5*(k22+k22b)
        rows['A2'].append(metrics(torch.outer(w,w)*C)); rows['A4'].append(metrics(torch.outer(w*w,w*w)*k))
        w2=w*w; kd=torch.diagonal(k).clone(); kk=k.clone(); kk.fill_diagonal_(0)
        sh.append((float((w2*w2*kd).sum()), float(3*(w2@kk@w2)), float(0.5*(k4p[j]+k4pb[j]))))
        rows['A2ctl'].append(metrics(torch.outer(r,r)*C)); rows['A4ctl'].append(metrics(torch.outer(r*r,r*r)*k))
    print('\nшар %d (входи = h шару %d), дамп %d, N=%d; повторюваність κ22 між половинами corr %.3f'%(L,L-1,i,N,noise))
    print('  матриця     перехресні/сума   скорочення |Σ|/Σ|·|   партнерів на вхід (з 1023)   топ-1%% пар: частка маси')
    for nm,rr in rows.items():
        m={k:np.mean([x[k] for x in rr]) for k in rr[0]}
        print('  %-7s     %+.3f            %.3f                 %7.1f                       %.3f'%(nm,m['off'],m['cancel'],m['partners'],m['top1']))
    sh=np.array(sh); print('  κ4(p_j): власний ексцес входів %.3f + пари κ22 (x3) %.3f від істинного; решта (κ31, трійки, четвірки) %.3f; κ4/σ^4 сер %.3f'%(np.mean(sh[:,0]/sh[:,2]),np.mean(sh[:,1]/sh[:,2]),np.mean(1-(sh[:,0]+sh[:,1])/sh[:,2]),0))
    torch.save(dict(C=C.float().cpu(),k22=k.float().cpu(),w=W[L][:,js[0]].cpu()),'/home/trim/wblogs/attn_L%d.pt'%L)
