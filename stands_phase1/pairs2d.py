"""ПАРИ НАПРЯМКІВ У БАЗИСІ МЕРЕЖІ — діра, яку ніхто не міряв.

ЛОГІКА:
 1) [[finding-noise-is-one-scalar]]: ціль звузилась до СКАЛЯРА α(x)=a^L(x)·v*, де v* —
    безкоштовний атрактор; 65%/50% усього шуму саме там.
 2) quartic3.py міряв 4-й порядок у СТАНДАРТНОМУ базисі x ⟹ нуль. Але «мало індексів»
    ЗАЛЕЖИТЬ ВІД БАЗИСУ: діагональне в базисі мережі виглядає як нуль у стандартному.
 3) strata.py перевірив ОДИНИЧНІ напрямки базису мережі ⟹ нуль. ПАРИ — не перевіряв ніхто.
 4) f(x·w1, x·w2) має ТОЧНЕ середнє через 2D квадратуру ⟹ безкоштовний якір
    ([[finding-anchor-cost-identity]] не діє). І можна брати не He2He2, а ОПТИМАЛЬНУ
    функцію — саме умовне середнє E[α|t1,t2].

ЩО МІРЯЄМО: частку Var(α_even), яку бере E[α|t_i,t_j], ПОНАД усе, що ZCA вже має
(знімаємо {1, t_i, t_j, t_i²−1, t_j²−1, t_i t_j} зваженою регресією по бінах).
КОНТРОЛІ: (a) shuffle міток бінів; (b) ВИПАДКОВІ напрямки; (c) ПОЗИТИВНИЙ — підсадка
He2(t_i)He2(t_j) відомої амплітуди (бо «нуль» і «зламаний прилад» виглядають однаково).
"""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time(); NB=24; NS=400000
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a

def content2d(t1,t2,a,tag,nrep=1):
    """частка Var(a), яку бере E[a|t1,t2] понад порядок ≤2. Повертає (сира, понад-2, shuffle)"""
    n=t1.shape[0]
    def lab(t):
        idx=torch.argsort(t); l=torch.empty_like(idx); l[idx]=torch.arange(n,device=dev)
        return (l*NB)//n
    l1,l2=lab(t1),lab(t2); cell=l1*NB+l2
    oh=torch.zeros(NB*NB,n,dtype=torch.float64,device=dev)
    oh[cell,torch.arange(n,device=dev)]=1.0
    cnt=oh.sum(1); ok=cnt>20
    a=a-a.mean(); v=float((a*a).mean())
    mb=(oh@a)/cnt.clamp(min=1)
    w=(cnt/n)[ok]; y=mb[ok]
    raw=float((w*y*y).sum())
    # базис порядку ≤2 у центрах бінів
    c1=((oh@t1)/cnt.clamp(min=1))[ok]; c2=((oh@t2)/cnt.clamp(min=1))[ok]
    B=torch.stack([torch.ones_like(c1),c1,c2,c1*c1-1,c2*c2-1,c1*c2],1)
    Wd=torch.diag(w); G=B.t()@Wd@B; rhs=B.t()@(w*y)
    coef=torch.linalg.solve(G+1e-12*torch.eye(6,dtype=torch.float64,device=dev),rhs)
    res=y-B@coef
    beyond=float((w*res*res).sum())
    # shuffle-контроль
    sh=0.0
    for r in range(nrep):
        g=torch.Generator(device=dev).manual_seed(7000+r)
        p=torch.randperm(n,device=dev,generator=g)
        mb2=(oh[:,p]@a)/cnt.clamp(min=1); y2=mb2[ok]
        res2=y2-B@torch.linalg.solve(G+1e-12*torch.eye(6,dtype=torch.float64,device=dev),B.t()@(w*y2))
        sh+=float((w*res2*res2).sum())/nrep
    return 100*raw/v, 100*(beyond-sh)/v, 100*sh/v

for net in (0,1):
    W=load(net); log('='*78); log('МЕРЕЖА %d'%net)
    g=torch.Generator(device=dev).manual_seed(4242)
    X=torch.randn(8,D,device=dev,generator=g); X=X/X.norm(dim=1,keepdim=True)
    for _ in range(3):
        Y=fwd(W,X); X=Y/Y.norm(dim=1,keepdim=True).clamp(min=1e-30)
    vstar=(X[0]/X[0].norm())
    # активний підпростір парної частини
    M=torch.zeros(D,D,dtype=torch.float64,device=dev)
    for c in range(10):
        gg=torch.Generator(device=dev).manual_seed(31000+c)
        x=torch.randn(2000,D,device=dev,generator=gg,requires_grad=True)
        vv=torch.randn(2000,D,device=dev,generator=gg)
        r=0.5*(fwd(W,x)+fwd(W,-x)); s=(r*vv).sum()
        gr,=torch.autograd.grad(s,x); gr=gr.double(); M+=gr.t()@gr
    ev,V=torch.linalg.eigh(M); V=V.flip(1)
    U=torch.cat([vstar.double().unsqueeze(1), V[:,:7]],1)          # v* + топ-7 активних
    U,_=torch.linalg.qr(U)
    gg=torch.Generator(device=dev).manual_seed(99)
    UR,_=torch.linalg.qr(torch.randn(D,8,device=dev,generator=gg,dtype=torch.float64))
    # семпли
    A=[];T1=[];T2=[]
    Ts=[];As=[]
    for c in range(NS//20000):
        gg=torch.Generator(device=dev).manual_seed(52000+c)
        x=torch.randn(20000,D,device=dev,generator=gg)
        with torch.no_grad(): r=0.5*(fwd(W,x)+fwd(W,-x))
        As.append((r.double()@vstar.double()))
        Ts.append(torch.stack([x.double()@U, x.double()@UR],0))
    al=torch.cat(As); TT=torch.cat(Ts,1)  # 2 × 8 × N
    log('  Var(α_even) = {:.4e};  напрямки: 0=v*, 1-7=активні'.format(float(al.var())))
    best=[]
    for tagi,(name,idx) in enumerate((('БАЗИС МЕРЕЖІ',0),('ВИПАДКОВІ (контроль)',1))):
        log('  --- {} ---'.format(name))
        log('   пара     сира 2D    ПОНАД порядок-2    shuffle')
        rows=[]
        for i in range(8):
            for j in range(i+1,8):
                raw,bey,sh=content2d(TT[idx,:,i],TT[idx,:,j],al,'')
                rows.append((bey,i,j,raw,sh))
        rows.sort(reverse=True)
        for bey,i,j,raw,sh in rows[:6]:
            log('   {:d}-{:d}   {:9.3f}%      {:9.4f}%     {:8.4f}%'.format(i,j,raw,bey,sh))
        log('   СУМА понад-2 по всіх 28 парах: {:.3f}%'.format(sum(max(r[0],0) for r in rows)))
        best.append(sum(max(r[0],0) for r in rows))
    # ПОЗИТИВНИЙ КОНТРОЛЬ
    t1,t2=TT[0,:,0],TT[0,:,1]
    inj=(t1*t1-1)*(t2*t2-1)
    inj=inj/inj.std()*float(al.std())*0.05                          # 0.25% дисперсії
    raw,bey,sh=content2d(t1,t2,al+inj,'')
    log('  ПОЗИТИВНИЙ КОНТРОЛЬ: підсадка He2·He2 на 0.25% дисперсії ⟹ понад-2 = {:.4f}%'.format(bey))
    log('  ⟹ базис мережі {:.3f}%  проти контролю {:.3f}%'.format(best[0],best[1]))
