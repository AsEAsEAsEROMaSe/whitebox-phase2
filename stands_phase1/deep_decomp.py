"""Розклад «призу глибоких координат» (poly_deep.py, R2=0.3717) — held-out, накопичення моментів.
TAPS = ||a^l||^2 на 8 глибинах = ПРОФІЛЬ ГУЧНОСТІ. Скільки з призу дає ОДИН скаляр?"""
import torch, numpy as np, time
dev='cuda:1'; D=256; L=32; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True)
    open('/mnt/win_d/llm/White-Box/deep_decomp.log','a').write(s+'\n')
open('/mnt/win_d/llm/White-Box/deep_decomp.log','w').close()
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
TAPS=[3,7,11,15,19,23,27,30]
PU=torch.triu_indices(8,8,offset=1).to(dev)
def build(Z):
    """80 ознак як у poly_deep: z(8), z^2(8), z^3(8), пари(28), пари*(сума)(28)"""
    a=Z[:,PU[0]]*Z[:,PU[1]]
    return torch.cat([Z,Z*Z,Z**3,a,a*(Z[:,PU[0]]+Z[:,PU[1]])],1)
IDX={'1 лін. норма (найглибша l=30)':[7],
     '1 норма + poly (z,z²,z³)':[7,15,23],
     '8 ЛІНІЙНИХ норм':list(range(8)),
     '8 норм + poly діаг (24)':list(range(24)),
     'ПОВНИЙ набір 80 (як poly_deep)':list(range(80))}
for net in (0,1):
    W=[torch.tensor(d['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    def run(w):
        def fwd(x):
            out=[]
            for li in range(L):
                x=relu(x@W[li])
                if li in TAPS: out.append((x*x).sum(1))
            return x,out
        ap,na=fwd(w); am,nb=fwd(-w)
        return (0.5*(ap+am)).double(), torch.stack([0.5*(a+b) for a,b in zip(na,nb)],1).double()
    # прохід 1: статистики нормування T на train
    S=torch.zeros(8,dtype=torch.float64,device=dev); S2=torch.zeros(8,dtype=torch.float64,device=dev); n=0
    for c in range(3000,3040):
        w=torch.randn(10000,D,device=dev,generator=torch.Generator(device=dev).manual_seed(c))
        _,T=run(w); S+=T.sum(0); S2+=(T*T).sum(0); n+=T.shape[0]
    mu=S/n; sd=(S2/n-mu*mu).clamp(min=1e-18).sqrt()
    def moments(seeds):
        F=80
        A=torch.zeros(F,F,dtype=torch.float64,device=dev); B=torch.zeros(F,D,dtype=torch.float64,device=dev)
        SX=torch.zeros(F,dtype=torch.float64,device=dev)
        SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev); N=0
        for c in seeds:
            w=torch.randn(10000,D,device=dev,generator=torch.Generator(device=dev).manual_seed(c))
            Y,T=run(w); X=build((T-mu)/sd)
            A+=X.t()@X; B+=X.t()@Y; SX+=X.sum(0); SY+=Y.sum(0); SY2+=(Y*Y).sum(0); N+=Y.shape[0]
            del Y,T,X; torch.cuda.empty_cache()
        return A,B,SX,SY,SY2,N
    Atr,Btr,SXtr,SYtr,_,Ntr=moments(range(3000,3040))
    Ate,Bte,SXte,SYte,SY2te,Nte=moments(range(6000,6030))
    m=SYtr/Ntr
    SStot=SY2te-2*(m*SYte)+Nte*m*m
    alive=SStot>1e-9*float(SStot.max())
    log('==== НЕТ {} ====  train {}k / test {}k, held-out, alive {}/256'
        .format(net,Ntr//1000,Nte//1000,int(alive.sum())))
    for nm,idx in IDX.items():
        I=torch.tensor(idx,device=dev)
        c=(SXtr[I]/Ntr)
        Ac=Atr[I][:,I]-torch.outer(c,SXtr[I])-torch.outer(SXtr[I],c)+Ntr*torch.outer(c,c)
        Bc=Btr[I]-torch.outer(c,SYtr)-torch.outer(SXtr[I],m)+Ntr*torch.outer(c,m)
        b=torch.linalg.solve((Ac+1e-8*torch.eye(len(idx),dtype=torch.float64,device=dev)).cpu(),Bc.cpu()).to(dev)
        GG=Ate[I][:,I]-torch.outer(c,SXte[I])-torch.outer(SXte[I],c)+Nte*torch.outer(c,c)
        GY=Bte[I]-torch.outer(c,SYte)-torch.outer(SXte[I],m)+Nte*torch.outer(c,m)
        SSres=SStot-2*(b*GY).sum(0)+((GG@b)*b).sum(0)
        log('   {:36s}: {:.4f}'.format(nm,float((1-SSres/SStot)[alive].mean())))
log('done.')
