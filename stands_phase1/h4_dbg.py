import torch, numpy as np
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
G0=W[0].t()@W[0]; std0=torch.sqrt(torch.diagonal(G0).clamp(min=1e-12))
def ff(w):
    x=w
    for li in range(L): x=relu(x@W[li])
    return x
Ff=2*D
A=torch.zeros(Ff,Ff,dtype=torch.float64); BY=torch.zeros(Ff,D,dtype=torch.float64)
SX=torch.zeros(Ff,dtype=torch.float64); SY=torch.zeros(D,dtype=torch.float64); SY2=torch.zeros(D,dtype=torch.float64); N=0
for c in range(4):
    w=torch.randn(12500,D,device=dev,generator=torch.Generator(device=dev).manual_seed(9000+c))
    Y=0.5*(ff(w)+ff(-w))
    s=(w@W[0])/std0; he2=s*s-1; he4=s**4-6*s*s+3
    X=torch.cat([he2,he4],1).double(); Yd=Y.double()
    A+=(X.t()@X).cpu(); BY+=(X.t()@Yd).cpu(); SX+=X.sum(0).cpu(); SY+=Yd.sum(0).cpu(); SY2+=(Yd*Yd).sum(0).cpu(); N+=w.shape[0]
print('A nan',torch.isnan(A).any().item(),'BY nan',torch.isnan(BY).any().item())
SStot=SY2-SY*SY/N
print('SStot min',float(SStot.min()),'max',float(SStot.max()))
idx=torch.arange(0,D)
Ai=A[idx][:,idx]; Bi=BY[idx]; SXi=SX[idx]
Ac=Ai-torch.outer(SXi,SXi)/N; Bc=Bi-torch.outer(SXi,SY)/N
print('Ac diag min',float(torch.diagonal(Ac).min()),'max',float(torch.diagonal(Ac).max()))
dsc=torch.sqrt(torch.diagonal(Ac).clamp(min=1e-12))
As=Ac/torch.outer(dsc,dsc); Bs=Bc/dsc[:,None]
print('As nan',torch.isnan(As).any().item(),'cond~',float(torch.linalg.cond(As)))
beta=torch.linalg.solve(As+1e-8*torch.eye(D,dtype=torch.float64),Bs)
print('beta nan',torch.isnan(beta).any().item())
SSres=SStot-(beta*Bs).sum(0)
print('SSres min',float(SSres.min()),'r2 mean',float((1-SSres/SStot).mean()))
