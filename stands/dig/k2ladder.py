"""Драбина порядків: з чого складається точність.
A: середнє поле (діагональна гаусова, C позадіагональ = 0)
B: повне двомоментне замикання (m, C), Мелер до порядку KM
C: v34-ланцюг K=3 (pred_<i>.npz)
Передбачення до запуску: B гірше за C у ×5..×30 (фаза 1: K=2 0.5%, фаза 2 вузький 1/n ×4 краще);
A гірше за B у ×10+. Шанс, що B ближче ніж ×3 до C: 15%.
"""
import numpy as np, torch, math, sys
dev='cuda'; torch.set_default_dtype(torch.float64)
KM=int(sys.argv[2]) if len(sys.argv)>2 else 8
def He(k,t):
    a,b=torch.ones_like(t),t
    if k==0: return a
    for n in range(1,k): a,b=b,t*b-n*a
    return b
def run(W,mode):
    n=W.shape[1]; m=torch.zeros(n,device=dev); C=torch.eye(n,device=dev); out=[]
    for l in range(W.shape[0]):
        Wl=W[l].T; mu=Wl@m; S=Wl@C@Wl.T; s=S.diag().clamp_min(1e-300).sqrt(); t=mu/s
        Ph=0.5*(1+torch.erf(t/math.sqrt(2))); ph=torch.exp(-t*t/2)/math.sqrt(2*math.pi)
        mh=s*(ph+t*Ph); out.append(mh)
        if l==W.shape[0]-1: break
        Eh2=s*s*((1+t*t)*Ph+t*ph); var=Eh2-mh*mh
        if mode=='diag':
            C=torch.diag(var)
        else:
            rho=S/(s[:,None]*s[None,:]); rho.fill_diagonal_(0)
            Cn=torch.zeros_like(S); rk=torch.ones_like(S); fk=1.0
            for k in range(1,KM+1):
                ck=s*Ph if k==1 else s*((-1)**k)*He(k-2,t)*ph
                rk=rk*rho; fk*=k
                Cn+= (ck[:,None]*ck[None,:])*rk/fk
            Cn.diagonal().copy_(var); C=Cn
        m=mh
    return torch.stack(out)
i=int(sys.argv[1])
W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],device=dev).double()
gt=torch.tensor(np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)['gt_mean'],device=dev).double()
# орієнтація: шар 0
r=W[0].norm(dim=1)/math.sqrt(2*math.pi); c=W[0].norm(dim=0)/math.sqrt(2*math.pi)
print('orient rows %.3e cols %.3e'%(((r-gt[0])**2).mean(),((c-gt[0])**2).mean()))
pv=np.load('/home/trim/wb2data/pred_%d.npz'%i); pc=torch.tensor(pv['pred'],device=dev).double(); gp=torch.tensor(pv['gt'],device=dev).double()
print('pred_ gt == kp gt:', float((gp-gt).abs().max()))
res={}
for mode in ['diag','full']:
    p=run(W,mode); res[mode]=((p-gt)**2).mean(1).cpu().numpy()
res['v34']=((pc-gt)**2).mean(1).cpu().numpy()
g2=(gt**2).mean(1).cpu().numpy()
print('layer  diag        full        v34        rms(gt)')
for l in [0,1,2,4,8,12,15]: print('%2d  %.3e  %.3e  %.3e  %.3f'%(l,res['diag'][l],res['full'][l],res['v34'][l],math.sqrt(g2[l])))
np.savez('/home/trim/wb2data/k2ladder_%d_K%d.npz'%(i,KM),**res,g2=g2)
