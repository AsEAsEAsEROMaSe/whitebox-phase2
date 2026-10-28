"""Універсальні ваги членів «лінії назад» для κ4(p_L): ознаки own/star/path/dbl з ІСТИННИХ C,t банку (pre_M11, pre_mean, pre_m2),
ціль = істинний κ4 банку; фіт зважений чутливістю E[ReLU] до κ4: b = (t²−1)φ/(24 s³); пул шарів 2..15, LOO по дампах.
Пише dig/backline_coef.json: {"<dump>": [c_own,c_star,c_path,c_dbl]} (коеф. фітовані БЕЗ цього дампа)."""
import numpy as np, torch, json
from math import sqrt, pi
dev='cuda'
def feats(i):
    W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],dtype=torch.float64,device=dev)
    kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)
    st=[]
    for l in range(16):
        a=torch.tensor(kp['pre_mean'][l].astype(float),device=dev); E2=torch.tensor(kp['pre_m2'][l].astype(float),device=dev)
        M11=torch.tensor(kp['pre_M11'][l].astype(float),device=dev); C=M11-torch.outer(a,a); C.diagonal().copy_(E2-a*a)
        st.append((a,C))
    x=torch.linspace(-14,14,4001,dtype=torch.float64,device=dev); wx=torch.exp(-x*x/2); wx/=wx.sum()
    pre=[]
    for l in range(16):
        m,C=st[l]; s=torch.diagonal(C).sqrt(); t=m/s; Ph=0.5*(1+torch.erf(t/sqrt(2))); ph=torch.exp(-t*t/2)/sqrt(2*pi)
        Co=C.clone(); Co.fill_diagonal_(0); r=torch.relu(m[:,None]+s[:,None]*x[None,:]); e1=(r*wx).sum(1); rc=r-e1[:,None]
        k4own=(rc**4*wx).sum(1)-3*((rc**2*wx).sum(1))**2
        pre.append(dict(Ph=Ph,f2=ph/s,f3=-t*ph/s**2,Co=Co,k4own=k4own,s=s,t=t,ph=ph))
    out={}
    for L in range(2,16):
        G=W[L]; F=torch.zeros(4,1024,dtype=torch.float64,device=dev)
        for l in range(L-1,-1,-1):
            P=pre[l]; V=P['Co']@(P['Ph'][:,None]*G)
            F[0]+=((G**4)*P['k4own'][:,None]).sum(0); F[1]+=4*(G*P['f3'][:,None]*V**3).sum(0)
            U=G*P['f2'][:,None]*V; F[2]+=12*(U*(P['Co']@U)).sum(0); G2=G*G*P['f2'][:,None]; F[3]+=3*(G2*((P['Co']**2)@G2)).sum(0)
            if l>0: G=W[l]@(P['Ph'][:,None]*G)
        a=kp['pre_mean'][L].astype(float); E2=kp['pre_m2'][L].astype(float); E3=kp['pre_m3'][L].astype(float); E4=kp['pre_m4'][L].astype(float)
        v=E2-a*a; s=np.sqrt(v); t=a/s; k4=E4-4*a*E3+6*a*a*E2-3*a**4-3*v*v
        b=(t*t-1)*np.exp(-t*t/2)/sqrt(2*pi)/(24*s**3)
        out[L]=(F.cpu().numpy().T,k4,b)
    return out
D={i:feats(i) for i in range(8)}
def stack(ids):
    X=np.concatenate([D[i][L][0]*D[i][L][2][:,None] for i in ids for L in D[i]]); y=np.concatenate([D[i][L][1]*D[i][L][2] for i in ids for L in D[i]]); return X,y
coef={}
for k in range(8):
    X,y=stack([j for j in range(8) if j!=k]); coef[str(k)]=np.linalg.lstsq(X,y,rcond=None)[0].tolist()
print('коеф. (own,star,path,dbl) по фолдах:'); [print(' ',k,['%.3g'%c for c in v]) for k,v in coef.items()]
for L in [4,8,12,15]:
    r=[];rl=[]
    for k in range(8):
        X,k4,b=D[k][L]; p=X@np.array(coef[str(k)]); r.append(1-np.sum((b*(k4-p))**2)/np.sum((b*(k4-k4.mean()))**2))
    print('шар %2d: зважений R² понад середнє (LOO дамп) %.3f'%(L,np.mean(r)))
json.dump(coef,open('/home/trim/wb2lean/dig/backline_coef.json','w'))
