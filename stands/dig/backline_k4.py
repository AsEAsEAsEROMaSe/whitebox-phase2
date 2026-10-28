"""«Пустити лінію назад» (юзер 15.10): κ4(p_L,j) ≈ Σ_{l<L} Σ_abcd g_a g_b g_c g_d · κ4_народж,l(h_a,h_b,h_c,h_d),
g = G^{(l)}[:,j], G^{(L-1)} = W_L, G^{(l)} = W_{l+1}·diag(Φ_{l+1})·G^{(l+1)} (лінійний перенос h_l → p_L).
Народження з гаусового p_l (C, t, σ ІСТИННІ з семплів, щоб міряти саму ідею): власний ексцес κ4(ReLU) (1-D точно),
зірка 4·Σ_a g_a f'''_a v_a³, шлях 12·uᵀC_off u, v = C_off(Φ⊙g), u = g⊙f''⊙v; f''=φ/σ, f'''=−tφ/σ². Ціна n³ на шар для всіх j.
Порівняння з ІСТИННИМ κ4(p_L) по нейронах; база: пари κ22+κ31 давали corr 0.33 (attn2). Шар 12, дамп 0, N=2^20.
Передбачення до запуску: corr 0.4..0.6 (перенос лінійний, без κ3·κ3 і нелінійного переносу); шанс > 0.8: 15%."""
import numpy as np, torch
from math import sqrt, pi
dev='cuda'; i=0; L=12
W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],dtype=torch.float32,device=dev)
N=2**20; CH=2**15
S1=[torch.zeros(1024,dtype=torch.float64,device=dev) for _ in range(L+1)]; S2=[torch.zeros(1024,1024,dtype=torch.float64,device=dev) for _ in range(L)]
Q=[torch.zeros(1024,dtype=torch.float64,device=dev) for _ in range(4)]; Q2=[torch.zeros(1024,dtype=torch.float64,device=dev) for _ in range(4)]
g=torch.Generator(device=dev).manual_seed(1)
for c in range(N//CH):
    h=torch.randn(CH,1024,generator=g,device=dev)
    for l in range(L+1):
        p=h@W[l]; pd=p.double(); S1[l]+=pd.sum(0)
        if l<L: S2[l]+=pd.T@pd; h=torch.relu(p)
        else:
            T=Q if c%2==0 else Q2
            for k in range(4): T[k]+=(pd**(k+1)).sum(0)
def k4of(T,n):
    m1,m2,m3,m4=[x/n for x in T]; return m4-4*m1*m3+6*m1**2*m2-3*m1**4-3*(m2-m1**2)**2
k4a=k4of(Q,N/2); k4b=k4of(Q2,N/2); k4t=0.5*(k4a+k4b)
x=torch.linspace(-14,14,4001,dtype=torch.float64,device=dev); wx=torch.exp(-x*x/2); wx/=wx.sum()
G=W[L].double(); parts={'own':torch.zeros(1024,dtype=torch.float64,device=dev),'star':torch.zeros_like(S1[0]),'path':torch.zeros_like(S1[0]),'dbl':torch.zeros_like(S1[0])}
for l in range(L-1,-1,-1):
    m=S1[l]/N; C=S2[l]/N-torch.outer(m,m); s=torch.diagonal(C).sqrt(); t=m/s
    Ph=0.5*(1+torch.erf(t/sqrt(2))); ph=torch.exp(-t*t/2)/sqrt(2*pi); f2=ph/s; f3=-t*ph/s**2
    Co=C.clone(); Co.fill_diagonal_(0)
    r=torch.relu(m[:,None]+s[:,None]*x[None,:]); e1=(r*wx).sum(1); rc=r-e1[:,None]; k4own=(rc**4*wx).sum(1)-3*((rc**2*wx).sum(1))**2
    V=Co@(Ph[:,None]*G)
    parts['own']+=((G**4)*k4own[:,None]).sum(0)
    parts['star']+=4*(G*f3[:,None]*V**3).sum(0)
    U=G*f2[:,None]*V; parts['path']+=12*(U*(Co@U)).sum(0)
    G2=G*G*f2[:,None]; parts['dbl']+=3*(G2*((Co*Co)@G2)).sum(0)
    if l>0: G=W[l].double() @ (Ph[:,None]*G)     # h_{l-1} → p_l → (Φ_l) → h_l → … → p_L
cc=lambda a,b: float(torch.corrcoef(torch.stack([a,b]))[0,1]); sl=lambda p_,y: float((p_*y).sum()/(p_*p_).sum())
tot=parts['own']+parts['star']+parts['path']+parts['dbl']
print('шар %d: повторюваність κ4 corr %.3f; середнє κ4 істинне %.3e'%(L,cc(k4a,k4b),float(k4t.mean())))
for nm,v in list(parts.items())+[('разом',tot)]:
    print('  %-6s середнє/істинне %.3f  corr з κ4 по нейронах %.3f  нахил %.3f'%(nm,float(v.mean()/k4t.mean()),cc(v,k4t),sl(v,k4t)))
perm=torch.randperm(1024,device=dev); print('  🔒 разом перемішане corr %.3f'%cc(tot[perm],k4t))

# перехресна перевірка по нейронах: фіт ваг членів на половині нейронів, R² на другій (і навпаки), 20 розбиттів
X=torch.stack([torch.ones_like(k4t)]+[parts[k] for k in ['own','star','path','dbl']],1).cpu().numpy(); y=k4t.cpu().numpy()
def cv(cols):
    rs=[]; rng=np.random.default_rng(0)
    for _ in range(20):
        pm=rng.permutation(1024); a,b=pm[:512],pm[512:]
        for tr,te in [(a,b),(b,a)]:
            c=np.linalg.lstsq(X[tr][:,cols],y[tr],rcond=None)[0]; r=y[te]-X[te][:,cols]@c; rs.append(1-r.var()/y[te].var())
    return np.mean(rs)
rep=cc(k4a,k4b)**2
print('  повторювана дисперсія (стеля R²) %.3f'%rep)
for nm,cols in [('own',[0,1]),('own+star',[0,1,2]),('own+path',[0,1,3]),('own+dbl',[0,1,4]),('усі 4',[0,1,2,3,4])]:
    print('  CV R² %-9s %.3f'%(nm,cv(cols)))
np.savez('/home/trim/wblogs/backline_L%d_d%d.npz'%(L,i),X=X,y=y)
