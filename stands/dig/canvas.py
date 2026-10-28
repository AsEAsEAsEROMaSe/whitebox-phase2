"""Полотно ряду по нейронах (юзер 15.10): m1 (середнє поле), m2 (K=2), m3 (v34) на ВСІХ шарах; t з K=2 (лише з ваг).
Тест прицілу: чи |d3|=|m3-m2| (останній обчислений член ряду) вказує, ДЕ похибка |e3|=|m3-gt|?
  corr(|d3|,|e3|) по нейронах; частка MSE у топ-10% за |d3| проти випадкових 10%; 🔒 контроль: |d3| перемішаний.
Передбачення до запуску: частка у топ-10% за |d3| 12..20% (трохи більше 10%, бо |d3| несе масштаб нейрона),
після нормування на масштаб (|d3|/|m3| проти |e3|/|m3|) ≈ контроль. Шанс, що топ-10% за |d3| бере >30% MSE: 10%."""
import numpy as np, torch, math, sys, json
sys.argv=[sys.argv[0],'0','6']; exec(open('/home/trim/wb2lean/dig/k2ladder.py').read().split("i=int(sys.argv[1])")[0])
def run_t(W):
    n=W.shape[1]; m=torch.zeros(n,device=dev); C=torch.eye(n,device=dev); ts=[]
    for l in range(W.shape[0]):
        Wl=W[l].T; mu=Wl@m; S=Wl@C@Wl.T; s=S.diag().clamp_min(1e-300).sqrt(); t=mu/s; ts.append(t)
        Ph=0.5*(1+torch.erf(t/math.sqrt(2))); ph=torch.exp(-t*t/2)/math.sqrt(2*math.pi); mh=s*(ph+t*Ph)
        Eh2=s*s*((1+t*t)*Ph+t*ph); var=Eh2-mh*mh
        rho=S/(s[:,None]*s[None,:]); rho.fill_diagonal_(0); Cn=torch.zeros_like(S); rk=torch.ones_like(S); fk=1.0
        for k in range(1,7):
            ck=s*Ph if k==1 else s*((-1)**k)*He(k-2,t)*ph; rk=rk*rho; fk*=k; Cn+=(ck[:,None]*ck[None,:])*rk/fk
        Cn.diagonal().copy_(var); C=Cn; m=mh
    return torch.stack(ts).cpu().numpy()
rng=np.random.default_rng(0); R={L:[] for L in [4,8,12,15]}; CAN=None
for i in range(8):
    W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],device=dev).double()
    gt=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)['gt_mean'].astype(float)
    m1=run(W,'diag').cpu().numpy(); m2=run(W,'full').cpu().numpy(); m3=np.load('/home/trim/wb2data/pred_%d.npz'%i)['pred'].astype(float)
    d3=np.abs(m3-m2); d2=np.abs(m2-m1); e3=np.abs(m3-gt)
    for L in R:
        e2=e3[L]**2; tot=e2.sum(); k=102
        top=e2[np.argsort(-d3[L])[:k]].sum()/tot
        topn=e2[np.argsort(-(d3[L]/(np.abs(m3[L])+1e-12)))[:k]].sum()/tot     # нормоване на масштаб
        topm=e2[np.argsort(-np.abs(m3[L]))[:k]].sum()/tot                      # просто великі нейрони
        ctl=e2[rng.permutation(1024)[:k]].sum()/tot
        c=np.corrcoef(d3[L],e3[L])[0,1]; cn=np.corrcoef(d3[L]/(np.abs(m3[L])+1e-12),e3[L]/(np.abs(m3[L])+1e-12))[0,1]
        R[L].append((top,topn,topm,ctl,c,cn))
    if i==0:
        t=run_t(W); o=np.argsort(t,axis=1)
        sc=np.sqrt((gt**2).mean(1,keepdims=True))
        tk=lambda a: np.take_along_axis(a,o,1)
        CAN=dict(t=np.round(tk(t),2).tolist(), ratio=np.round(tk(np.log10((d3+1e-14)/(d2+1e-14))),2).tolist(),
                 err=np.round(tk(np.log10(e3/sc+1e-14)),2).tolist(), d3=np.round(tk(np.log10(d3/sc+1e-14)),2).tolist())
print('шар | MSE у топ-10%: за |d3|  за |d3|/|m|  за |m| (масштаб)  🔒випадкові | corr(|d3|,|e3|)  нормоване')
for L,v in R.items():
    a=np.array(v).mean(0); print('%2d  |  %.3f   %.3f   %.3f   %.3f  |  %+.3f  %+.3f'%(L,*a))
json.dump(CAN,open('/home/trim/wblogs/canvas0.json','w'))
