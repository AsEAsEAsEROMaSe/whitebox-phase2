"""Оракул «сума крон по всіх шарах»: у lean-ланцюгу (regen, як база v34) після кожного шару l<15
mu <- mu + G_k(gt_l - mu), G_k = середнє по групах розміру k (групи за t_l банку) ⟹ прибирається лише крона похибки.
Контроль: випадкові групи. Еталон: повна підстановка mu <- gt_l на кожному шарі.
Передбачення до запуску: крони k=16 дають x1.05-1.3 (сума 15 шарів по ~1%, але перенос не стискає); шанс x1.5+: 15%."""
import os, sys, numpy as np, torch
sys.argv=['lean_oracle.py','0','regen','x']; sys.path.insert(0,'/home/trim/wb2lean'); os.chdir('/home/trim/wb2lean')
import lean_k3_aug_gpu as lk
exec(compile(open('/home/trim/wb2lean/lean_oracle.py').read().split('if __name__ == "__main__":')[0],'lo','exec'))
DUMPS=[int(x) for x in os.environ.get('DUMPS','0,1,2,3').split(',')]
def run(i, mode, k):
    d=np.load(DUMP.format(i)); w=d['weights']; gtf=d['final_means'].astype(np.float64); kp=np.load(BANK.format(i))
    gt=torch.tensor(kp['gt_mean'].astype(np.float64),device=lk.DEV)
    mu_=kp['pre_mean'].astype(float); t=mu_/np.sqrt(kp['pre_m2']-mu_**2)
    rng=np.random.default_rng(i)
    groups={}
    for l in range(15):
        o=np.argsort(t[l]) if mode=='crown' else rng.permutation(1024)
        lab=np.empty(1024,int); lab[o]=np.arange(1024)//k; groups[l]=torch.tensor(lab,device=lk.DEV)
    def hook(l,mu):
        if mode=='none' or l>=15: return mu
        g=gt[l].to(mu.dtype)
        if mode=='full': return g.clone()
        r=g-mu; lab=groups[l]; G=int(lab.max())+1
        s=torch.zeros(G,dtype=mu.dtype,device=mu.device).index_add_(0,lab,r); c=torch.bincount(lab,minlength=G).to(mu.dtype)
        return mu+(s/c)[lab]
    statics=lk.build_static_coefs(w.shape[1]); lk.MU_HOOK=hook; lk.REGEN_STATE.clear(); lk.WK_HOOK=None; lk.D21_HOOK=None
    fit=FitC(); lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,g_hook=fit)
    pred=lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,aug_no_k211=True,g_hook=Replay(fit.coef),g_feed_hook=approx_pre,g_use_hook=approx_pre,shadow_B2=True,no_wk22_fac=True,aug_no_31=True,leg_trunc={"rres":16,"d21":16})
    lk.MU_HOOK=None
    return float(np.mean((pred[-1].double().cpu().numpy()-gtf)**2))
for mode,k in eval(os.environ.get('MODES','[]')):
    m=[run(i,mode,k) for i in DUMPS]; print('%-6s k=%-3d MSE %.4e  по дампах %s'%(mode,k,np.mean(m),' '.join('%.3e'%x for x in m)),flush=True)
