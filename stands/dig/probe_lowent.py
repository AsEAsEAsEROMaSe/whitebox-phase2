"""Зонди з низькою ентропією (юзер 15.10): вхід z = sqrt(n/k)·ξ Qᵀ, ξ~N(0,I_k), Q — випадковий k-вимірний підпростір (слід Cov = n, як у справжнього).
Еквівалентно мережі з W0' = sqrt(n/k) Q Qᵀ W0 на звичайному вході ⟹ ланцюг рахує без змін.
Істина ТОЧНО (однорідність): E[out] = E‖ξ‖ · E_{θ∈S^{k-1}} out(θ) (k=1: два прогони; k≥2: багато точок на сфері).
Питання: чи схожа похибка ланцюга на зонді на його похибку на справжньому вході (corr по нейронах, шар 15)?
Передбачення до запуску: на k ≤ 8 похибка ланцюга величезна (розподіли далекі від гауса) і з повною corr ~0; шанс corr > 0.3: 5%."""
import os, sys, numpy as np, torch, math
sys.argv=['lean_oracle.py','0','regen','x']; sys.path.insert(0,'/home/trim/wb2lean'); os.chdir('/home/trim/wb2lean')
import lean_k3_aug_gpu as lk
exec(compile(open('/home/trim/wb2lean/lean_oracle.py').read().split('if __name__ == "__main__":')[0],'lo','exec'))
def chain(w):
    statics=lk.build_static_coefs(w.shape[1]); lk.REGEN_STATE.clear(); lk.WK_HOOK=None; lk.D21_HOOK=None; lk.MU_HOOK=None
    fit=FitC(); lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,g_hook=fit)
    p=lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,aug_no_k211=True,g_hook=Replay(fit.coef),g_feed_hook=approx_pre,g_use_hook=approx_pre,shadow_B2=True,no_wk22_fac=True,aug_no_31=True,leg_trunc={"rres":16,"d21":16})
    return p[-1].double().cpu().numpy()
i=0; d=np.load(DUMP.format(i)); w=d['weights'].astype(np.float64); gtf=d['final_means'].astype(np.float64); n=1024
e_full=np.load('/home/trim/wb2data/pred_%d.npz'%i)['pred'][15].astype(float)-np.load(BANK.format(i))['gt_mean'][15].astype(float)
Wt=torch.tensor(w,dtype=torch.float64,device=lk.DEV)
def fwd(X):   # X: (B, n) входи
    h=X
    for l in range(16): h=torch.relu(h@Wt[l])
    return h
rng=np.random.default_rng(7)
for k in [1,2,4,8]:
    Q,_=np.linalg.qr(rng.standard_normal((n,k))); A=math.sqrt(n/k)*Q            # z = ξ Aᵀ
    w2=w.copy(); w2[0]=(A@A.T@w[0])*(1.0) / (n/k) * (n/k)                      # W0' = A Aᵀ W0 /(n/k)·(n/k)… див. нижче
    w2[0]=math.sqrt(n/k)*(Q@(Q.T@w[0]))                                          # z_full W0' = (z_full Q)·sqrt(n/k)QᵀW0
    pc=chain(w2.astype(np.float32) if dtype==torch.float32 else w2)
    At=torch.tensor(A,device=lk.DEV)
    if k==1:
        u=At[:,0][None,:]; truth=((fwd(u)+fwd(-u))/math.sqrt(2*math.pi))[0].cpu().numpy(); se=0
    else:
        Er=math.sqrt(2)*math.exp(math.lgamma((k+1)/2)-math.lgamma(k/2)); acc=torch.zeros(n,dtype=torch.float64,device=lk.DEV); acc2=torch.zeros_like(acc); M=0
        g=torch.Generator(device=lk.DEV).manual_seed(11)
        for _ in range(64):
            th=torch.randn(16384,k,generator=g,device=lk.DEV,dtype=torch.float64); th=th/th.norm(dim=1,keepdim=True)
            o=fwd(th@At.T); acc+=o.sum(0); acc2+=(o*o).sum(0); M+=o.shape[0]
        mean=acc/M; truth=(Er*mean).cpu().numpy(); se=float((Er**2*(acc2/M-mean**2)/M).mean())
    e=pc-truth
    c=np.corrcoef(e,e_full)[0,1]
    print('k=%d: MSE ланцюга на зонді %.3e (шум істини %.1e) · rms(істина) %.3f · corr(e_зонд, e_повний) %+.3f'%(k,np.mean(e**2),se,np.sqrt(np.mean(truth**2)),c),flush=True)
