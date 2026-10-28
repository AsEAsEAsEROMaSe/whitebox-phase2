"""Зв'язок похибки середніх МІЖ шарами (питання 15.10 після «обмінна по нейронах»).
Два прогони ланцюга (regen як база v34): none (запис mu_l) і full (mu <- gt_l після кожного шару ⟹ похибка шару l+1 = народження b_{l+1}).
Перенос T_l e = Phi(t_{l+1}) * (e @ W[l+1]) (лінеаризація ReLU-середнього, t з банку).
Міряємо: (1) cos(e_{l+1}, T e_l) і частку перенесення; (2) cos(b_{l+1}, T b_l) проти контролю T r (r випадковий, та сама норма);
(3) замикання суми: cos(e_{l+1}, T e_l + b_{l+1}).
Передбачення до запуску: e_{l+1} на 60-90% перенесення (R^2), народження між шарами НЕ корелюють понад контроль
(|cos| ~ контроль, бо ранг переносу малий і контроль теж дає ненуль); шанс, що народження b_{l+1} передбачне з T b_l з cos>0.3 понад контроль: 15%.
СИСТЕМА (15.10, ідея юзера «всі шари разом»): P_l = перенос шар l -> вихід (добуток M_k = W[k]*Phi_k).
внесок c_l = |b_l P_l|^2 / |e_15|^2 (пломба: сума ~1, і |сума b_l P_l| ~ |e_15|);
ефективний ранг P_l (participation); вирівнювання = (|bP|^2/|b|^2) / (|P|_F^2/n) (1 = ізотропно);
частка b_l у топ-16 вхідних напрямках P_l проти 16/1024.
Передбачення: останні 5 шарів несуть >50% внеску; вирівнювання 0.8..1.3; ранг P_l >100 при l>=5; шанс вирівнювання >2: 15%."""
import os, sys, numpy as np, torch
from math import sqrt
sys.argv=['lean_oracle.py','0','regen','x']; sys.path.insert(0,'/home/trim/wb2lean'); os.chdir('/home/trim/wb2lean')
import lean_k3_aug_gpu as lk
exec(compile(open('/home/trim/wb2lean/lean_oracle.py').read().split('if __name__ == "__main__":')[0],'lo','exec'))
from scipy.special import ndtr
DUMPS=[int(x) for x in os.environ.get('DUMPS','0,1,2,3').split(',')]
def chain(w, gt, mode):
    rec=[]
    def hook(l,mu):
        rec.append((l,mu.double().cpu().numpy().copy()))
        return gt[l].to(mu.dtype).clone() if mode=='full' else mu
    statics=lk.build_static_coefs(w.shape[1]); lk.MU_HOOK=hook; lk.REGEN_STATE.clear(); lk.WK_HOOK=None; lk.D21_HOOK=None
    fit=FitC(); lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,g_hook=fit)
    rec.clear()
    pred=lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,aug_no_k211=True,g_hook=Replay(fit.coef),g_feed_hook=approx_pre,g_use_hook=approx_pre,shadow_B2=True,no_wk22_fac=True,aug_no_31=True,leg_trunc={"rres":16,"d21":16})
    lk.MU_HOOK=None
    mus=np.zeros((16,1024))
    for l,m in rec: mus[l]=m
    return mus, pred[-1].double().cpu().numpy()
cos=lambda a,b: float(a@b/np.sqrt((a@a)*(b@b)+1e-300))
SYS=[]
acc={k:[] for k in ['ne','nb','cT','R2fit','cB','cBctl','cSum']}
for i in DUMPS:
    d=np.load(DUMP.format(i)); w=d['weights']; W=w.astype(np.float64); gtf=d['final_means'].astype(np.float64); kp=np.load(BANK.format(i))
    gtn=kp['gt_mean'].astype(np.float64); gt=torch.tensor(gtn,device=lk.DEV)
    m=kp['pre_mean'].astype(float); t=m/np.sqrt(kp['pre_m2']-m**2); Ph=ndtr(t)
    mn,pn=chain(w,gt,'none'); mf,pf=chain(w,gt,'full')
    e=mn-gtn; b=mf-gtn            # b[l] = народження на шарі l при точному вхідному середньому
    print('дамп %d: MSE none %.4e (база %s) full %.4e'%(i,np.mean((pn-gtf)**2),'2.29e-8 сер.',np.mean((pf-gtf)**2)),flush=True)
    P=np.eye(1024); sysrow={}
    for l in range(15,-1,-1):
        bp=b[l]@P; nb2=b[l]@b[l]
        if l<15:
            U,S,Vt=np.linalg.svd(P); s2=S**2
            er=s2.sum()**2/(s2**2).sum(); al=(bp@bp/nb2)/(s2.sum()/1024); top=np.sum((b[l]@U[:,:16])**2)/nb2
        else: er,al,top=1024.,1.,16/1024
        sysrow[l]=(bp, er, al, top)
        P=(W[l]*Ph[l][None,:])@P
    E2=e[15]@e[15]; tot=sum(sysrow[l][0] for l in range(16))
    print('  пломба: сума c_l %.3f  |сума bP|^2/|e15|^2 %.3f  cos(сума bP, e15) %.3f'%(sum(sysrow[l][0]@sysrow[l][0] for l in range(16))/E2, tot@tot/E2, cos(tot,e[15])),flush=True)
    for l in range(16): SYS.append((i,l,sysrow[l][0]@sysrow[l][0]/E2,sysrow[l][1],sysrow[l][2],sysrow[l][3]))
    rng=np.random.default_rng(100+i)
    T=lambda l,v: Ph[l+1]*(v@W[l+1])
    row=[]
    for l in range(15):
        g=np.linalg.norm(gtn[l+1])
        Te=T(l,e[l]); Tb=T(l,b[l])
        c=Te@e[l+1]/(Te@Te)
        ctl=np.sqrt(np.mean([cos(b[l+1],T(l,rng.standard_normal(1024)*np.linalg.norm(b[l])/32))**2 for _ in range(20)]))
        vals=dict(ne=np.linalg.norm(e[l+1])/g, nb=np.linalg.norm(b[l+1])/g, cT=cos(e[l+1],Te),
                  R2fit=1-np.sum((e[l+1]-c*Te)**2)/np.sum(e[l+1]**2), cB=cos(b[l+1],Tb), cBctl=ctl, cSum=cos(e[l+1],Te+b[l+1]))
        for k,v in vals.items(): acc[k].append((i,l+1,v))
print('\nшар  |e|/|g|   |b|/|g|   cos(e,Te)  R2перен  cos(b,Tb)  контроль  cos(e,Te+b)')
for L in range(1,16):
    r={k:np.mean([v for (i,l,v) in acc[k] if l==L]) for k in acc}
    print('%3d  %.3e  %.3e  %+.3f     %+.3f    %+.3f     %.3f     %+.3f'%(L,r['ne'],r['nb'],r['cT'],r['R2fit'],r['cB'],r['cBctl'],r['cSum']))
cb=np.array([v for (_,_,v) in acc['cB']]); ct=np.array([v for (_,_,v) in acc['cBctl']])
print('\ncos(b,Tb) сер %+.3f  sd %.3f  |сер| %.3f   контроль rms %.3f   N=%d'%(cb.mean(),cb.std(),np.abs(cb).mean(),ct.mean(),len(cb)))
print('\nСИСТЕМА: шар  внесок c_l   еф.ранг P_l   вирівнювання   частка в топ-16 (ізотр. 0.016)')
for L in range(16):
    r=np.array([x[2:] for x in SYS if x[1]==L]).mean(0)
    print('  %3d  %.3f   %7.1f   %.3f   %.4f'%(L,*r))
al=np.array([x[4] for x in SYS if x[1]<15]); print('вирівнювання по всіх: сер %.3f мед %.3f sd %.3f'%(al.mean(),np.median(al),al.std()))
