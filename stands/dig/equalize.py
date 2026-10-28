"""«Урівняти всіх» (юзер 15.10): точна симетрія ReLU — W[l][:,j]·c_j, W[l+1][j,:]/c_j не змінює функцію. c_j = 1/σ_j (банк) ⟹ σ = 1 у всіх
нейронах шарів 0..14 (останній шар не чіпаємо, бо вихід мусить лишитися тим самим). Ланцюг на урівняній мережі проти бази.
Передбачення до запуску: ланцюг еквіваріантний (t = m/σ інваріантне) ⟹ MSE та сама до шуму округлення (~0.5%); шанс > x1.02: 3%."""
import os, sys, numpy as np, torch
sys.argv=['lean_oracle.py','0','regen','x']; sys.path.insert(0,'/home/trim/wb2lean'); os.chdir('/home/trim/wb2lean')
import lean_k3_aug_gpu as lk
exec(compile(open('/home/trim/wb2lean/lean_oracle.py').read().split('if __name__ == "__main__":')[0],'lo','exec'))
def chain(w):
    statics=lk.build_static_coefs(w.shape[1]); lk.REGEN_STATE.clear(); lk.WK_HOOK=None; lk.D21_HOOK=None; lk.MU_HOOK=None
    fit=FitC(); lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,g_hook=fit)
    return lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,aug_no_k211=True,g_hook=Replay(fit.coef),g_feed_hook=approx_pre,g_use_hook=approx_pre,shadow_B2=True,no_wk22_fac=True,aug_no_31=True,leg_trunc={"rres":16,"d21":16})[-1].double().cpu().numpy()
ALPHA=float(os.environ.get('ALPHA','-1')); _g=os.environ.get('GAUGE'); GAUGE=None if not _g else (float(_g.split(',')[0]),np.deg2rad(float(_g.split(',')[1])),int(_g.split(',')[2])); DD=[int(x) for x in os.environ.get('DUMPS','0,1,2,3,4,5,6,7').split(',')]
BASE={0:2.1908e-08,1:2.1339e-08,2:2.4535e-08,3:2.3754e-08,4:2.1946e-08,5:1.8771e-08,6:1.9787e-08,7:2.3914e-08}
r=[]
for i in DD:
    d=np.load(DUMP.format(i)); w=d['weights'].astype(np.float64); gt=d['final_means'].astype(np.float64); kp=np.load(BANK.format(i))
    s=np.sqrt(kp['pre_m2']-kp['pre_mean']**2)        # σ передактивацій (16,1024)
    w2=w.copy()
    for l in range(15):
        if os.environ.get('GAB'):
            _a,_b=[float(x) for x in os.environ['GAB'].split(',')]; c=np.exp(_a*np.log(s[l])+_b*kp['pre_mean'][l]/s[l])
        elif GAUGE is None: c=s[l]**ALPHA
        else:
            _t=kp['pre_mean'][l]/s[l]; _lc=np.full(1024,GAUGE[0]) if abs(GAUGE[1]-np.deg2rad(999))<1e-9 else GAUGE[0]*(np.cos(GAUGE[1])*np.log(s[l])+np.sin(GAUGE[1])*_t); c=np.exp(_lc)
            if GAUGE[2]: c=c/c.mean()
        w2[l]=w2[l]*c[None,:]; w2[l+1]=w2[l+1]/c[:,None]
    a=BASE[i]; pp=chain(w2.astype(np.float32));  b=np.mean((pp-gt)**2); r.append((a,b))
    print('дамп %d: база %.4e  урівняна %.4e  x%.4f'%(i,a,b,a/b),flush=True)
r=np.array(r); print('СЕРЕДНЄ: база %.4e урівняна %.4e x%.4f'%(r[:,0].mean(),r[:,1].mean(),r[:,0].mean()/r[:,1].mean()))
