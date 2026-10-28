"""Діагностика: частка норми рядків мертвих нейронів (t<-3) у ногах κ3, P, G на момент linear_."""
import os, sys, numpy as np, torch
sys.argv=['lean_oracle.py','0','regen','x']
sys.path.insert(0,'/home/trim/wb2lean'); os.chdir('/home/trim/wb2lean')
import lean_k3_aug_gpu as lk
src=open('/home/trim/wb2lean/lean_oracle.py').read().split('if __name__ == "__main__":')[0]
exec(compile(src,'lean_oracle_head','exec'))
TH=3.0; state={'dead':None}; rec=[]
orig=lk.K3Blocks.linear_
def lin(self,W):
    d=state['dead']
    if d is not None:
        fr={}
        for key in self.KEYS+("P",):
            num=den=0.0
            for s in self.sources:
                M=s.get(key)
                if M is None or (key=="P" and M is None): continue
                num+=float((M[d]**2).sum()); den+=float((M**2).sum())
            if den>0: fr[key]=num/den
        rec.append((state['l'],float(d.float().mean()),fr))
    return orig(self,W)
lk.K3Blocks.linear_=lin
# мертві: з mu/var кожного шару через K3_PROBE (викликається після linear, з D3,D21) — беремо з D21_HOOK live
class Probe:
    def __init__(s): s.l=0
    def __call__(s,l,D3,D21,live):
        t=live['mu']/live['var'].sqrt(); state['dead']=t<-TH; state['l']=l+1
        return D3,D21,None
d=np.load(DUMP.format(0)); w=d['weights']; statics=lk.build_static_coefs(w.shape[1])
fit=FitC(); lk.D21_HOOK=None
lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,g_hook=fit)
rec.clear(); state['dead']=None; lk.D21_HOOK=Probe()
pred=lk.lean_k3_predict(w,dtype=dtype,statics=statics,k4_aug=True,aug_no_k211=True,g_hook=Replay(fit.coef),g_feed_hook=approx_pre,g_use_hook=approx_pre,shadow_B2=True,no_wk22_fac=True,aug_no_31=True,leg_trunc={"rres":16,"d21":16})
print('mse',float(np.mean((pred[-1].double().cpu().numpy()-d['final_means'])**2)))
for l,f,fr in rec: print('l=%2d dead %.3f '%(l,f)+' '.join('%s %.4f'%(k,v) for k,v in fr.items()))
