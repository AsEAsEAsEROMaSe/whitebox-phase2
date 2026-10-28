"""Спектр пропагатора P_{l<-b} = T_{l-1}...T_b, T_k = W_{k+1} diag(w1_k) (перенос ніг у ланцюгу) за віком.
Питання: який ранг (99% / 99.9% енергії, PR) мають ноги старих джерел; чи рангове правило 504aldo (3n/8 @4, n/8 @8) = спектр P."""
import sys, numpy as np, torch
sys.argv=[sys.argv[0],"0"]
import lean_oracle as lo, lean_k3_aug_gpu as lk
REC=[]; _lin,_wick=lk.K3Blocks.linear_,lk.K3Blocks.wick_
def lin(self,W): REC.append(("W",W.detach().clone())); _lin(self,W)
def wick(self,w1): REC.append(("w1",w1.detach().clone())); _wick(self,w1)
lk.K3Blocks.linear_,lk.K3Blocks.wick_=lin,wick
for dump in (0,1):
    d=np.load(lo.DUMP.format(dump)); w=d["weights"]; st=lk.build_static_coefs(w.shape[1])
    lk.REGEN_STATE.clear(); fit=lo.FitC(); lk.lean_k3_predict(w,dtype=torch.float32,statics=st,k4_aug=True,g_hook=fit)
    REC.clear()
    lk.lean_k3_predict(w,dtype=torch.float32,statics=st,k4_aug=True,aug_no_k211=True,g_hook=lo.Replay(fit.coef),g_feed_hook=lo.approx_pre,g_use_hook=lo.approx_pre,shadow_B2=True,no_wk22_fac=True,aug_no_31=True,leg_trunc={"rres":16,"d21":16})
    Ws=[r[1].double() for r in REC if r[0]=="W"]; w1s=[r[1].double() for r in REC if r[0]=="w1"]
    n=w.shape[1]
    # T from probe(l) to probe(l+1) = Ws[l]*w1s[l-1]; source born at b (probe b has P=I) -> at probe l: P = T_{l-1} ... T_b
    for l in (8, 15):
        line=[]
        P=torch.eye(n,dtype=torch.float64,device=Ws[0].device)
        for b in range(l-1, 0, -1):
            P = P @ (Ws[b]*w1s[b-1][None,:])   # P_{l<-b} = T_{l-1} ... T_b: append T_b on the right as b decreases
            age=l-b
            s=torch.linalg.svdvals(P); e=s**2; c=torch.cumsum(e,0)/e.sum()
            r99=int((c<0.99).sum())+1; r999=int((c<0.999).sum())+1; pr=float(e.sum()**2/(e*e).sum()); pr1=float(s.sum()**2/(s*s).sum())
            line.append("age %2d: r99 %4d r99.9 %4d PR(s²) %6.1f PR(s) %6.1f s1/s_n %.1e" % (age, r99, r999, pr, pr1, float(s[0]/s[-1])))
        print("dump%d l=%d" % (dump,l)); print("\n".join("   "+x for x in line), flush=True)
