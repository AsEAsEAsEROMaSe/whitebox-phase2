import sys, numpy as np, torch
sys.argv=[sys.argv[0],"0"]
import lean_oracle as lo, lean_k3_aug_gpu as lk
REC=[]; _lin,_wick=lk.K3Blocks.linear_,lk.K3Blocks.wick_
def lin(self,W): REC.append(("W",W.detach().clone())); _lin(self,W)
def wick(self,w1): REC.append(("w1",w1.detach().clone())); _wick(self,w1)
lk.K3Blocks.linear_,lk.K3Blocks.wick_=lin,wick
PS={}
def probe(l,K3,D3,D21):
    s=K3.sources[0]; PS[l]=(s["P"].detach().clone() if s["P"] is not None else None, s["X1"].detach().clone())
d=np.load(lo.DUMP.format(0)); w=d["weights"]; st=lk.build_static_coefs(w.shape[1])
lk.REGEN_STATE.clear(); fit=lo.FitC(); lk.lean_k3_predict(w,dtype=torch.float32,statics=st,k4_aug=True,g_hook=fit)
REC.clear(); lk.K3_PROBE=probe
lk.lean_k3_predict(w,dtype=torch.float32,statics=st,k4_aug=True,aug_no_k211=True,g_hook=lo.Replay(fit.coef),g_feed_hook=lo.approx_pre,g_use_hook=lo.approx_pre,shadow_B2=True,no_wk22_fac=True,aug_no_31=True,leg_trunc={"rres":16,"d21":16})
print("sequence:", "".join("W" if r[0]=="W" else "w" for r in REC))
Ws=[r[1] for r in REC if r[0]=="W"]; w1s=[r[1] for r in REC if r[0]=="w1"]
for l in (3,8):
    X=PS[l][1]; Xn=PS[l+1][1]
    for name,(iw,iv) in {"Ws[l]*w1s[l]":(l,l),"Ws[l]*w1s[l-1]":(l,l-1),"Ws[l+1]*w1s[l]":(l+1,l),"Ws[l-1]*w1s[l-1]":(l-1,l-1)}.items():
        if iw>=len(Ws) or iv<0: continue
        for orient in ("W@(w1*X)","(W*w1[None])@X","W.T@(w1*X)"):
            W=Ws[iw]; v=w1s[iv]
            P = W@(v[:,None]*X) if orient=="W@(w1*X)" else ((W*v[None,:])@X if orient=="(W*w1[None])@X" else W.T@(v[:,None]*X))
            print(l,name,orient,"rel err %.2e"%float((P-Xn).norm()/Xn.norm()))
