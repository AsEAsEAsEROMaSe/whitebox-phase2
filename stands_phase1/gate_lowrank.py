"""НИЗЬКОРАНГОВИЙ АРКСИНУСНИЙ CV — чи придатний за ціною.
Ознака: p_k(x) = (s·v_k)²,  s=sign(x·W₀).  Середнє E[p_k] = v_kᵀ A v_k, де
A_ij = (2/π)·arcsin(ρ_ij), ρ = кореляція h⁰ = нормована W₀ᵀW₀ — ТОЧНО, БЕЗКОШТОВНО.
Ціна на семпл: s уже є у форварді; (S@V) це 256×k множення = мізер проти 4.2e6 флоп форварду.
МІРЯЄМО R² out-of-sample НА ВСЬОМУ ВЕКТОРІ (зважено дисперсією) — саме він іде в score."""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time(); NS=300000; CH=20000
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a
for net in (0,1):
    W=load(net); log('='*74); log('МЕРЕЖА %d'%net)
    Sl=[];Rl=[]
    for c in range(NS//CH):
        g2=torch.Generator(device=dev).manual_seed(60000+c)
        x=torch.randn(CH,D,device=dev,generator=g2)
        with torch.no_grad():
            Rl.append(0.5*(fwd(W,x)+fwd(W,-x)))
            Sl.append(torch.sign(x@W[0]).to(torch.int8))
    S=torch.cat(Sl).float(); R=torch.cat(Rl); n=S.shape[0]; h=n//2
    R=R-R.mean(0,keepdim=True)
    vtot=float((R*R).mean(0).sum())
    # арксинусна матриця A — БЕЗКОШТОВНА, лише з W₀
    K0=(W[0].t()@W[0]).double(); d=K0.diagonal().sqrt()
    A=(2/math.pi)*torch.arcsin((K0/torch.outer(d,d)).clamp(-1,1))
    # напрямки: топ-власні вектори Cov(вихід, s_i s_j), агреговані по всіх координатах
    #   T_ij = Σ_c Cov(R_c, s_i s_j)²  ⟹ беремо через випадкові проби, щоб не будувати 256³
    gg=torch.Generator(device=dev).manual_seed(3)
    M=torch.zeros(D,D,dtype=torch.float64,device=dev)
    for t in range(24):
        u=torch.randn(D,device=dev,generator=gg); u=u/u.norm()
        y=(R@u)
        Cm=((S*y.unsqueeze(1)).t()@S)/n
        Cm=0.5*(Cm+Cm.t()).double(); M+=Cm@Cm
    ev,V=torch.linalg.eigh(M); V=V.flip(1)
    log('  Var(парної частини, весь вектор)={:.4e}'.format(vtot))
    log('  {:>5s} {:>12s} {:>12s} {:>14s} {:>12s}'.format('ранг','R² вектор','R² α','виграш','score'))
    for k in (1,2,4,8,16,32,64,128,256):
        Vk=V[:,:k].float()
        P=(S@Vk)**2
        mu=torch.stack([ (Vk[:,j].double()@A@Vk[:,j].double()) for j in range(k)])   # ТОЧНЕ середнє
        P=P.double()-mu.unsqueeze(0)                                                  # центровано БЕЗКОШТОВНО
        P1,P2=P[:h],P[h:]
        G=(P1.t()@P1)/h; r=(P1.t()@R[:h].double())/h
        B=torch.linalg.solve(G+1e-6*float(G.diagonal().mean())*torch.eye(k,dtype=torch.float64,device=dev),r)
        res=R[h:].double()-P2@B
        r2v=1-float((res*res).mean(0).sum())/vtot
        log('  {:5d} {:12.4f} {:12s} {:13.4f}× {:11.3e}'
            .format(k,r2v,'-',1.0/max(1e-9,1.0-max(r2v,0)),3.09e-7*(1.0-max(r2v,0))))
        del P,P1,P2,G,r,B,res
