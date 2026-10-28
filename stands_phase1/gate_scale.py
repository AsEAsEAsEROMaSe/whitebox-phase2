"""Продовження gatepairs.py: (1) як R² росте з числом пар; (2) чи низькоранговий оптимум.
Якщо B у sᵀBs низькоранговий ⟹ ознака (s·v)², середнє vᵀAv з арксинуса = БЕЗКОШТОВНО,
ціна на семпл мізерна. Це різниця між «цікаво» і «придатно»."""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time(); NS=300000; CH=20000
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a
def oosR2(F1,y1,F2,y2,lam=1e-4):
    F1=F1-F1.mean(0,keepdim=True); y1=y1-y1.mean()
    G=(F1.t()@F1)/F1.shape[0]; r=(F1.t()@y1)/F1.shape[0]
    b=torch.linalg.solve(G+lam*float(G.diagonal().mean())*torch.eye(G.shape[0],dtype=G.dtype,device=dev),r)
    F2=F2-F2.mean(0,keepdim=True); y2=y2-y2.mean()
    return float(1-((y2-F2@b)**2).mean()/(y2*y2).mean())

for net in (0,1):
    W=load(net); log('='*70); log('МЕРЕЖА %d'%net)
    g=torch.Generator(device=dev).manual_seed(4242)
    X=torch.randn(8,D,device=dev,generator=g); X=X/X.norm(dim=1,keepdim=True)
    for _ in range(3):
        Y=fwd(W,X); X=Y/Y.norm(dim=1,keepdim=True).clamp(min=1e-30)
    vs=(X[0]/X[0].norm())
    # один прохід: тримаємо ЗНАКИ (fp8-подібно як int8) + α
    Sl=[];Al=[]
    for c in range(NS//CH):
        g2=torch.Generator(device=dev).manual_seed(60000+c)
        x=torch.randn(CH,D,device=dev,generator=g2)
        with torch.no_grad():
            r=0.5*(fwd(W,x)+fwd(W,-x))
            Sl.append(torch.sign(x@W[0]).to(torch.int8)); Al.append(r@vs)
    S=torch.cat(Sl); al=torch.cat(Al).double(); n=S.shape[0]; h=n//2
    al=al-al.mean(); vtot=float((al*al).mean())
    log('  Var(α_even)={:.4e}, семплів={:d}'.format(vtot,n))
    # (1) масштабування по числу пар
    log('  --- R² out-of-sample vs число пар (з 32640) ---')
    for NP in (500,1000,2000,4000,8000,16000):
        gg=torch.Generator(device=dev).manual_seed(11)
        I=torch.randint(0,D,(NP,),device=dev,generator=gg); J=torch.randint(0,D,(NP,),device=dev,generator=gg)
        m=I!=J; I,J=I[m],J[m]
        F=(S[:,I]*S[:,J]).float()
        r2=oosR2(F[:h],al[:h].float(),F[h:],al[h:].float())
        log('    пар {:6d} ({:4.1f}% сім\'ї)   R² = {:7.4f}'.format(I.shape[0],100*I.shape[0]/32640,r2))
        del F
    # (2) низький ранг: C_ij = Cov(α, s_i s_j) ⟹ топ-власні вектори ⟹ ознаки (s·v)²
    Sf=S.float()
    C=(Sf*al.float().unsqueeze(1)).t()@Sf/n - (Sf.mean(0).unsqueeze(1)*0)  # E[α s_i s_j], α центроване
    C=0.5*(C+C.t())
    ev,V=torch.linalg.eigh(C.double())
    o=torch.argsort(ev.abs(),descending=True); ev=ev[o]; V=V[:,o]
    log('  --- НИЗЬКИЙ РАНГ: ознаки (s·v_k)², середнє vᵀAv з арксинуса = БЕЗКОШТОВНЕ ---')
    log('    спектр |λ| топ-6: {}'.format(' '.join('{:.3e}'.format(float(e)) for e in ev[:6].abs())))
    for k in (1,2,4,8,16,32,64):
        P=(Sf@V[:,:k].float())**2
        r2=oosR2(P[:h],al[:h].float(),P[h:],al[h:].float())
        log('    ранг {:3d}   R² = {:7.4f}    ⟹ повний виграш {:.4f}× (α несе ~60% шуму)'
            .format(k,r2,1.0/(1.0-0.6*max(r2,0))))
