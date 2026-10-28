"""fp64 на 1080Ti + ПОЗАДІАГОНАЛЬ у подвійній точності.

Теорія: Cov(a_i,g_j) = -Cov(a_j,g_i) -> матриця АНТИСИМЕТРИЧНА.
Діагональ занулена теоремою (перевірено). Позадіагональ теорія НЕ закриває.
У float32 вона показала 0.004 (=шум), але float32 щойно збрехав на діагоналі -> міряємо у fp64.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/offdiag.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32

# ---------- 0. чесний бенчмарк fp64 vs fp32 на цій карті ----------
def bench(dt,n=4096,rep=5):
    a=torch.randn(n,n,device=dev,dtype=dt); b=torch.randn(n,n,device=dev,dtype=dt)
    torch.cuda.synchronize(); ts=[]
    for _ in range(rep):
        t0=time.perf_counter(); c=a@b; torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    return 2*n**3/min(ts)/1e12
log('=== пропускна здатність {} ==='.format(torch.cuda.get_device_name(4)))
f32=bench(torch.float32); f64=bench(torch.float64)
log('  fp32 {:7.2f} TFLOP/s'.format(f32))
log('  fp64 {:7.2f} TFLOP/s   ->  fp64 повільніша у {:.1f}x'.format(f64,f32/f64))

dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def fwd_jvp(W,x,v):
    a=x; t=v
    for li in range(L):
        h=a@W[li]; th=t@W[li]; g=(h>0).to(h.dtype)
        a=h*g; t=th*g
    return a,t

for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float64),device=dev,dtype=torch.float64) for i in range(L)]
    gs=torch.Generator(device=dev).manual_seed(600+net)
    M=torch.randn(D,D,device=dev,generator=gs,dtype=torch.float64)
    A=(M-M.t())/math.sqrt(2.0*D)
    U=W[0]/W[0].norm(dim=0,keepdim=True)
    A2=torch.outer(U[:,0],U[:,1])-torch.outer(U[:,1],U[:,0])       # у напрямках мережі
    log(''); log('######## NET {} (fp64) ########'.format(net))
    for tag,AA in (('випадкова',A),('мережа',A2)):
        BL=24; CH=20_000
        blocks=[]
        for b in range(BL):
            z=torch.randn(CH//2,D,device=dev,dtype=torch.float64,
                          generator=torch.Generator(device=dev).manual_seed(4000+net*100+b))
            x=torch.cat([z,-z],0)
            a,g=fwd_jvp(W,x,x@AA.t())
            blocks.append(((a.t()@g)/x.shape[0]).cpu())            # (D,D) на блок
            if b==0: sa=a.std(0).cpu(); sg=g.std(0).cpu()
        V=torch.stack(blocks)                                       # (BL,D,D)
        m=V.mean(0); se=V.std(0)/math.sqrt(BL)
        R=m/torch.outer(sa,sg).clamp(min=1e-300)                    # у одиницях кореляції
        Z=(m/se.clamp(min=1e-300)).abs()
        dz=torch.diagonal(Z); od=Z-torch.diag(torch.diagonal(Z))
        anti=float((m+m.t()).abs().mean()/ (m.abs().mean()+1e-300)) # перевірка антисиметрії
        log(' {:10s}: діаг |z| сер {:.2f} макс {:.2f} | ПОЗАдіаг |z| сер {:.2f} макс {:.2f}'.format(
            tag,float(dz.mean()),float(dz.max()),
            float(od.abs().mean()),float(od.abs().max())))
        log('             позадіаг |r| сер {:.5f} макс {:.5f} | пар >3σ: {}/{}'.format(
            float((R-torch.diag(torch.diagonal(R))).abs().mean()),
            float((R-torch.diag(torch.diagonal(R))).abs().max()),
            int((od>3).sum()),D*(D-1)))
        log('             |m+m^T|/|m| = {:.4f}  (0 = ідеальна антисиметрія, як каже теорія)'.format(anti))
log('done.')
