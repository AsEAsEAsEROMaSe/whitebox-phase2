"""ПЕРЕВІРКА суперечності: теорема каже E[a_j g_j]=0, вимір каже 0.034σ.

1. чи g справді = J·(Ax)?  -> скінченні різниці: a(e^{eA}x) - a(x) ≈ e·g
2. чи E[a_j g_j] справді нуль? -> оцінка з ЧЕСНОЮ похибкою (по блоках)
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/antisym_check.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(dat['w_0_%d'%i].astype(np.float32),device=dev,dtype=torch.float64) for i in range(L)]

def fwd(x):
    a=x
    for li in range(L): a=torch.clamp(a@W[li],min=0.0)
    return a
def fwd_jvp(x,v):
    a=x; t=v
    for li in range(L):
        h=a@W[li]; th=t@W[li]; g=(h>0).to(h.dtype)
        a=h*g; t=th*g
    return a,t

gseed=torch.Generator(device=dev).manual_seed(5)
M=torch.randn(D,D,device=dev,generator=gseed,dtype=torch.float64)
A=(M-M.t())/math.sqrt(2.0*D)

# --- 1. скінченні різниці ---
x=torch.randn(64,D,device=dev,generator=gseed,dtype=torch.float64)
a0,g=fwd_jvp(x,x@A.t())
log('1. ПЕРЕВІРКА JVP скінченними різницями:')
for eps in (1e-4,1e-5,1e-6):
    R=torch.matrix_exp(eps*A)
    num=(fwd(x@R.t())-a0)/eps
    rel=float((num-g).norm()/g.norm())
    log('   eps={:.0e}: ||числ - аналіт||/||аналіт|| = {:.3e}'.format(eps,rel))

# --- 2. E[a_j g_j] з ЧЕСНОЮ похибкою по блоках ---
log('')
log('2. E[a_j g_j] — оцінка й похибка (30 незалежних блоків по 20k):')
BL=30; CH=20_000
vals=[]
for b in range(BL):
    z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(9000+b),dtype=torch.float64)
    x=torch.cat([z,-z],0)
    a,g=fwd_jvp(x,x@A.t())
    vals.append(((a*g).mean(0)).cpu())          # (D,) на блок
V=torch.stack(vals)                              # (BL,D)
m=V.mean(0); se=V.std(0)/math.sqrt(BL)
z_=(m/se.clamp(min=1e-300)).abs()
log('   |E[a g]| сер по компонентах : {:.4e}'.format(float(m.abs().mean())))
log('   станд.похибка сер           : {:.4e}'.format(float(se.mean())))
log('   |z| сер {:.2f}  макс {:.2f}   (>3 = значуще відхилення від нуля)'.format(
    float(z_.mean()),float(z_.max())))
log('   компонент із |z|>3: {}/{}'.format(int((z_>3).sum()),D))
# для масштабу: те саме у одиницях кореляції
sa=torch.stack([ (fwd(torch.cat([torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(77),dtype=torch.float64)]*1+[-torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(77),dtype=torch.float64)],0)).std(0)).cpu()])[0]
log('')
log('ВИСНОВОК: якщо |z| ~ 1 -> теорема вірна, вимір 0.034 був артефактом.')
log('          якщо |z| >> 3 -> теорема хибна, є реальний сигнал.')
log('done.')
