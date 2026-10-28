"""ЕФЕКТИВНИЙ РАНГ КОВАРІАЦІЇ ПО ГЛИБИНІ — пряма підказка організаторів (town hall, 10:46):
«as the depth increases, the effective rank of the covariance matrix decreases — so it may be
possible to recover the deficit between the two classes of methods».

ЯКЩО ранг падає до ~8, то E[relu(h^L)] це фактично 8-ВИМІРНИЙ інтеграл, а не 256-вимірний.
Тоді детерміновані вузли (кубатура) перестають бути безсилими:
[[finding-cubature-deterministic-flat]] міряло їх у 256 вимірах — там вони й мусили програти.

МІРЯЄМО по кожному шару:
  • участь (participation ratio) = (Σλ)²/Σλ² — «скільки мод реально працює»
  • частку дисперсії в топ-1/4/8/16/32
  • те саме для ПАРНОЇ частини (саме її бачить v2 після антитетики)
"""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time(); NS=200000; CH=20000
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]

for net in (0,1):
    W=load(net); log('='*86); log('МЕРЕЖА %d'%net)
    for tag,anti in (('СИРА активація a^l',False),('ПАРНА частина ½(a(x)+a(−x))',True)):
        C=[torch.zeros(D,D,dtype=torch.float64,device=dev) for _ in range(L)]
        S=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
        n=0
        for c in range(NS//CH):
            g=torch.Generator(device=dev).manual_seed(88000+c)
            x=torch.randn(CH,D,device=dev,generator=g)
            with torch.no_grad():
                a=x; b=-x
                for l in range(L):
                    a=torch.relu(a@W[l]); b=torch.relu(b@W[l])
                    v=(0.5*(a+b) if anti else a).double()
                    S[l]+=v.sum(0); C[l]+=v.t()@v
            n+=CH
        log('  --- {} ---'.format(tag))
        log('  {:>4s} {:>9s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}'
            .format('шар','участь','топ-1','топ-4','топ-8','топ-16','топ-32'))
        for l in (0,1,2,4,8,12,16,20,24,28,30,31):
            m=S[l]/n; K=C[l]/n-torch.outer(m,m)
            ev=torch.linalg.eigvalsh(K).flip(0).clamp(min=0)
            t=float(ev.sum()); pr=float(ev.sum()**2/(ev*ev).sum())
            log('  {:4d} {:9.2f} {:7.1f}% {:7.1f}% {:7.1f}% {:7.1f}% {:7.1f}%'
                .format(l,pr,*[100*float(ev[:k].sum())/t for k in (1,4,8,16,32)]))
