"""ОКРЕМИЙ НЕЙРОН (звуження юзера С47): чи дешевше оцінити (m,s) маргіналі, ніж усереднити relu?

ЧОМУ ЗВУЖЕННЯ ЗМІСТОВНЕ: відповідь E[a³²_i] = E[relu(h³¹_i)], а ReLU ПОКООРДИНАТНИЙ ⟹
коваріація МІЖ нейронами у відповідь НЕ ВХОДИТЬ. Досить маргіналі кожного: (m_i, s_i).
512 чисел замість 65536 ⟹ блокер [[finding-synthetic-deep-start]] (K не купується) знято.

ЗСУВ уже виміряний (synth_deep.py, шар 31, великий пілот): 6.8e-7 / 1.1e-6 —
НИЖЧЕ за поточний MSE 2.5e-6. Тобто наближення досить добре.

ЛИШАЄТЬСЯ ОДНЕ ПИТАННЯ: чи дешевше оцінити (m,s), ніж прямо усереднити relu(h)?
  олівцем: Var через (m̂,ŝ) = Φ²s²/P + φ²s²/(2P) ≈ 0.33s²/P
           Var прямого     = Var(relu)/P        ≈ 0.34s²/P     ⟹ нічия
Але олівець сьогодні вже тричі помилявся ⟹ МІРЯЄМО на продакшн-конфігурації.

ТРИ ОЦІНЮВАЧІ, однакова кількість прогонів:
  A) v2 прямо:      середнє relu(h³¹) по вибілених антитетичних парах  ← наш робочий
  B) замикання:     оцінити (m,s) тими самими семплами → mΦ(z)+sφ(z)
  C) гібрид:        A + β·(B−A), β з тих самих семплів (чи додає B щось до A)
"""
import torch, time, math, numpy as np
LOG='/mnt/win_d/llm/White-Box/single_neuron.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32; SQ2PI=math.sqrt(2*math.pi)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def pre31(W,x):
    a=x
    for i in range(L-1): a=torch.relu(a@W[i])
    return a@W[L-1]
def zca(x):
    C=(x.t()@x)/x.shape[0]
    e,V=torch.linalg.eigh(C.double().cpu())
    return (V@torch.diag(e.clamp(min=1e-14).rsqrt())@V.t()).to(dev).float()
def Phi(z): return 0.5*(1+torch.erf(z/math.sqrt(2)))
def phi(z): return torch.exp(-0.5*z*z)/SQ2PI

for net in (0,1):
    W=load(net); log('='*80); log('МЕРЕЖА %d'%net)
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(30):
        g=torch.Generator(device=dev).manual_seed(990000+c)
        z=torch.randn(60000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        S+=torch.relu(pre31(W,xx)).double().sum(0); n+=xx.shape[0]
    TH=S/n
    # частка прибитих — для читання результату
    P0=torch.zeros(D,dtype=torch.float64,device=dev); m0=0
    for c in range(5):
        g=torch.Generator(device=dev).manual_seed(880000+c)
        z=torch.randn(40000,D,device=dev,generator=g)
        P0+=(pre31(W,z)>0).double().sum(0); m0+=z.shape[0]
    pin=float(((P0/m0<0.02)|(P0/m0>0.98)).double().mean())
    log('  прибитих нейронів на шарі 31: {:.0f}%'.format(100*pin))
    log('')
    log('  {:>8s} {:>13s} {:>13s} {:>13s} {:>11s}'
        .format('прогонів','A: v2 прямо','B: замикання','C: гібрид','B/A'))
    for NF in (6500,13000,26000):
        npair=NF//2; R=40
        eA=[];eB=[];eC=[]
        for r in range(R):
            g=torch.Generator(device=dev).manual_seed(41000+131*r)
            z=torch.randn(npair,D,device=dev,generator=g)
            X=torch.cat([z,-z],0); Xw=X@zca(X)
            h=pre31(W,Xw).double()
            a=torch.relu(h)
            estA=((a[:npair]+a[npair:])*0.5).mean(0)
            mm=((h[:npair]+h[npair:])*0.5).mean(0)
            h2=h*h
            ss=(((h2[:npair]+h2[npair:])*0.5).mean(0)-mm*mm).clamp(min=1e-20).sqrt()
            zz=mm/ss
            estB=mm*Phi(zz)+ss*phi(zz)
            # β для гібриду з тих самих семплів (по всіх координатах, скаляр)
            d=(estB-estA)
            beta=float(-(d*(estA-TH)).sum()/ (d*d).sum().clamp(min=1e-30))  # ОРАКУЛ-β, верхня межа
            estC=estA+beta*d
            eA.append(float(((estA-TH)**2).mean())); eB.append(float(((estB-TH)**2).mean()))
            eC.append(float(((estC-TH)**2).mean()))
        mA,mB,mC=np.mean(eA),np.mean(eB),np.mean(eC)
        log('  {:8d} {:13.4e} {:13.4e} {:13.4e} {:10.2f}×'.format(NF,mA,mB,mC,mB/mA))
    log('  ⚠️ C рахований з ОРАКУЛ-β (знає істину) = ВЕРХНЯ МЕЖА гібриду, не метод')
log('done.')
