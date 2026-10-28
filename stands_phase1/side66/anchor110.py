# С110: чи є усереднення образів ваг (точка опори ЮЗЕРА) корисним ЯКОРЕМ поруч із МК.
# ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: виграш ×1.00..1.03 у V без ціни; з ціною 3.8% підлоги -> НУЛЬ.
# Шанс, що LOO-виграш понад ×1.05: ~8%.
import numpy as np, time
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(3141); NET=8; NTRUE=131072; NMC=6144
def fwd(h,Wn,l0=0):
    for l in range(l0,32): h=np.maximum(h@Wn[l],0.0)
    return h
def cerr(a,T):
    c=float(a@T/(np.linalg.norm(a)*np.linalg.norm(T))); return np.sqrt(max(0.0,1-c*c))
T=[];A=[];Bp=[];Bw=[]
for n in range(NET):
    t0=time.time(); Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    acc=np.zeros(256)
    for i in range(0,NTRUE,8192): acc+=fwd(rng.standard_normal((8192,256)),Wn).sum(0)
    T.append(acc/NTRUE)
    D=Wn[0].T.copy(); D/=np.linalg.norm(D,axis=1,keepdims=True); D*=16.0
    A.append(fwd(np.vstack([D,-D]),Wn).mean(0))                    # опора: ±симетризовані ваги
    z=rng.standard_normal((NMC,256)); Bp.append(fwd(z.copy(),Wn).mean(0))
    zc=z-z.mean(0); cov=zc.T@zc/len(zc)
    Lw=np.linalg.inv(np.linalg.cholesky(cov)).T                    # ZCA-вибілювання
    Bw.append(fwd(zc@Lw,Wn).mean(0))
    print(f'мережа {n} {time.time()-t0:.0f}s',flush=True)
T=np.array(T);A=np.array(A);Bp=np.array(Bp);Bw=np.array(Bw)
print('\n=== ПОХИБКИ (косинусні, %) ===')
for nm,X in [('опора з ваг',A),('МК 6144 гола',Bp),('МК 6144 вибілена',Bw)]:
    print(f'  {nm:<22}{np.mean([cerr(X[i],T[i]) for i in range(NET)])*100:.4f}%')
print('\n=== ЗМІШУВАННЯ опори з вибіленою МК (LOO по мережах) ===')
def mix(w,B):  # спільний масштаб опори підбирається, потім вага w
    return np.mean([cerr(B[i]+w*sc[i]*A[i],T[i]) for i in range(NET)])
sc=np.array([float(T[i]@A[i]/(A[i]@A[i])) for i in range(NET)])
base=np.mean([cerr(Bw[i],T[i]) for i in range(NET)])
ws=np.linspace(0,1,51); errs=[mix(w,Bw) for w in ws]
best=int(np.argmin(errs)); print(f'  in-sample: w*={ws[best]:.2f}  {base*100:.4f}% -> {errs[best]*100:.4f}%  x{base/errs[best]:.4f}')
loo=[]
for i in range(NET):
    o=[j for j in range(NET) if j!=i]
    e=[np.mean([cerr(Bw[j]+w*sc[j]*A[j],T[j]) for j in o]) for w in ws]
    wi=ws[int(np.argmin(e))]
    loo.append(cerr(Bw[i]+wi*np.median(sc[o])*A[i],T[i]))
print(f'  LOO      : {base*100:.4f}% -> {np.mean(loo)*100:.4f}%  x{base/np.mean(loo):.4f}')
print(f'\n  масштаб опори sc: медіана {np.median(sc):.4f} розкид {np.std(sc)/abs(np.mean(sc))*100:.1f}%')
print('  ЦІНА опори: 31 матмул 256^3 = 1.04e9 флопів = 3.8% підлоги ⟹ V росте 1.9%')
