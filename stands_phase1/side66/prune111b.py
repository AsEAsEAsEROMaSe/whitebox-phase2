# С111-г: ВУЗЬКА СМУГА. prune111 показав, що обрив між f=0.174 і 0.25, а моя
# точка A@0.174 була АГРЕСИВНІШОЮ за сабміт (ранг частоти, а не «frq==0»).
# Тут: (1) СПРАВЖНЯ база сабміту — бінарний критерій frq==0;
#      (2) сітка 0.18..0.24 для пілотного рангу і для детермінованого внеску;
#      (3) 🔒 усе в СЕРЕДНЬОМУ по 256 виходах, як у грейдера.
import numpy as np
from scipy.special import ndtr
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(4242); NET=6; NMC=32768; PILOT=256
INV=0.3989422804014327; UTIL=0.09011; BUD=2.72e11
LEVELS=[0.18,0.20,0.22,0.24]

def closure(Wn):
    m=np.zeros(256); C=np.eye(256); Ms=[]; Ph=[]
    for l in range(32):
        cp=Wn[l].T@C@Wn[l]; s=np.sqrt(np.maximum(np.diag(cp),1e-20))
        mp=m@Wn[l]; t=mp/s; P=ndtr(t); ph=np.exp(-0.5*t*t)*INV
        mn=mp*P+s*ph; e2=(mp*mp+s*s)*P+mp*s*ph
        C=cp*np.outer(P,P); C[np.diag_indices(256)]=np.maximum(e2-mn*mn,1e-20)
        m=mn; Ms.append(m.copy()); Ph.append(P.copy())
    return Ms,Ph
def contrib(Wn,Ms,Ph):
    B=np.eye(256); c=np.zeros((31,256))
    for l in range(30,-1,-1):
        B=(Wn[l+1]*Ph[l+1])@B; c[l]=Ms[l]*np.linalg.norm(B,axis=1)
    return c
def price(alive):
    tot=0.0; prev=256
    for l in range(32):
        out=int(alive[l].sum()) if l<31 else 256
        tot+=prev*out; prev=out
    return 2.0*tot
def fwd_mean(z,Wn,alive):
    h=np.maximum(z@Wn[0],0.0)
    if alive is not None: h=h*alive[0]
    for l in range(1,32):
        h=np.maximum(h@Wn[l],0.0)
        if alive is not None and l<31: h=h*alive[l]
    return h.mean(0),h

rows={}; fracs=[]
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    Ms,Ph=closure(Wn); c=contrib(Wn,Ms,Ph)
    zp=rng.standard_normal((PILOT,256)); xp=np.concatenate([zp,-zp])
    h=np.maximum(xp@Wn[0],0.0); frq=np.zeros((31,256)); frq[0]=(h>0).mean(0)
    for l in range(1,31):
        p=h@Wn[l]; frq[l]=(p>0).mean(0); h=np.maximum(p,0.0)
    z=rng.standard_normal((NMC,256))
    full,H=fwd_mean(z,Wn,None)
    a_ps=H.var(0,ddof=1).mean()          # 🔒 СЕРЕДНЄ, як у грейдера
    p_full=price([np.ones(256,bool)]*31)
    def run(name,alive):
        av=[alive[l].astype(np.float64) for l in range(31)]
        pm,_=fwd_mean(z,Wn,av)
        rows.setdefault(name,[]).append((((pm-full)**2).mean(),
                                         price([alive[l] for l in range(31)])/p_full,a_ps))
    # 🔒 СПРАВЖНЯ БАЗА САБМІТУ: бінарно, «жодного разу не відкрився»
    alive0=(frq>0.0); fracs.append(1.0-alive0.mean()); run('🔒БАЗА САБМІТУ frq==0',alive0)
    # детермінований аналог ТІЄЇ САМОЇ частки, але за внеском
    k0=int(round((1.0-alive0.mean())*31*256))
    for name,val in [('A пілотний ранг',frq),('C детермін. ВНЕСОК',c)]:
        order=np.argsort(val,axis=None)
        al=np.ones((31,256),bool); al.reshape(-1)[order[:k0]]=False
        run(f'{name} @ частка бази',al)
        for f in LEVELS:
            al=np.ones((31,256),bool); al.reshape(-1)[order[:int(f*31*256)]]=False
            run(f'{name} @ {f:.2f}',al)
    print(f'мережа {n}',flush=True)

print(f'\nчастка, яку ріже нинішній бінарний критерій: {np.mean(fracs)*100:.2f}%')
print(f"\n{'варіант':<30}{'зсув²':>12}{'ціна/повна':>12}{'MSE модель':>13}{'×бал':>9}")
base=None
for k in ['🔒БАЗА САБМІТУ frq==0','A пілотний ранг @ частка бази','C детермін. ВНЕСОК @ частка бази']+ \
         [f'{n} @ {f:.2f}' for n in ['A пілотний ранг','C детермін. ВНЕСОК'] for f in LEVELS]:
    if k not in rows: continue
    v=np.array(rows[k]); b2=v[:,0].mean(); pr=v[:,1].mean(); a=v[:,2].mean()
    mse=b2+a*pr*(2.0*32*256*256)/(UTIL*BUD)
    if base is None: base=mse
    print(f'{k:<30}{b2:>12.3e}{pr:>12.4f}{mse:>13.4e}{base/mse:>9.4f}')
