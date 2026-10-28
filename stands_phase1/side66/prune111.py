# С111-в: ОБРІЗКА ЗА ДЕТЕРМІНОВАНИМ ВНЕСКОМ. Б'ємо в `p` (ціну семпла), де ми
# програємо у 1.89×, а не в `a`, де ми вже попереду.
#
# Нинішній критерій — ПІЛОТНИЙ і БІНАРНИЙ: «живий, якщо хоч раз відкрився на
# 256 парах» ⟹ ріже 17.4%. Записано «поріг уже оптимальний», але це оптимум
# У МЕЖАХ бінарного критерію на семплах.
# Детермінований внесок нейрона j шару l у ВІДПОВІДЬ:
#     c[l,j] = m_l[j] · ‖B_l[j,:]‖ ,  B_l = (W_{l+1}Φ_{l+1})···(W_31Φ_31)
# Обидві величини (`m`, `Φ`) уже рахуються в КОЖНОМУ сабміті й викидаються.
#
# ЧОТИРИ КРИТЕРІЇ (усі ранжують, не бінарні ⟹ порівняння чесне):
#   A пілотна частота (емпірична Φ з 512 рядків) — узагальнення нинішнього
#   B детермінована Φ із замикання
#   C детермінований ВНЕСОК m·‖B‖            ← гіпотеза
#  🔒D випадковий ранг                        ← контроль
#
# МЕТРИКА: зсув² на СПІЛЬНИХ семплах (шум скорочується ⟹ чистий bias)
#          + ціна p(f) ⟹ модель MSE = bias² + a·p/(U·B)
# ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: C краще за A у зсуві при рівній частці (шанс 70%);
#   існує f з ×1.2..×1.5 у ціні при bias² < 20% від V (шанс 40%);
#   🔒D мусить бути НАЙГІРШИМ на кожному рівні, інакше прилад бреше.
import numpy as np
from scipy.special import ndtr
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(4242); NET=6; NMC=32768; PILOT=256
INV=0.3989422804014327; UTIL=0.09011; BUD=2.72e11
LEVELS=[0.174,0.25,0.35,0.45,0.55]

def closure(Wn):
    """m_l (E[h_l]) і Φ_l для всіх шарів; далі B_l назад."""
    m=np.zeros(256); C=np.eye(256); Ms=[]; Ph=[]
    for l in range(32):
        cp=Wn[l].T@C@Wn[l]; s=np.sqrt(np.maximum(np.diag(cp),1e-20))
        mp=m@Wn[l]; t=mp/s; P=ndtr(t); ph=np.exp(-0.5*t*t)*INV
        mn=mp*P+s*ph; e2=(mp*mp+s*s)*P+mp*s*ph
        C=cp*np.outer(P,P); C[np.diag_indices(256)]=np.maximum(e2-mn*mn,1e-20)
        m=mn; Ms.append(m.copy()); Ph.append(P.copy())
    return Ms,Ph

def contrib(Wn,Ms,Ph):
    """c[l,j] = m_l[j]·‖B_l[j,:]‖ для l=0..30 (виходи, які можна різати)."""
    B=np.eye(256); c=np.zeros((31,256))
    for l in range(30,-1,-1):
        B=(Wn[l+1]*Ph[l+1])@B
        c[l]=Ms[l]*np.linalg.norm(B,axis=1)
    return c

def price(alive):
    """ціна семпла обрізаної мережі = 2·Σ shape[0]·shape[1]"""
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

rows={}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    Ms,Ph=closure(Wn); c=contrib(Wn,Ms,Ph)
    # пілот: емпірична частота відкриття на 512 рядках
    zp=rng.standard_normal((PILOT,256)); xp=np.concatenate([zp,-zp])
    h=np.maximum(xp@Wn[0],0.0); frq=np.zeros((31,256))
    frq[0]=(h>0).mean(0)
    for l in range(1,31):
        p=h@Wn[l]; frq[l]=(p>0).mean(0); h=np.maximum(p,0.0)
    Phi_det=np.array([Ph[l] for l in range(31)])
    rnd=rng.random((31,256))
    crit={'A пілотна частота':frq,'B детермін. Φ':Phi_det,
          'C детермін. ВНЕСОК m·‖B‖':c,'🔒D випадковий':rnd}
    z=rng.standard_normal((NMC,256))
    full,H=fwd_mean(z,Wn,None)
    a_ps=H.var(0,ddof=1).sum()          # дисперсія на семпл
    p_full=price([np.ones(256,bool)]*31)
    for name,val in crit.items():
        order=np.argsort(val,axis=None)  # глобальний ранг
        for f in LEVELS:
            k=int(f*31*256)
            alive=np.ones((31,256),bool); flat=alive.reshape(-1)
            flat[order[:k]]=False
            # захист: лишити щонайменше 8 живих на шар
            for l in range(31):
                if alive[l].sum()<8:
                    idx=np.argsort(val[l])[::-1][:8]; alive[l][idx]=True
            av=[alive[l].astype(np.float64) for l in range(31)]
            pm,_=fwd_mean(z,Wn,av)
            bias2=((pm-full)**2).sum()
            pr=price([alive[l] for l in range(31)])
            rows.setdefault((name,f),[]).append((bias2,pr/p_full,a_ps))
    print(f'мережа {n}',flush=True)

print(f"\n{'критерій':<26}{'f':>6}{'зсув²':>12}{'ціна/повна':>12}{'MSE модель':>13}{'×бал':>8}")
# базова точка: нинішній сабміт ≈ пілотний бінарний, ріже 17.4%
base=None
for name in ['A пілотна частота','B детермін. Φ','C детермін. ВНЕСОК m·‖B‖','🔒D випадковий']:
    for f in LEVELS:
        v=np.array(rows[(name,f)])
        b2=v[:,0].mean(); pr=v[:,1].mean(); a=v[:,2].mean()
        mse=b2+a*pr*(2.0*32*256*256)/(UTIL*BUD)
        if base is None: base=mse
        print(f'{name:<26}{f:>6.3f}{b2:>12.3e}{pr:>12.4f}{mse:>13.4e}{base/mse:>8.4f}')
    print()
