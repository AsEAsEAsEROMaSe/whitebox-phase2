# С111-є: ЧИ Є МІСЦЕ ДЛЯ РІДКІСНОЇ ПАМ'ЯТІ? (заперечення ЮЗЕРА, і воно слушне)
# mem111 міряв вплив як ‖B‖·sd(h) по ВСІХ семплах ⟹ метрика КАРАЄ рідкісність
# за побудовою. Питання ЮЗЕРА було інше: що коли нейрон ТАКИ спрацьовує.
# РОЗДІЛЯЄМО:
#   канал     ‖B_l[j,:]‖              — геометрія ваг, від частоти НЕ залежить
#   амплітуда E[h_j | h_j>0]          — сила поштовху при спрацюванні
#   подієвий  ‖B‖·E[h_j|h_j>0]        — вплив ОДНОГО спрацювання
#   середній  ‖B‖·sd(h_j)             — те, що я міряв (для порівняння)
# 🔒 КОНТРОЛЬ: перемішані ρ ⟹ всі колонки мусять стати плоскими.
# ПЕРЕДБАЧЕННЯ: канал НЕ залежить від ρ (плоский) ⟹ місце для пам'яті Є;
#   амплітуда падає з рідкістю (хвіст біля нуля) ⟹ реалізації нема. (шанс 70%)
import numpy as np
from scipy.special import ndtr
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(777); NET=6; NMC=32768; INV=0.3989422804014327
BANDS=[(0.0,0.005),(0.005,0.02),(0.02,0.10),(0.10,0.30),(0.30,0.50),
       (0.50,0.70),(0.70,0.90),(0.90,0.98),(0.98,1.01)]
def closure(Wn):
    m=np.zeros(256); C=np.eye(256); Ph=[]
    for l in range(32):
        cp=Wn[l].T@C@Wn[l]; s=np.sqrt(np.maximum(np.diag(cp),1e-20))
        mp=m@Wn[l]; t=mp/s; P=ndtr(t); ph=np.exp(-0.5*t*t)*INV
        mn=mp*P+s*ph; e2=(mp*mp+s*s)*P+mp*s*ph
        C=cp*np.outer(P,P); C[np.diag_indices(256)]=np.maximum(e2-mn*mn,1e-20)
        m=mn; Ph.append(P.copy())
    return Ph
cols=['канал ‖B‖','амплітуда E[h|h>0]','ПОДІЄВИЙ ‖B‖·E[h|>0]','середній ‖B‖·sd']
acc={b:{c:[] for c in cols} for b in BANDS}; ctl={b:[] for b in BANDS}; cnt={b:0 for b in BANDS}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    Ph=closure(Wn); Bn=[None]*31; B=np.eye(256)
    for l in range(30,-1,-1):
        B=(Wn[l+1]*Ph[l+1])@B; Bn[l]=np.linalg.norm(B,axis=1)
    z=rng.standard_normal((NMC,256)); h=z.copy()
    R=[];K=[];A=[];E=[];S=[]
    for l in range(31):
        p=h@Wn[l]; hn=np.maximum(p,0.0)
        rho=(p>0).mean(0)
        cntpos=np.maximum((p>0).sum(0),1)
        amp=hn.sum(0)/cntpos                      # E[h | h>0]
        R.append(rho); K.append(Bn[l]); A.append(amp)
        E.append(Bn[l]*amp); S.append(Bn[l]*hn.std(0)); h=hn
    rho=np.concatenate(R)
    vals={'канал ‖B‖':np.concatenate(K),'амплітуда E[h|h>0]':np.concatenate(A),
          'ПОДІЄВИЙ ‖B‖·E[h|>0]':np.concatenate(E),'середній ‖B‖·sd':np.concatenate(S)}
    for c in cols: vals[c]=vals[c]/vals[c].mean()
    perm=rng.permutation(len(rho))
    for b in BANDS:
        m=(rho>=b[0])&(rho<b[1])
        if m.sum()>0:
            for c in cols: acc[b][c].append(vals[c][m].mean())
            cnt[b]+=int(m.sum())
        mc=(rho[perm]>=b[0])&(rho[perm]<b[1])
        if mc.sum()>0: ctl[b].append(vals['ПОДІЄВИЙ ‖B‖·E[h|>0]'][mc].mean())
    print(f'мережа {n}',flush=True)
print(f"\n{'смуга ρ':>15}{'к-сть':>8}{'канал ‖B‖':>12}{'амплітуда':>12}{'ПОДІЄВИЙ':>11}{'середній':>11}{'🔒контр':>9}")
for b in BANDS:
    if acc[b]['канал ‖B‖']:
        print(f"{'[%.3f,%.2f)'%b:>15}{cnt[b]//NET:>8}"
              f"{np.mean(acc[b]['канал ‖B‖']):>12.4f}{np.mean(acc[b]['амплітуда E[h|h>0]']):>12.4f}"
              f"{np.mean(acc[b]['ПОДІЄВИЙ ‖B‖·E[h|>0]']):>11.4f}{np.mean(acc[b]['середній ‖B‖·sd']):>11.4f}"
              f"{np.mean(ctl[b]):>9.4f}")
lo=BANDS[0]
print(f"\n🎯 НАЙРІДКІСНІШІ (ρ<0.005):  канал {np.mean(acc[lo]['канал ‖B‖']):.4f}"
      f" · амплітуда {np.mean(acc[lo]['амплітуда E[h|h>0]']):.4f}"
      f" · ПОДІЄВИЙ вплив {np.mean(acc[lo]['ПОДІЄВИЙ ‖B‖·E[h|>0]']):.4f}")
print("   канал ≈1 ⟹ АРХІТЕКТУРА дозволяє рідкісну пам'ять · канал ≪1 ⟹ виключає")
