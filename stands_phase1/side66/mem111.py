# С111-д: ЧИ МОВЧАЗНІ НЕЙРОНИ Є ПАМ'ЯТТЮ? (питання ЮЗЕРА про бінарну пам'ять)
# Заморожена маска ≠ пам'ять. Різниця ВИМІРНА:
#   пам'ять   = рідкісна подія, яка СИЛЬНО міняє вихід  ⟹ великий вплив
#   мертвий   = рідкісна подія, яка нічого не міняє     ⟹ малий вплив
# Вплив нейрона j шару l на ВІДПОВІДЬ: infl = ‖B_l[j,:]‖ · sd(h_j),
#   B_l = (W_{l+1}Φ_{l+1})···(W₃₁Φ₃₁) — спряжена карта з ЧИСТОГО замикання.
# Ділимо нейрони на смуги за ρ = P(p>0) і дивимось СЕРЕДНІЙ вплив у смузі.
# 🔒 КОНТРОЛЬ: та сама таблиця для ВИПАДКОВОЇ перестановки ρ (зв'язок знищено).
# ПЕРЕДБАЧЕННЯ: вплив монотонно РОСТЕ з наближенням ρ до 0.5 (тобто мовчазні
#   нейрони ще й найменш впливові) ⟹ «пам'яті» немає.  (шанс 80%)
#   Якщо ж смуга ρ<0.01 покаже вплив ВИЩИЙ за середній — це буде пам'ять.
import numpy as np
from scipy.special import ndtr
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(777); NET=6; NMC=32768; INV=0.3989422804014327
BANDS=[(0.0,0.01),(0.01,0.05),(0.05,0.20),(0.20,0.45),(0.45,0.55),
       (0.55,0.80),(0.80,0.95),(0.95,0.99),(0.99,1.01)]
def closure(Wn):
    m=np.zeros(256); C=np.eye(256); Ph=[]
    for l in range(32):
        cp=Wn[l].T@C@Wn[l]; s=np.sqrt(np.maximum(np.diag(cp),1e-20))
        mp=m@Wn[l]; t=mp/s; P=ndtr(t); ph=np.exp(-0.5*t*t)*INV
        mn=mp*P+s*ph; e2=(mp*mp+s*s)*P+mp*s*ph
        C=cp*np.outer(P,P); C[np.diag_indices(256)]=np.maximum(e2-mn*mn,1e-20)
        m=mn; Ph.append(P.copy())
    return Ph
acc={b:[] for b in BANDS}; ctl={b:[] for b in BANDS}; cnt={b:0 for b in BANDS}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    Ph=closure(Wn)
    Bn=[None]*31; B=np.eye(256)
    for l in range(30,-1,-1):
        B=(Wn[l+1]*Ph[l+1])@B; Bn[l]=np.linalg.norm(B,axis=1)
    z=rng.standard_normal((NMC,256)); h=np.maximum(z@Wn[0],0.0)
    rho_all=[]; infl_all=[]
    for l in range(31):
        p=h@Wn[l+1] if l+1<32 else None
        rho=(h>0).mean(0) if l==0 else None
        # ρ для ВИХОДУ шару l рахуємо з передактивації того ж шару
        h=h  # тримаємо h як вихід шару l
        break
    # повний прохід зі збором
    h=z.copy()
    for l in range(31):
        p=h@Wn[l]; hn=np.maximum(p,0.0)
        rho=(p>0).mean(0); infl=Bn[l]*hn.std(0)
        rho_all.append(rho); infl_all.append(infl); h=hn
    rho=np.concatenate(rho_all); infl=np.concatenate(infl_all)
    infl=infl/infl.mean()                      # нормуємо: середній вплив = 1
    perm=rng.permutation(len(rho))
    for b in BANDS:
        m=(rho>=b[0])&(rho<b[1])
        if m.sum()>0:
            acc[b].append(infl[m].mean()); cnt[b]+=int(m.sum())
        mc=(rho[perm]>=b[0])&(rho[perm]<b[1])
        if mc.sum()>0: ctl[b].append(infl[mc].mean())
    print(f'мережа {n}',flush=True)
print(f"\n{'смуга ρ':>16}{'нейронів':>11}{'вплив (сер=1)':>16}{'🔒контроль':>13}")
for b in BANDS:
    if acc[b]:
        print(f"{'[%.2f,%.2f)'%b:>16}{cnt[b]//NET:>11}{np.mean(acc[b]):>16.4f}{np.mean(ctl[b]):>13.4f}")
lo=np.mean(acc[(0.0,0.01)]) if acc[(0.0,0.01)] else float('nan')
mid=np.mean(acc[(0.45,0.55)]) if acc[(0.45,0.55)] else float('nan')
hi=np.mean(acc[(0.99,1.01)]) if acc[(0.99,1.01)] else float('nan')
print(f"\n🎯 МОВЧАЗНІ (ρ<0.01) вплив {lo:.4f}  ·  ПЕРЕМИКАЧІ (ρ≈0.5) {mid:.4f}"
      f"  ·  ЗАВЖДИ ВІДКРИТІ (ρ>0.99) {hi:.4f}")
print("   якщо мовчазні < 1 ⟹ це НЕ пам'ять, а мертвий вантаж")
