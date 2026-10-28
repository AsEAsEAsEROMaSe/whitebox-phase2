# С111-б: СИГМОЇДА ПРИ ФІКСОВАНОМУ РЕЖИМІ. Сира сигмоїда програла МАСШТАБОМ
# (sd(p) ×3.8 за шар ⟹ стан = константа з 4-го шару), а питання було про
# ГЛАДКІСТЬ. Наш закон: вісь режиму — насиченість воріт, порівнювати лише при
# фіксованому режимі.
# ⟹ h_l = a_l·(σ(p_l) − 0.5), де a_l підбирається так, щоб sd(h_l) збігалось
#    з ReLU-траєкторією того самого шару. Форма сигмоїди ЗБЕРЕЖЕНА, режим теж.
# 🔒 контроль: та сама процедура, застосована до самої ReLU, мусить дати a_l≈1
#    і відтворити ReLU-колонку — інакше нормування само по собі щось міняє.
# ПЕРЕДБАЧЕННЯ: похибка кроку сигмоїди < ReLU у 3..10× (гладка функція краще
#    описується двома моментами); але γ₄ і колапс рангу лишаться близькими,
#    бо їх робить множення на W, а не форма активації.  (шанс 65%)
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(1234); NET=4; NMC=32768; NRANK=32768
GX,GW=hermegauss(64); GW=GW/GW.sum()
def relu(x): return np.maximum(x,0.0)
def sig0(x): return 1.0/(1.0+np.exp(-x))-0.5
def effrank(H):
    C=np.cov(H,rowvar=False); ev=np.linalg.eigvalsh(C); ev=np.maximum(ev,0)
    return ev.sum()**2/max((ev*ev).sum(),1e-300)

KEYS=['sd_p','sd_h','g4','cos','erank','err_step','a_l']
out={a:{k:np.zeros((NET,32)) for k in KEYS} for a in ['ReLU','сигмоїда норм.','🔒ReLU норм.']}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    z=rng.standard_normal((NMC,256)); z2=rng.standard_normal((NMC,256))
    # 1) еталонна ReLU-траєкторія: запам'ятовуємо sd(h_l)
    tgt=np.zeros(32); h=z.copy(); h2=z2.copy()
    for l in range(32):
        p=h@Wn[l]; p2=h2@Wn[l]; hn=relu(p); h2n=relu(p2)
        tgt[l]=hn.std(0).mean()
        o=out['ReLU']; o['sd_p'][n,l]=p.std(0).mean(); o['sd_h'][n,l]=tgt[l]; o['a_l'][n,l]=1.0
        c=p-p.mean(0); o['g4'][n,l]=((c**4).mean(0)/np.maximum(p.std(0)**4,1e-30)-3.0).mean()
        pred=(relu(p.mean(0)[:,None]+p.std(0)[:,None]*GX[None,:])*GW[None,:]).sum(1)
        o['err_step'][n,l]=np.linalg.norm(pred-hn.mean(0))/max(np.linalg.norm(hn.mean(0)),1e-30)*100
        num=(hn*h2n).sum(1); den=np.linalg.norm(hn,axis=1)*np.linalg.norm(h2n,axis=1)
        o['cos'][n,l]=(num/np.maximum(den,1e-30)).mean(); o['erank'][n,l]=effrank(hn[:NRANK])
        h=hn; h2=h2n
    # 2) сигмоїда і ReLU, обидві з ПОШАРОВИМ нормуванням на ту саму sd
    for name,f in [('сигмоїда норм.',sig0),('🔒ReLU норм.',relu)]:
        h=z.copy(); h2=z2.copy(); o=out[name]
        for l in range(32):
            p=h@Wn[l]; p2=h2@Wn[l]
            raw=f(p); a=tgt[l]/max(raw.std(0).mean(),1e-300)
            hn=a*raw; h2n=a*f(p2)
            o['a_l'][n,l]=a; o['sd_p'][n,l]=p.std(0).mean(); o['sd_h'][n,l]=hn.std(0).mean()
            c=p-p.mean(0); o['g4'][n,l]=((c**4).mean(0)/np.maximum(p.std(0)**4,1e-30)-3.0).mean()
            pred=a*(f(p.mean(0)[:,None]+p.std(0)[:,None]*GX[None,:])*GW[None,:]).sum(1)
            o['err_step'][n,l]=np.linalg.norm(pred-hn.mean(0))/max(np.linalg.norm(hn.mean(0)),1e-30)*100
            num=(hn*h2n).sum(1); den=np.linalg.norm(hn,axis=1)*np.linalg.norm(h2n,axis=1)
            o['cos'][n,l]=(num/np.maximum(den,1e-30)).mean(); o['erank'][n,l]=effrank(hn[:NRANK])
            h=hn; h2=h2n
    print(f'мережа {n}',flush=True)

L=[0,1,2,4,8,12,16,20,24,28,31]
print('\n=== ФІКСОВАНИЙ РЕЖИМ: ТА САМА sd(h) НА КОЖНОМУ ШАРІ ===')
for name in out:
    o=out[name]; print(f'\n--- {name} ---')
    print(f"{'шар':>4}{'a_l':>9}{'sd(p)':>9}{'sd(h)':>9}{'γ₄':>8}{'cos(h,h′)':>11}{'eff.rank':>10}{'похибка кроку %':>17}")
    for l in L:
        print(f"{l:>4}{o['a_l'][:,l].mean():>9.4f}{o['sd_p'][:,l].mean():>9.4f}{o['sd_h'][:,l].mean():>9.4f}"
              f"{o['g4'][:,l].mean():>8.3f}{o['cos'][:,l].mean():>11.4f}{o['erank'][:,l].mean():>10.2f}"
              f"{o['err_step'][:,l].mean():>17.4f}")
r=out['ReLU']; s=out['сигмоїда норм.']; c=out['🔒ReLU норм.']
print(f"\n🔒 КОНТРОЛЬ нормування (ReLU норм. мусить = ReLU): a_l сер {c['a_l'].mean():.6f}"
      f" · похибка кроку {c['err_step'].mean():.4f}% проти {r['err_step'].mean():.4f}%"
      f" · ранг@31 {c['erank'][:,31].mean():.2f} проти {r['erank'][:,31].mean():.2f}")
print(f"\n🎯 ПОХИБКА КРОКУ (сер. по шарах):  ReLU {r['err_step'].mean():.4f}%"
      f"   сигмоїда норм. {s['err_step'].mean():.4f}%   ⟹ ×{r['err_step'].mean()/max(s['err_step'].mean(),1e-12):.2f}")
print(f"   на шарі 31:                    ReLU {r['err_step'][:,31].mean():.4f}%"
      f"   сигмоїда норм. {s['err_step'][:,31].mean():.4f}%")
print(f"🎯 КОЛАПС:  ранг@31  ReLU {r['erank'][:,31].mean():.2f}  сигмоїда {s['erank'][:,31].mean():.2f}"
      f"  ·  cos@31  ReLU {r['cos'][:,31].mean():.4f}  сигмоїда {s['cos'][:,31].mean():.4f}")
print(f"🎯 γ₄@31:  ReLU {r['g4'][:,31].mean():.4f}   сигмоїда {s['g4'][:,31].mean():.4f}")
