# С111-в: ЧИ Є В СИГМОЇДИ ТАКА САМА ТИША В ГЛИБИНІ? (питання ЮЗЕРА)
# Тиша в ReLU зміряна як 68.4% нейронів із ρ<0.01 на ш31 ⟹ 256 → 58 перемикачів.
# У сигмоїди бінарного «мовчить» немає ⟹ три метрики, щоб порівняння було чесним:
#   (1) ЗАМЕРЗЛІ: частка нейронів, чий вихід майже сталий — sd(h_j) < 0.1·медіана
#   (2) PR ДИСПЕРСІЙ: (Σ v_j)²/Σ v_j², v_j=Var(h_j) — скільки нейронів РЕАЛЬНО несуть
#       варіацію (для ReLU це і є «скільки перемикачів»)
#   (3) для ReLU  ρ_j = P(p_j>0) поза [0.01,0.99]  ·  для сигмоїди — насичення |p_j|>2
# 🔒 ПЛОМБА: ReLU-колонка мусить дати ~68% і PR ≈ 58 на шарі 31.
# ПЕРЕДБАЧЕННЯ: тиші в сигмоїди МЕНШЕ (cos=0 ⟹ стан не колапсує), але вона
#   не нульова — ранг Cov 8.21 теж малий. Шанс, що PR сигмоїди > ReLU у 2×: 70%.
import numpy as np
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(1234); NET=4; NMC=32768
def relu(x): return np.maximum(x,0.0)
def sig0(x): return 1.0/(1.0+np.exp(-x))-0.5
K=['frozen','pr','tail','sd_med']
out={a:{k:np.zeros((NET,32)) for k in K} for a in ['ReLU','сигмоїда норм.']}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    z=rng.standard_normal((NMC,256))
    tgt=np.zeros(32); h=z.copy()
    for l in range(32):
        p=h@Wn[l]; hn=relu(p); tgt[l]=hn.std(0).mean()
        v=hn.var(0); med=np.median(hn.std(0))
        o=out['ReLU']
        o['frozen'][n,l]=(hn.std(0)<0.1*med).mean()
        o['pr'][n,l]=v.sum()**2/max((v*v).sum(),1e-300)
        rho=(p>0).mean(0); o['tail'][n,l]=((rho<0.01)|(rho>0.99)).mean()
        o['sd_med'][n,l]=med
        h=hn
    h=z.copy()
    for l in range(32):
        p=h@Wn[l]; raw=sig0(p); a=tgt[l]/max(raw.std(0).mean(),1e-300); hn=a*raw
        v=hn.var(0); med=np.median(hn.std(0))
        o=out['сигмоїда норм.']
        o['frozen'][n,l]=(hn.std(0)<0.1*med).mean()
        o['pr'][n,l]=v.sum()**2/max((v*v).sum(),1e-300)
        o['tail'][n,l]=(np.abs(p).mean(0)>2.0).mean()   # насичення
        o['sd_med'][n,l]=med
        h=hn
    print(f'мережа {n}',flush=True)
L=[0,1,2,4,8,12,16,20,24,28,31]
print('\n=== ТИША В ГЛИБИНІ ===')
print(f"{'шар':>4}|{'ReLU: замерзлі%':>16}{'PR (з 256)':>12}{'ρ поза [.01,.99]':>18}"
      f" |{'сигм: замерзлі%':>16}{'PR (з 256)':>12}{'насичення%':>12}")
for l in L:
    r=out['ReLU']; s=out['сигмоїда норм.']
    print(f"{l:>4}|{r['frozen'][:,l].mean()*100:>16.2f}{r['pr'][:,l].mean():>12.2f}"
          f"{r['tail'][:,l].mean()*100:>18.2f} |{s['frozen'][:,l].mean()*100:>16.2f}"
          f"{s['pr'][:,l].mean():>12.2f}{s['tail'][:,l].mean()*100:>12.2f}")
r=out['ReLU']; s=out['сигмоїда норм.']
print(f"\n🔒 ПЛОМБА ReLU@31: замерзлих {r['frozen'][:,31].mean()*100:.1f}% · PR {r['pr'][:,31].mean():.1f}"
      f" · ρ поза {r['tail'][:,31].mean()*100:.1f}%   (записано ~68% і 256→58)")
print(f"🎯 PR@31:  ReLU {r['pr'][:,31].mean():.2f}   сигмоїда {s['pr'][:,31].mean():.2f}"
      f"   ⟹ ×{s['pr'][:,31].mean()/max(r['pr'][:,31].mean(),1e-9):.2f}")
print(f"🎯 замерзлих@31:  ReLU {r['frozen'][:,31].mean()*100:.1f}%   сигмоїда {s['frozen'][:,31].mean()*100:.1f}%")
