# С111-г: ЧИ ТИША — ВЛАСТИВІСТЬ ПОРОГУ, А НЕ АКТИВАЦІЇ? (продовження питання ЮЗЕРА)
# sig111c дав «сигмоїда: 0% замерзлих», АЛЕ насичення теж 0% на всіх шарах ⟹
# нормована сигмоїда сидить у ЛІНІЙНІЙ зоні (sd(p)@31 = 0.1965, σ(x)−0.5 ≈ x/4).
# ⟹ я порівняв ReLU не з сигмоїдою, а фактично з ЛІНІЙНОЮ мережею.
# ТЕСТ: підсилення g у σ(g·p) — при великому g сигмоїда → sign, тобто ПОРІГ.
# ПЕРЕДБАЧЕННЯ: тиша з'явиться і зросте монотонно з g; при g≈16..32 частка
#   замерзлих наблизиться до ReLU-рівня (35%) або перевищить його.  (шанс 75%)
# 🔒 контроль: g→0 мусить дати РІВНО лінійну мережу (PR майже 256, тиша 0).
import numpy as np
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(1234); NET=4; NMC=32768
GS=[0.5,1.0,2.0,4.0,8.0,16.0,32.0]
def relu(x): return np.maximum(x,0.0)
res={}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    z=rng.standard_normal((NMC,256))
    tgt=np.zeros(32); h=z.copy()
    for l in range(32):
        p=h@Wn[l]; hn=relu(p); tgt[l]=hn.std(0).mean(); h=hn
    v=hn.var(0); med=np.median(hn.std(0)); rho=(p>0).mean(0)
    res.setdefault('ReLU',[]).append((( hn.std(0)<0.1*med).mean(),
        v.sum()**2/max((v*v).sum(),1e-300), ((rho<0.01)|(rho>0.99)).mean(), np.nan))
    for g in GS:
        h=z.copy()
        for l in range(32):
            p=h@Wn[l]
            raw=1.0/(1.0+np.exp(-g*p))-0.5
            a=tgt[l]/max(raw.std(0).mean(),1e-300); h=a*raw
        v=h.var(0); med=np.median(h.std(0))
        sat=(np.abs(g*p)>2.0).mean()          # частка семплів у насиченні
        frz=(h.std(0)<0.1*med).mean()
        # аналог ρ: частка часу «на верхній половині»
        rho=(p>0).mean(0)
        res.setdefault(g,[]).append((frz, v.sum()**2/max((v*v).sum(),1e-300),
                                     ((rho<0.01)|(rho>0.99)).mean(), sat))
    print(f'мережа {n}',flush=True)
print('\n=== ТИША ЯК ФУНКЦІЯ ПІДСИЛЕННЯ (шар 31) ===')
print(f"{'активація':>14}{'замерзлі%':>12}{'PR (з 256)':>12}{'ρ поза%':>10}{'насичення%':>12}")
r=np.array(res['ReLU']).mean(0)
for g in GS:
    a=np.array(res[g]).mean(0)
    print(f"{'σ(%.1f·p)'%g:>14}{a[0]*100:>12.2f}{a[1]:>12.2f}{a[2]*100:>10.2f}{a[3]*100:>12.2f}")
print(f"{'ReLU':>14}{r[0]*100:>12.2f}{r[1]:>12.2f}{r[2]*100:>10.2f}{'—':>12}")
