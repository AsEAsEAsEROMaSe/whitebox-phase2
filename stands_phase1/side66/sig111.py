# С111: СИГМОЇДА В ГЛИБИНІ (чітка вимога ЮЗЕРА). Ті самі ваги, та сама мережа,
# активація замінена на σ(x)=1/(1+e^{-x}). Дивимось ПО ШАРАХ 0..31.
# 🔒 ПЛОМБИ (ReLU-колонка мусить відтворити корпус, інакше стенд не читати):
#     частка p>0 = 0.491 на кожному шарі · γ₄(ш31) = 0.3947 · cos@31 = 0.9741
#     похибка кроку E[relu] = 0.139% · eff.rank Cov(h) 165.6(ш1) → 2.8(ш31)
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(1234); NET=4; NMC=32768
GX,GW=hermegauss(64); GW=GW/GW.sum()          # вузли для N(0,1)

def relu(x): return np.maximum(x,0.0)
def sigm(x): return 1.0/(1.0+np.exp(-x))
ACTS={'ReLU':relu,'сигмоїда':sigm}

def step_gauss(m,s,f):
    """E[f(P)] при P~N(m,s²) — гаусова гіпотеза кроку, квадратура 64 вузли."""
    return (f(m[:,None]+s[:,None]*GX[None,:])*GW[None,:]).sum(1)

def effrank(H):
    C=np.cov(H,rowvar=False); ev=np.linalg.eigvalsh(C); ev=np.maximum(ev,0)
    return ev.sum()**2/max((ev*ev).sum(),1e-300)

out={a:{k:np.zeros((NET,32)) for k in
        ['sd_p','mean_h','frac_pos','g3','g4','cos','erank','err_step','sd_h']}
     for a in ACTS}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    z=rng.standard_normal((NMC,256)); z2=rng.standard_normal((NMC,256))
    for aname,f in ACTS.items():
        h=z.copy(); h2=z2.copy()
        for l in range(32):
            p=h@Wn[l]; p2=h2@Wn[l]
            mu=p.mean(0); sd=p.std(0)
            c=p-mu
            g3=(c**3).mean(0)/np.maximum(sd**3,1e-30)
            g4=(c**4).mean(0)/np.maximum(sd**4,1e-30)-3.0
            hn=f(p); h2n=f(p2)
            # похибка КРОКУ: гаусова формула на РЕАЛЬНИХ (m,s) проти реального
            pred=step_gauss(mu,sd,f); real=hn.mean(0)
            err=np.linalg.norm(pred-real)/max(np.linalg.norm(real),1e-30)
            o=out[aname]
            o['sd_p'][n,l]=sd.mean(); o['mean_h'][n,l]=real.mean()
            o['sd_h'][n,l]=hn.std(0).mean()
            o['frac_pos'][n,l]=(p>0).mean(); o['g3'][n,l]=g3.mean(); o['g4'][n,l]=g4.mean()
            o['err_step'][n,l]=err*100.0
            num=(hn*h2n).sum(1); den=np.linalg.norm(hn,axis=1)*np.linalg.norm(h2n,axis=1)
            o['cos'][n,l]=(num/np.maximum(den,1e-30)).mean()
            o['erank'][n,l]=effrank(hn[:4096])
            h=hn; h2=h2n
    print(f'мережа {n}',flush=True)

L=[0,1,2,4,8,12,16,20,24,28,30,31]
print('\n=== ПОВЕДІНКА В ГЛИБИНІ (середнє по 4 мережах) ===')
for aname in ACTS:
    o=out[aname]
    print(f'\n--- {aname} ---')
    print(f"{'шар':>4}{'sd(p)':>9}{'mean(h)':>10}{'sd(h)':>9}{'частка p>0':>12}"
          f"{'γ₃':>8}{'γ₄':>8}{'cos(h,h′)':>11}{'eff.rank':>10}{'похибка кроку %':>17}")
    for l in L:
        print(f"{l:>4}{o['sd_p'][:,l].mean():>9.4f}{o['mean_h'][:,l].mean():>10.4f}"
              f"{o['sd_h'][:,l].mean():>9.4f}{o['frac_pos'][:,l].mean():>12.4f}"
              f"{o['g3'][:,l].mean():>8.3f}{o['g4'][:,l].mean():>8.3f}"
              f"{o['cos'][:,l].mean():>11.4f}{o['erank'][:,l].mean():>10.2f}"
              f"{o['err_step'][:,l].mean():>17.4f}")
r=out['ReLU']; s=out['сигмоїда']
print('\n🔒 ПЛОМБИ ReLU проти корпусу:')
print(f"   частка p>0 (сер. по шарах)  {r['frac_pos'].mean():.4f}   записано 0.491")
print(f"   γ₄ шар 31                   {r['g4'][:,31].mean():.4f}   записано 0.3947")
print(f"   cos@31                      {r['cos'][:,31].mean():.4f}   записано 0.9741")
print(f"   похибка кроку (сер.)        {r['err_step'].mean():.4f}%  записано 0.139%")
print(f"   eff.rank ш1 → ш31           {r['erank'][:,1].mean():.1f} → {r['erank'][:,31].mean():.2f}   записано 165.6 → 2.8")
print(f"\n🎯 ПОХИБКА КРОКУ: ReLU {r['err_step'][:,31].mean():.4f}%  проти  сигмоїда {s['err_step'][:,31].mean():.4f}%  (шар 31)")
print(f"   середнє по всіх шарах:  ReLU {r['err_step'].mean():.4f}%  сигмоїда {s['err_step'].mean():.4f}%")
