# С111-е: ЧИ ПОВЕРТАЄ ЦЕНТРУВАННЯ ЕФЕКТИВНУ ШИРИНУ? (думка ЮЗЕРА про ліміт нейронів)
# Ланцюг, зміряний сьогодні: relu≥0 ⟹ mean(h)=0.487 ⟹ |t|=3.5 ⟹ 70% бітів
# заморожені ⟹ PR 77 із 256. Якщо винен ЗСУВ, то центрування (=LayerNorm без
# масштабу) мусить розморозити біти НА ТИХ САМИХ ВАГАХ.
# ВАРІАНТИ (усі — та сама мережа, ті самі ваги):
#   A ReLU як є
#   B ReLU + центрування h ← h − mean_j(h)         (LayerNorm без ділення)
#   C ReLU + повний LayerNorm  h ← (h−μ)/σ · sd_target
#   🔒D ReLU + віднімання ВИПАДКОВОЇ сталої тієї ж норми (контроль: чи справа
#      в центруванні, чи будь-який зсув щось міняє)
# ПЕРЕДБАЧЕННЯ: B і C піднімуть PR із 77 до 150+ і зронять замерзлих із 35% до
#   <5%; 🔒D не змінить нічого.  (шанс 70%)
import numpy as np
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(31337); NET=4; NMC=32768
def stats(p,h):
    v=h.var(0); med=np.median(h.std(0)); rho=(p>0).mean(0)
    return ((h.std(0)<0.1*med).mean(), v.sum()**2/max((v*v).sum(),1e-300),
            ((rho<0.01)|(rho>0.99)).mean(), np.abs(p.mean(0)/np.maximum(p.std(0),1e-30)).mean())
KEYS=['A ReLU як є','B +центрування','C +повний LayerNorm','🔒D +випадковий зсув']
res={k:[] for k in KEYS}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    z=rng.standard_normal((NMC,256))
    tgt=np.zeros(32); h=z.copy()
    for l in range(32):
        p=h@Wn[l]; h=np.maximum(p,0.0); tgt[l]=h.std(0).mean()
    res['A ReLU як є'].append(stats(p,h))
    for name in KEYS[1:]:
        h=z.copy(); rc=rng.standard_normal(256)
        for l in range(32):
            p=h@Wn[l]; h=np.maximum(p,0.0)
            if name=='B +центрування':
                h=h-h.mean(1,keepdims=True)
            elif name=='C +повний LayerNorm':
                h=(h-h.mean(1,keepdims=True))/np.maximum(h.std(1,keepdims=True),1e-30)*tgt[l]
            else:
                sh=rc/np.linalg.norm(rc)*np.linalg.norm(h.mean(1))/np.sqrt(NMC)*np.sqrt(256)
                h=h-sh
        res[name].append(stats(p,h))
    print(f'мережа {n}',flush=True)
print(f"\n{'варіант':<24}{'замерзлі%':>12}{'PR (з 256)':>12}{'ρ поза%':>10}{'медіана |t|':>13}")
for k in KEYS:
    a=np.array(res[k]).mean(0)
    print(f'{k:<24}{a[0]*100:>12.2f}{a[1]:>12.2f}{a[2]*100:>10.2f}{a[3]:>13.3f}')
