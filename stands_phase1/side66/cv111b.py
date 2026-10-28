# С111-б: ВИРОК. Та сама контрольна змінна g = mean(relu(p0)) − ‖w‖/√(2π), але
# база — РЕАЛЬНА база сабміту: вибілювання Холецьким + антитетика в шарі 0.
# 🚨 ЗАКОН: новий конструкт змагається НЕ з голою МК, а з вибілюванням
#    (так померли сферизація ×0.74, RQMC ×1.0156, вирівнювання 4-х моментів).
# 🚨 ВИПРАВЛЕНО БАГ cv111: adjoint() стартував з l=1 при m=0,C=I ⟹ трактував
#    вхід шару 1 як N(0,I), хоча там relu(z@W0). Тепер рекурсія з l=0.
#
# ВАРІАНТИ:
#   A  база (вибілювання + антитетика), без КЗ
#   B  −g@Bdet   детермінований Φ із замикання (виправлений)
#   B0 −g@Bbug   старий (з багом) — 🔒 пломба, наскільки баг коштував
#   E  наш λ-шлях: v=g, v←v@W_l·gate_l, gate ЕМПІРИЧНИЙ з 1024 рядків
#   C  ОРАКУЛ (фіт по цій же репліці) — стеля
#  🔒D чужа карта (Φ сусідньої мережі)
#
# ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (щоб не тлумачити заднім числом):
#   вибілювання з'їсть частину ⟹ B впаде з ×1.226 до ×1.10..1.18  (шанс 60%)
#   E (емпіричний gate) ≈ B ± 3%: обидва оцінюють ту саму Φ        (шанс 55%)
#   якщо B > E на >5% — детермінований Φ кращий за пілотний gate, і це ДАРОМ
import numpy as np
from scipy.special import ndtr
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(2718); NET=8; NPAIR=3072; REP=48; NTRUE=262144
INV=0.3989422804014327; GATE_N=1024

def adjoint(Wn,skip0=False):
    """B = (W1·Φ1)···(W31·Φ31). skip0=True відтворює БАГ cv111."""
    m=np.zeros(256); C=np.eye(256); B=np.eye(256)
    for l in range(1 if skip0 else 0,32):
        cp=Wn[l].T@C@Wn[l]; s=np.sqrt(np.maximum(np.diag(cp),1e-20))
        mp=m@Wn[l]; t=mp/s; Phi=ndtr(t); ph=np.exp(-0.5*t*t)*INV
        if l>=1: B=B@(Wn[l]*Phi)
        mn=mp*Phi+s*ph; e2=(mp*mp+s*s)*Phi+mp*s*ph
        C=cp*np.outer(Phi,Phi); C[np.diag_indices(256)]=np.maximum(e2-mn*mn,1e-20)
        m=mn
    return B

def fwd(z,Wn):
    h=np.maximum(z@Wn[0],0.0)
    for l in range(1,32): h=np.maximum(h@Wn[l],0.0)
    return h

KEYS=['A база (вибіл.+антитет.)','B детермін. Φ (виправл.)','B0 детермін. Φ (з багом)',
      'E наш λ-шлях (емпір. gate)','C ОРАКУЛ (фіт)','🔒D чужа карта']
res={}; Wprev=None
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    sg0=np.linalg.norm(Wn[0],axis=0)                 # ‖стовпці W0‖
    Bdet=adjoint(Wn); Bbug=adjoint(Wn,skip0=True)
    Bfor=adjoint(Wprev) if Wprev is not None else None
    Wprev=Wn
    acc=np.zeros(256)
    for i in range(0,NTRUE,8192): acc+=fwd(rng.standard_normal((8192,256)),Wn).sum(0)
    T=acc/NTRUE
    e={k:[] for k in KEYS}
    for r in range(REP):
        z=rng.standard_normal((NPAIR,256))
        cov=(z.T@z)/NPAIR
        wh=np.linalg.inv(np.linalg.cholesky(cov)).T   # як у сабміті
        w0=wh@Wn[0]
        y=z@w0
        x=np.concatenate([np.maximum(y,0.0),np.maximum(-y,0.0)],axis=0)
        denom=2.0*NPAIR
        g=x.sum(0)/denom - sg0*INV                    # КЗ з ТОЧНО відомим середнім
        # прохід + наш λ-перенос з ЕМПІРИЧНИМ gate
        v=g.copy(); h=x
        for l in range(1,32):
            p=h@Wn[l]
            gate=(p[:GATE_N]>0.0).sum(0)/GATE_N
            v=(v@Wn[l])*gate
            h=np.maximum(p,0.0)
        base=h.sum(0)/denom
        e['A база (вибіл.+антитет.)'].append(((base-T)**2).sum())
        e['B детермін. Φ (виправл.)'].append(((base-g@Bdet-T)**2).sum())
        e['B0 детермін. Φ (з багом)'].append(((base-g@Bbug-T)**2).sum())
        e['E наш λ-шлях (емпір. gate)'].append(((base-v-T)**2).sum())
        if Bfor is not None: e['🔒D чужа карта'].append(((base-g@Bfor-T)**2).sum())
        Gc=x-x.mean(0); Yc=h-h.mean(0)
        Bopt=np.linalg.lstsq(Gc,Yc,rcond=None)[0]
        e['C ОРАКУЛ (фіт)'].append(((base-g@Bopt-T)**2).sum())
    for k,vv in e.items():
        if vv: res.setdefault(k,[]).append(np.mean(vv))
    print(f'мережа {n}',flush=True)
print()
b=np.mean(res['A база (вибіл.+антитет.)'])
print(f"{'варіант':<28}{'MSE середнього':>18}{'редукція V':>13}")
for k in KEYS:
    if k in res:
        a=np.mean(res[k]); print(f'{k:<28}{a:>18.6e}{b/a:>13.4f}')
# знаковий тест B проти E, помережно
bb=np.array(res['B детермін. Φ (виправл.)']); ee=np.array(res['E наш λ-шлях (емпір. gate)'])
print(f'\nзнаковий B<E: {(bb<ee).sum()}/{len(bb)}   B/E = {ee.mean()/bb.mean():.4f}')
