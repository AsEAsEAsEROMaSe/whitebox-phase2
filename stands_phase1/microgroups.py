"""МІКРОГРУПИ Й РІДКІСНІ СПРАЦЮВАННЯ (ідея юзера С47: аналогія зі складанням фото)

АНАЛОГІЯ ЮЗЕРА: складаєш кадри з камери — нерухоме накопичується, рухоме зникає.
У нас те саме: прибиті нейрони = нерухоме тло (складається когерентно),
рідкісні перемикання = те, що рухається, і усереднення їх топить.
ПИТАННЯ: чи не в них уся справа? Чи не програємо ми, пакуючи рідкісне разом з усім?

ПІДСТАВА: всередині групи з ОДНАКОВИМ візерунком затворів мережа СТРОГО ЛІНІЙНА
(усі ReLU вже вирішили) ⟹ задача всередині групи розв'язується точно.

МІРЯЄМО:
 A) СТЕЛЯ СТРАТИФІКАЦІЇ: беремо k затворів 1-го шару, групуємо за їх візерунком (2^k груп),
    рахуємо частку дисперсії МІЖ групами. Це максимум, який дала б ідеальна стратифікація.
 B) РІДКІСНІ СПРАЦЮВАННЯ (ядро ідеї юзера): для кожного семпла рахуємо, скільки ПРИБИТИХ
    затворів (p<0.05 або p>0.95, по всіх 32 шарах) опинились у своєму РІДКІСНОМУ стані.
    Групуємо за цим числом і дивимось: чи несуть рідкісні семпли НЕПРОПОРЦІЙНО багато
    дисперсії? Якщо так — їх варто виловлювати окремо.
 C) наскільки взагалі семпли різняться візерунком: середня відстань Геммінга між затворами.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/microgroups.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def run(net):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    N=200_000; CH=25_000
    # ---- прохід 1: p_i по всіх затворах ----
    P=[torch.zeros(D,device=dev,dtype=torch.float64) for _ in range(L)]
    n=0
    for c in range(N//CH):
        g=torch.Generator(device=dev).manual_seed(3141+c)
        a=torch.randn(CH,D,device=dev,generator=g)
        for li in range(L):
            h=a@W[li]; P[li]+=(h>0).double().sum(0); a=torch.relu(h)
        n+=CH
    P=[p/n for p in P]
    pin=[( (p<0.05)|(p>0.95) ) for p in P]                   # прибиті
    rare_state=[(p<0.05).float() for p in P]                  # для прибитих: рідкісний стан = ВІДКРИТИЙ якщо p мале
    npin=int(sum(int(x.sum()) for x in pin))
    ham=float(2*sum(float((p*(1-p)).sum()) for p in P))
    log('  прибитих затворів: {} з {} ({:.1%});  сер. відстань Геммінга між семплами = {:.0f} з {}'
        .format(npin,32*D,npin/(32*D),ham,32*D))
    # ---- прохід 2: збираємо ознаки + вихід ----
    KS=(4,8,12)
    keys={k:[] for k in KS}; nrare=[]; Y=[]
    gsel=torch.Generator(device=dev).manual_seed(99)
    sel=torch.randperm(D,generator=gsel,device=dev)[:max(KS)]
    for c in range(N//CH):
        g=torch.Generator(device=dev).manual_seed(3141+c)
        x=torch.randn(CH,D,device=dev,generator=g)
        a=x; cnt=torch.zeros(CH,device=dev)
        for li in range(L):
            h=a@W[li]; gt=(h>0).float()
            if li==0:
                s=gt[:,sel]
                for k in KS:
                    keys[k].append((s[:,:k]*(2**torch.arange(k,device=dev,dtype=torch.float32))).sum(1).long())
            # скільки ПРИБИТИХ затворів у РІДКІСНОМУ стані
            m=pin[li]
            cnt+=((gt[:,m]-rare_state[li][m][None,:]).abs()<0.5).float().sum(1)
            a=torch.relu(h)
        nrare.append(cnt); Y.append(a.double())
    for k in KS: keys[k]=torch.cat(keys[k])
    nrare=torch.cat(nrare); Y=torch.cat(Y); NT=Y.shape[0]
    E=Y.mean(0); tot=float(((Y-E)**2).sum())
    log('  A) СТЕЛЯ СТРАТИФІКАЦІЇ за затворами 1-го шару:')
    def between_frac(key,G):
        cntg=torch.zeros(G,device=dev).index_add_(0,key,torch.ones(NT,device=dev))
        sm=torch.zeros(G,D,dtype=torch.float64,device=dev).index_add_(0,key,Y)
        mg=sm/cntg.clamp(min=1)[:,None].double()
        return float((cntg.double()[:,None]*(mg-E)**2).sum())/tot, int((cntg>0).sum())
    log('     (КОНТРОЛЬ = ті самі групи з ПЕРЕМІШАНИМИ мітками: чиста упередженість дрібних груп)')
    for k in KS:
        key=keys[k]; G=2**k
        f,nz=between_frac(key,G)
        gp=torch.Generator(device=dev).manual_seed(555+k)
        fsh,_=between_frac(key[torch.randperm(NT,generator=gp,device=dev)],G)
        log('     k={:2d}, {:5d} груп (непорожніх {:5d}): виміряно {:.4f}  контроль {:.4f}  ⟹ СПРАВЖНЄ {:.4f}'
            .format(k,G,nz,f,fsh,max(f-fsh,0.0)))
    log('  B) РІДКІСНІ СПРАЦЮВАННЯ прибитих затворів (ядро ідеї):')
    log('     скільки прибитих у рідкісному стані: сер={:.1f} std={:.1f} мін={} макс={}'
        .format(float(nrare.mean()),float(nrare.std()),int(nrare.min()),int(nrare.max())))
    q=torch.quantile(nrare.float(),torch.tensor([.1,.5,.9,.99],device=dev))
    log('     квантилі 10/50/90/99%: {:.0f} / {:.0f} / {:.0f} / {:.0f}'.format(*[float(v) for v in q]))
    err=((Y-E)**2).sum(1)
    idx=torch.argsort(nrare)
    parts=[float(err[idx[b*NT//10:(b+1)*NT//10]].sum())/tot for b in range(10)]
    log('     децилі за кількістю рідкісних спрацювань, частка дисперсії (рівномірно 0.100):')
    log('       '+'  '.join('{:.3f}'.format(p) for p in parts))
    log('     корел(рідкісні, дисперсія семпла) = {:+.4f}'
        .format(float(torch.corrcoef(torch.stack([nrare.double(),err]))[0,1])))
    log('     верхній дециль / нижній = {:.2f}×'.format(parts[-1]/max(parts[0],1e-12)))

for net in (0,1):
    log('==== НЕТ {} ===='.format(net)); run(net)
log('done.')
