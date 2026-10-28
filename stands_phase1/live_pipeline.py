"""ЧИ ВИЖИВЕ КОНЦЕНТРАЦІЯ В ПОВНОМУ КОНВЕЄРІ? (обов'язковий крок перед будівництвом, С47)

closure_where.py показав: похибка замикання сидить на ЖИВИХ (прибита половина несе 0.3-1.1%).
АЛЕ там бралися ІСТИННІ mu, sd. У справжньому методі вони ВЛАСНІ й накопичено криві —
і разом з ними може поїхати саме ВИЗНАЧЕННЯ, який нейрон живий.

ПОВНЕ ГАУСОВЕ ЗАМИКАННЯ (робимо чесно, з накопиченням):
  на кожному шарі беремо поточні (m, K) активацій, ПІДМІНЯЄМО розподіл гаусовим із тими
  самими моментами (Холецький + свіжі семпли) і тягнемо далі. Це РІВНО те, що робить
  аналітичний метод, включно з накопиченням похибки.
  Паралельно жене ІСТИННИЙ потік (без підміни) на тих самих сідах.

МІРЯЄМО на кожному шарі:
  1) похибка замикання по нейронах |m_gauss - m_true|
  2) ЖИВІСТЬ, ВИЗНАЧЕНА З ВЛАСНОГО (кривого) стану методу: p_hat = Phi(mu_hat/sd_hat)
  3) чи концентрується похибка на живих ЗА ВЛАСНОЮ класифікацією (це те, що доступне методу)
  4) чи правильно метод класифікує (узгодженість p_hat з істинним p)
ЧИТАННЯ: якщо концентрація зникає або класифікація ламається — важіль мертвий, будувати нема сенсу.
"""
import torch, time, numpy as np, math
LOG='/mnt/win_d/llm/White-Box/live_pipeline.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; DT=torch.float64
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
SQ2PI=math.sqrt(2*math.pi)
def Phi(z): return 0.5*(1+torch.erf(z/math.sqrt(2)))

def run(net,N=200_000):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev).double() for i in range(L)]
    g=torch.Generator(device=dev).manual_seed(555)
    x=torch.randn(N,D,device=dev,generator=g,dtype=DT)
    at=x.clone()       # істинний потік
    ag=x.clone()       # гаусів потік (з підміною на кожному шарі)
    log('  шар | похибка ‖m_g−m_t‖/‖m_t‖ | живих(власна оцінка) | їх частка похибки | конц. | помилок класиф.')
    for li in range(L):
        ht=at@W[li]; hg=ag@W[li]
        # --- істина ---
        mt=torch.relu(ht).mean(0)
        pt=(ht>0).double().mean(0)
        # --- гаусів прогноз ЦЬОГО кроку з ВЛАСНОГО стану ---
        mu=hg.mean(0); sd=hg.std(0)
        z=mu/sd
        mg=mu*Phi(z)+sd*torch.exp(-0.5*z*z)/SQ2PI
        ph=Phi(z)                                   # ЖИВІСТЬ за власною оцінкою методу
        err=(mg-mt).abs()
        live=torch.minimum(ph,1-ph)>0.05            # класифікація, доступна методу
        livet=torch.minimum(pt,1-pt)>0.05           # істинна
        f_neu=float(live.double().mean())
        f_err=float(err[live].sum()/err.sum()) if err.sum()>0 else float('nan')
        miscls=float((live!=livet).double().mean())
        if li in (0,1,2,4,8,16,24,30,31):
            log('   {:3d} |        {:.4f}         |       {:.3f}        |       {:.3f}       | ×{:.1f}  |     {:.3f}'
                .format(li,float((mg-mt).norm()/mt.norm()),f_neu,f_err,
                        f_err/max(f_neu,1e-9),miscls))
        # --- крок далі: істина природно, гаусів потік ЧЕРЕЗ ПІДМІНУ ---
        at=torch.relu(ht)
        rg=torch.relu(hg)
        m=rg.mean(0); Kc=((rg-m).t()@(rg-m))/rg.shape[0]
        Kc=Kc+1e-10*torch.eye(D,dtype=DT,device=dev)
        Ls=torch.linalg.cholesky(Kc)
        gg=torch.Generator(device=dev).manual_seed(9000+li)
        ag=m[None,:]+torch.randn(N,D,device=dev,generator=gg,dtype=DT)@Ls.t()   # ПІДМІНА
    log('  ПІДСУМОК: ‖m_g−m_t‖/‖m_t‖ на виході = {:.4f}'.format(float((mg-mt).norm()/mt.norm())))

for net in (0,1):
    log('==== НЕТ {} ===='.format(net)); run(net)
log('done.')
