"""ІДЕЯ ЮЗЕРА: «relu зрізає -> на наступний шар іде МЕНША сума, ніж мала б бути».

Ваги N(0,2/d): двійка компенсує зріз У СЕРЕДНЬОМУ ПО АНСАМБЛЮ (E[relu²]=σ²/2).
Але для КОНКРЕТНОЇ мережі — ні. Класика мультиплікативного процесу:
   E[добутку] зберігається, а ТИПОВА реалізація спадає (середнє тягнуть рідкісні викиди)
   бо E[log X] < log E[X] (Єнсен)

ПРОГНОЗ: NNGP рахує ансамблеве середнє -> має СИСТЕМАТИЧНО ЗАВИЩУВАТИ слід реальної мережі.
Якщо так — частина похибки якоря СИСТЕМАТИЧНА, а не випадкова, і знімається універсальною
поправкою БЕЗКОШТОВНО. Це прямо в [[finding-anchor-cost-identity]] (треба 7.6×).

Дані: netbank/micro_trace_cache.pt — 200 нетів, TRN (NNGP-сліди) і TRT (істинні).
"""
import torch, numpy as np, math
LOG='/mnt/win_d/llm/White-Box/drift.log'
def log(m):
    print(m,flush=True); open(LOG,'a').write(m+'\n')
open(LOG,'w').close()

c=torch.load('/mnt/win_d/llm/White-Box/netbank/micro_trace_cache.pt',weights_only=False)
log('ключі кешу: {}'.format(list(c.keys())))
TRN=np.asarray(c['TRN'],dtype=np.float64)      # NNGP
TRT=np.asarray(c['TRT'],dtype=np.float64)      # істина
log('форма TRN {}  TRT {}'.format(TRN.shape,TRT.shape))
nets,K=TRN.shape
REL=(TRN-TRT)/TRT

log('')
log('=== 1) ЧИ СИСТЕМАТИЧНО NNGP ЗАВИЩУЄ? ({} нетів) ==='.format(nets))
log(' відвід   сер.відн.похибка    медіана      частка ЗАВИЩЕНЬ    ст.відх     |сер|/ст.відх')
for k in range(K):
    r=REL[:,k]
    frac=float((r>0).mean())
    log('   {:2d}     {:+.5e}    {:+.5e}      {:.3f}          {:.4e}    {:.3f}'.format(
        k, r.mean(), np.median(r), frac, r.std(), abs(r.mean())/max(r.std(),1e-300)))

log('')
log('=== 2) ЧИ СПАДАЄ ТИПОВИЙ СЛІД ПО ГЛИБИНІ? ===')
log(' відвід   сер Tr_іст     медіана Tr_іст    сер Tr_NNGP    медіана/перший')
for k in range(K):
    log('   {:2d}     {:.4f}      {:.4f}        {:.4f}       {:.4f}'.format(
        k, TRT[:,k].mean(), np.median(TRT[:,k]), TRN[:,k].mean(),
        np.median(TRT[:,k])/max(np.median(TRT[:,0]),1e-300)))

log('')
log('=== 3) ЧИ ЗНІМАЄТЬСЯ УНІВЕРСАЛЬНОЮ ПОПРАВКОЮ? ===')
log('   (ділимо TRN на середній коефіцієнт по ВСІХ нетах -> дивимось залишкову похибку)')
log(' відвід   до поправки RMS    після поправки RMS     виграш')
for k in range(K):
    r=REL[:,k]
    fac=1.0+r.mean()                       # універсальний коефіцієнт (той самий для всіх нетів)
    r2=(TRN[:,k]/fac-TRT[:,k])/TRT[:,k]
    log('   {:2d}     {:.4e}         {:.4e}          {:.2f}×'.format(
        k, np.sqrt((r*r).mean()), np.sqrt((r2*r2).mean()),
        np.sqrt((r*r).mean())/max(np.sqrt((r2*r2).mean()),1e-300)))

log('')
log('=== 4) ЧИ Є ПОПРАВКА СПІЛЬНОЮ ПО ГЛИБИНІ (один множник на шар)? ===')
lead=np.sqrt((REL**2).mean())
corr=np.corrcoef(REL.T)
log('   RMS усієї матриці REL = {:.4e}'.format(lead))
log('   кореляція між відводами (сусідні): ' + ' '.join('{:.3f}'.format(corr[i,i+1]) for i in range(K-1)))
log('   середня позадіагональна кореляція: {:.3f}'.format(
    float((corr.sum()-np.trace(corr))/(K*K-K))))
log('   -> висока кореляція = похибка йде ОДНІЄЮ модою, знімається одним числом на нет')
log('done.')
