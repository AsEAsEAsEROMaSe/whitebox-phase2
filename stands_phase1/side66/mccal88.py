"""ПЛОМБА ПРИЛАДУ ПЕРЕД БУДЬ-ЯКИМ ВИСНОВКОМ (С88, гілка «вбити дисперсію МК»).

Міряю, а НЕ виводжу з пам'яті:
  1. RMS(істина)² усереднена ПО МЕРЕЖАХ (пам'ять: 0.88404)
  2. Var(a_L)/семпл ⟹ MSE чистої МК при N=65000 (пам'ять: 6.741e-07,
     MC baseline грейдера 6.47e-07)
  3. частка живих нейронів після пілота ⟹ реальна ціна семпла
  4. розподіл t=m/s останнього шару (моя модель t~N(0,4.5²), |t|<2 ⇒ 31.6%)
  5. скільки дає АНТИТЕТИКА на глибині 32 (частка парної дисперсії)
  6. профіль Var(a_k) по глибині
"""
import os
import sys
import time

import numpy as np

W_, L_ = 256, 32
WZ_P = '/mnt/win_d/llm/White-Box/arena60/fs_w.npy'
NNET = int(os.environ.get('NNET', '32'))
NPAIR = int(os.environ.get('NPAIR', '32768'))
CHUNK = 8192


def main():
    wz = np.load(WZ_P, mmap_mode='r')
    V = []; RMS2 = []; ALIVE = []; TS = []; PAIRF = []
    t0 = time.time()
    for net in range(NNET):
        Wf = [np.ascontiguousarray(np.asarray(wz[net][i], dtype=np.float32))
              for i in range(L_)]
        s1 = np.zeros(W_); s2 = np.zeros(W_); spair = np.zeros(W_)
        sp1 = np.zeros(W_); sp2 = np.zeros(W_)
        n = 0; alive = np.zeros(L_)
        rng = np.random.default_rng(4242 + net)
        done = 0
        while done < NPAIR:
            c = min(CHUNK, NPAIR - done)
            z = rng.standard_normal((c, W_), dtype=np.float32)
            a = np.concatenate([z, -z], 0)
            for i in range(L_):
                p = a @ Wf[i]
                if done == 0 and i < L_ - 1:
                    # пілот сабміту: 256 пар = 512 семплів
                    alive[i] = (np.maximum(p[:256], p[c:c + 256]).max(0) > 0).sum()
                a = np.maximum(p, 0.0)
            ad = a.astype(np.float64); pd = p.astype(np.float64)
            s1 += ad.sum(0); s2 += (ad * ad).sum(0)
            pa = 0.5 * (ad[:c] + ad[c:])
            spair += (pa * pa).sum(0)
            sp1 += pd.sum(0); sp2 += (pd * pd).sum(0)
            n += 2 * c; done += c
        mu = s1 / n
        var = s2 / n - mu * mu                       # дисперсія ОДНОГО семпла
        vpair = spair / (n // 2) - mu * mu           # дисперсія середника пари
        mp = sp1 / n
        sd = np.sqrt(np.maximum(sp2 / n - mp * mp, 1e-30))
        V.append(var.mean()); RMS2.append((mu * mu).mean())
        ALIVE.append(alive[:L_ - 1].mean()); TS.append(mp / sd)
        PAIRF.append(2.0 * vpair.mean() / var.mean())
        if net < 3 or net == NNET - 1:
            print(f'  net{net}: Var={var.mean():.5f} RMS²={(mu*mu).mean():.5f} '
                  f'живих={alive[:L_-1].mean():.1f} антитет={PAIRF[-1]:.4f}',
                  flush=True)
    V = np.array(V); RMS2 = np.array(RMS2); TS = np.concatenate(TS)
    al = np.mean(ALIVE); pf = np.mean(PAIRF)
    print(f'\n═══ {NNET} мереж × {2*NPAIR} семплів, {time.time()-t0:.0f}s ═══')
    print(f'RMS(істина)²   сер={RMS2.mean():.5f} медіана={np.median(RMS2):.5f} '
          f'[{RMS2.min():.3f} … {RMS2.max():.3f}]   (пам\'ять: 0.88404)')
    print(f'Var(a_L)/семпл сер={V.mean():.5f} медіана={np.median(V):.5f}')
    print(f'MSE чистої МК N=65000 незалежних = {V.mean()/65000:.4e}   '
          f'(пам\'ять: 6.741e-07)')
    print(f'MSE чистої МК 65000 ПАР (антитет., N=130000) = '
          f'{V.mean()*pf/130000:.4e}')
    print(f'частка дисперсії, що ЛИШАЄТЬСЯ після антитетики: {pf:.4f} '
          f'⟹ антитетика дає ×{1/pf:.3f} проти незалежних пар')
    print(f'живих нейронів (пілот 512 семплів): {al:.1f}/256 = {al/256*100:.1f}%')
    fc = 2.0 * L_ * W_ * W_
    cc = 2.0 * (W_ * al + (L_ - 2) * al * al + al * W_)
    print(f'ціна семпла: повна={fc:.3e} обрізана={cc:.3e} ⟹ ×{fc/cc:.2f}')
    print(f'⟹ N при 8.75%% від B=2.72e11: {0.0875*2.72e11/cc:.0f} семплів')
    print(f'⟹ V/N при цьому N = {V.mean()/(0.0875*2.72e11/cc):.4e} '
          f'(пам\'ять для sub97: 1.514e-06)')
    print(f't=m/s ост.шару: std={TS.std():.3f} |t|<2 → {np.mean(np.abs(TS)<2)*100:.1f}%'
          f'  |t|<1 → {np.mean(np.abs(TS)<1)*100:.1f}%  '
          f'(моя модель: std 4.5, |t|<2 = 34%)')


if __name__ == '__main__':
    sys.exit(main())
