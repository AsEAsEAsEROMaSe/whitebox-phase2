"""РОЗКЛАД ДИСПЕРСІЇ МК ПО КАНАЛАХ ВПОРСКУВАННЯ (С88). CPU/numpy.

ТОТОЖНІСТЬ (виведення, не здогад). Для кожного шару
    μ_k = E[relu(p_k)] = G(m_k, s_k) + R_k ,  G(m,s)=mΦ(m/s)+sφ(m/s)
    m_k = W_kᵀ μ_{k-1}  — ТОЧНО (лінійність),  s_k = std(p_k),  R_k = негаус. нев'язка.
Отже ланцюг «детермінований перенос середнього + впорскування» ТОЧНИЙ, а вся
похибка будь-якої оцінки μ̂_L = сума впорскувань через ДВА канали: δs і δR.

ОЦІНЮВАЧІ (усі на ОДНИХ І ТИХ САМИХ семплах):
  A: чиста МК  μ̂_L = mean relu(p_L)
  B: ланцюг із перенесеним m (≡ що робить `v` у сабміті), вибіркові s,R
  C: ланцюг, s = ОРАКУЛ ⟹ лишається лише канал R
  D: ланцюг, R = ОРАКУЛ ⟹ лишається лише канал s
  E: обидва оракули ⟹ мусить дати ІСТИНУ (ПЛОМБА тотожності)

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (записано до числа):
 П1 (95%): chain E збігається з істиною до 1e-6 відносно.
 П2 (60%): Var(B) ≈ Var(A) ±20% — перенос середнього НЕ знижує дисперсію.
 П3 (70%): Var(D) < Var(C), канал `s` домінує (моя алгебра дає ~3:1).
 П4 (35%): Var(C)/Var(A) < 0.1. Пам'ять С72 (`last31`) каже, що оракульна σ
           дає лише 2.45× на ПОВНІЙ MSE ⟹ швидше за все П4 ПРОВАЛИТЬСЯ.
"""
import math
import os
import sys
import time

import numpy as np

W_ = 256
L_ = 32
WZ_P = '/mnt/win_d/llm/White-Box/arena60/fs_w.npy'

NNET = int(os.environ.get('NNET', '4'))
NREF = int(os.environ.get('NREF', '524288'))    # пар в еталоні
NBAT = int(os.environ.get('NBAT', '64'))
NPAIR = int(os.environ.get('NPAIR', '13312'))   # N = 26624
CHUNK = 16384
ISQ2PI = 0.3989422804014327
ISQ2 = 0.7071067811865476

_erf = np.frompyfunc(math.erf, 1, 1)


def Phi(x):
    return 0.5 * (1.0 + _erf(x * ISQ2).astype(np.float64))


def G(m, s):
    t = m / s
    return m * Phi(t) + s * np.exp(-0.5 * t * t) * ISQ2PI


def main():
    wz = np.load(WZ_P, mmap_mode='r')
    rows = []
    for net in range(NNET):
        t0 = time.time()
        Wf = [np.ascontiguousarray(np.asarray(wz[net][i], dtype=np.float32))
              for i in range(L_)]
        Wd = [w.astype(np.float64) for w in Wf]

        # ═══ ЕТАЛОН ═══
        accA = np.zeros((L_, W_)); accP = np.zeros((L_, W_))
        accP2 = np.zeros((L_, W_)); accA2 = np.zeros(W_)
        rng = np.random.default_rng(1000 + net)
        done = 0
        while done < NREF:
            c = min(CHUNK, NREF - done)
            z = rng.standard_normal((c, W_), dtype=np.float32)
            a = np.concatenate([z, -z], 0)
            for i in range(L_):
                p = a @ Wf[i]
                a = np.maximum(p, 0.0)
                pd = p.astype(np.float64)
                accP[i] += pd.sum(0); accP2[i] += (pd * pd).sum(0)
                accA[i] += a.astype(np.float64).sum(0)
            accA2 += (a.astype(np.float64) ** 2).sum(0)
            done += c
        NS = 2.0 * NREF
        mu_t = accA / NS
        mp_t = accP / NS
        s_t = np.sqrt(np.maximum(accP2 / NS - mp_t * mp_t, 1e-30))
        varL = accA2 / NS - mu_t[-1] ** 2

        prev = np.zeros(W_)
        m_ch = np.empty((L_, W_))
        for i in range(L_):
            m_ch[i] = prev @ Wd[i]
            prev = mu_t[i]
        seal_m = np.abs(m_ch - mp_t).max() / np.abs(mp_t).mean()
        R_t = mu_t - G(m_ch, s_t)

        V_ps = varL.mean(); rms2 = (mu_t[-1] ** 2).mean()
        print(f'[net{net}] ПЛОМБА m=Wᵀμ: {seal_m:.2e} | Var(a_L)/семпл={V_ps:.6f}'
              f' | RMS(істина)²={rms2:.5f} | MSE(МК,N=65000)={V_ps/65000:.4e}',
              flush=True)

        # ═══ ЖЕРЕБИ ═══
        est = {k: np.zeros((NBAT, W_)) for k in 'ABCDE'}
        alive_cnt = np.zeros(L_)
        for b in range(NBAT):
            rb = np.random.default_rng(777000 + net * 1000 + b)
            z = rb.standard_normal((NPAIR, W_), dtype=np.float32)
            a = np.concatenate([z, -z], 0)
            mB = np.zeros(W_); mC = np.zeros(W_)
            mD = np.zeros(W_); mE = np.zeros(W_)
            for i in range(L_):
                p = a @ Wf[i]
                a = np.maximum(p, 0.0)
                pd = p.astype(np.float64)
                mh = pd.mean(0)
                sh = np.sqrt(np.maximum((pd * pd).mean(0) - mh * mh, 1e-30))
                ah = a.astype(np.float64).mean(0)
                if b == 0:
                    alive_cnt[i] = (p.max(0) > 0).sum()
                st = s_t[i]
                mB = G(mB @ Wd[i], sh) + (ah - G(mh, sh))
                mC = G(mC @ Wd[i], st) + (ah - G(mh, st))
                mD = G(mD @ Wd[i], sh) + R_t[i]
                mE = G(mE @ Wd[i], st) + R_t[i]
            est['A'][b] = ah; est['B'][b] = mB
            est['C'][b] = mC; est['D'][b] = mD; est['E'][b] = mE

        errE = np.abs(est['E'] - mu_t[-1]).max()
        print(f'[net{net}] ПЛОМБА chain E vs істина: max|Δ|={errE:.3e} '
              f'(рівень {np.abs(mu_t[-1]).mean():.4f}) | живих сер '
              f'{alive_cnt.mean():.1f}/256', flush=True)

        res = {}
        for k in 'ABCD':
            v = est[k].var(0, ddof=1).mean()
            b2 = ((est[k].mean(0) - mu_t[-1]) ** 2).mean()
            res[k] = (v, b2)
        vA = res['A'][0]
        print(f'[net{net}] ДИСПЕРСІЯ ОЦІНКИ, N={2*NPAIR}:', flush=True)
        for k in 'ABCD':
            v, b2 = res[k]
            print(f'    {k}: Var={v:.4e}  ×{vA/v:8.2f}  bias²={b2:.3e}',
                  flush=True)
        print(f'    [{time.time()-t0:.0f}s]', flush=True)
        rows.append([res[k][0] for k in 'ABCD'] + [V_ps, rms2,
                                                   alive_cnt.mean()])

    a = np.array(rows)
    print(f'\n═══ ПІДСУМОК по {len(rows)} мережах ═══')
    print(f'  Var(a_L)/семпл = {a[:,4].mean():.6f} | RMS(істина)² = '
          f'{a[:,5].mean():.5f} | живих {a[:,6].mean():.1f}/256')
    print(f'  MSE чистої МК при N=65000 = {a[:,4].mean()/65000:.4e}')
    for i, k in enumerate('ABCD'):
        print(f'  {k}: Var={a[:,i].mean():.4e}  ×{a[:,0].mean()/a[:,i].mean():.2f}'
              f'  (медіана відношення {np.median(a[:,0]/a[:,i]):.2f})')


if __name__ == '__main__':
    sys.exit(main())
