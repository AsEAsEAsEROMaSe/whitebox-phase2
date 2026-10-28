"""ПРОФІЛЬ ВПОРСКУВАННЯ ДИСПЕРСІЇ ПО ШАРАХ І ПО КАНАЛАХ (С88).

ТОТОЖНІСТЬ (виведена, не вгадана):
    μ_k = G(m_k, s_k) + R_k,  G(m,s)=mΦ(m/s)+sφ(m/s),  m_k = W_kᵀμ_{k-1} ТОЧНО.
⟹ будь-яка вибіркова оцінка = ланцюг, у який на КОЖНОМУ шарі впорскується
   inj_k = Φ·0 + φ(t)·δs_k + δR_k   (δm несеться ланцюгом, не впорскується).

ЩО МІРЯЄМО (усі оцінювачі на ОДНИХ семплах, дисперсія по 64 незалежних жеребах):
  A   чиста МК
  C   оракульна s на ВСІХ шарах (лишається лише канал R)
  D   оракульна R на ВСІХ шарах (лишається лише канал s)
  E   обидва оракули — ПЛОМБА (мусить дати істину точно)
  L_k оракул скрізь, КРІМ шару k ⟹ ізольоване впорскування шару k
  ПЛОМБА АДИТИВНОСТІ: Σ_k (L_k − істина) ≈ (A − істина)

ПЕРЕДБАЧЕННЯ (до числа):
 П1 95%: Σ_k L_k відтворює похибку A з R² > 0.9 (впорскування майже адитивні).
 П2 70%: профіль впорскування СПАДАЄ до глибини (ранні шари дають більше,
         бо їхня похибка встигає «розтруситись»)? — або РІВНИЙ. Ставлю на рівний.
 П3 55%: Var(D) < Var(C) — канал s важчий за канал R.
 П4 30%: Var(C)/Var(A) < 0.1. Пам'ять (С72 last31) дає лише 2.45× на повній
         MSE ⟹ очікую Var(C)/Var(A) ≈ 0.2…0.4, тобто П4 провалиться.
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
NREF = int(os.environ.get('NREF', '524288'))
NBAT = int(os.environ.get('NBAT', '64'))
NPAIR = int(os.environ.get('NPAIR', '13312'))
CHUNK = 16384
ISQ2PI = 0.3989422804014327
ISQ2 = 0.7071067811865476
_erf = np.frompyfunc(math.erf, 1, 1)


def G(m, s):
    t = m / s
    return m * (0.5 * (1.0 + _erf(t * ISQ2).astype(np.float64))) \
        + s * np.exp(-0.5 * t * t) * ISQ2PI


def main():
    wz = np.load(WZ_P, mmap_mode='r')
    agg = []
    for net in range(NNET):
        t0 = time.time()
        Wf = [np.ascontiguousarray(np.asarray(wz[net][i], dtype=np.float32))
              for i in range(L_)]
        Wd = [w.astype(np.float64) for w in Wf]

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
        mu_t = accA / NS; mp_t = accP / NS
        s_t = np.sqrt(np.maximum(accP2 / NS - mp_t * mp_t, 1e-30))
        varL = accA2 / NS - mu_t[-1] ** 2
        prev = np.zeros(W_); m_ch = np.empty((L_, W_))
        for i in range(L_):
            m_ch[i] = prev @ Wd[i]; prev = mu_t[i]
        seal_m = np.abs(m_ch - mp_t).max() / np.abs(mp_t).mean()
        R_t = mu_t - G(m_ch, s_t)
        V_ps = varL.mean(); rms2 = (mu_t[-1] ** 2).mean()
        print(f'[net{net}] ПЛОМБА m=Wᵀμ {seal_m:.2e} | Var/семпл={V_ps:.6f} | '
              f'RMS(іст)²={rms2:.5f} | шум еталона/комп={V_ps/NS:.3e}', flush=True)

        A = np.zeros((NBAT, W_)); C = np.zeros((NBAT, W_))
        D = np.zeros((NBAT, W_)); E = np.zeros((NBAT, W_))
        Lk = np.zeros((NBAT, L_, W_))
        alive = np.zeros(L_)
        for b in range(NBAT):
            rb = np.random.default_rng(777000 + net * 1000 + b)
            z = rb.standard_normal((NPAIR, W_), dtype=np.float32)
            a = np.concatenate([z, -z], 0)
            mC = np.zeros(W_); mD = np.zeros(W_); mE = np.zeros(W_)
            lk = np.zeros((L_, W_)); born = np.zeros(L_, dtype=bool)
            for i in range(L_):
                p = a @ Wf[i]
                a = np.maximum(p, 0.0)
                pd = p.astype(np.float64)
                mh = pd.mean(0)
                sh = np.sqrt(np.maximum((pd * pd).mean(0) - mh * mh, 1e-30))
                ah = a.astype(np.float64).mean(0)
                if b == 0:
                    alive[i] = (p.max(0) > 0).sum()
                st = s_t[i]; Rt = R_t[i]
                Rh = ah - G(mh, sh)                # R̂ з ВИБІРКОВОЮ s
                mC = G(mC @ Wd[i], st) + Rh        # оракул s
                mD = G(mD @ Wd[i], sh) + Rt        # оракул R
                mE = G(mE @ Wd[i], st) + Rt        # обидва
                for k in range(L_):
                    if k < i:
                        lk[k] = G(lk[k] @ Wd[i], st) + Rt
                    elif k == i:
                        lk[k] = G(m_ch[i], sh) + Rh
                        born[k] = True
            A[b] = ah; C[b] = mC; D[b] = mD; E[b] = mE; Lk[b] = lk

        errE = np.abs(E - mu_t[-1]).max()
        eA = A - mu_t[-1]
        eL = Lk - mu_t[-1][None, None, :]
        sumL = eL.sum(1)
        r2 = 1.0 - ((eA - sumL) ** 2).sum() / (eA ** 2).sum()
        print(f'[net{net}] ПЛОМБА E={mu_t[-1].shape} max|Δ|={errE:.2e} | '
              f'ПЛОМБА адитивності R²(ΣL_k vs A)={r2:.4f} | живих '
              f'{alive.mean():.1f}/256', flush=True)
        vA = A.var(0, ddof=1).mean()
        vC = C.var(0, ddof=1).mean(); vD = D.var(0, ddof=1).mean()
        prof = eL.var(1 if False else 0, ddof=1).mean(1)   # дисперсія по жеребах
        print(f'[net{net}] Var A={vA:.4e} | C(оракул s)={vC:.4e} ×{vA/vC:.2f} | '
              f'D(оракул R)={vD:.4e} ×{vA/vD:.2f}  [{time.time()-t0:.0f}s]',
              flush=True)
        pr = prof / prof.sum()
        print('[net%d] профіль впорскування по шарах (%% від суми):' % net)
        print('   ' + ' '.join(f'{100*x:.1f}' for x in pr), flush=True)
        agg.append([vA, vC, vD, V_ps, rms2, alive.mean(), prof.sum()])
        agg_prof = pr if not agg[:-1] else None

    a = np.array(agg)
    print(f'\n═══ ПІДСУМОК {len(agg)} мереж ═══')
    print(f'  Var(a_L)/семпл={a[:,3].mean():.6f}  RMS(іст)²={a[:,4].mean():.5f}'
          f'  живих={a[:,5].mean():.1f}  MSE(МК N=65000)={a[:,3].mean()/65000:.4e}')
    print(f'  A={a[:,0].mean():.4e}  C(оракул s) ×{np.median(a[:,0]/a[:,1]):.2f}'
          f'  D(оракул R) ×{np.median(a[:,0]/a[:,2]):.2f}')


if __name__ == '__main__':
    sys.exit(main())
