"""РОЗДІЛИТИ ПОХИБКУ САБМІТА НА ДВІ ПРИЧИНИ: ХИБНІ ВОРОТА ПРОТИ ХИБНОГО `c` (С78).

`linrelu.py` дав ТОЧНУ тотожність (пломба 5.35e-09) — вона справджується і для
істини, і для будь-якої скінченної вибірки:

    m_ℓ  = (m_{ℓ−1}·W_ℓ)⊙g_ℓ + c_ℓ        g = P(p>0),  c = Cov(p, gate)
    m̂_ℓ  = (m̂_{ℓ−1}·W_ℓ)⊙ĝ_ℓ + ĉ_ℓ        (ĝ, ĉ — вибіркові, тотожність ТОЧНА)

Уся відповідь тримається на ДВОХ наборах чисел — воротах `g_ℓ` і зсувах `c_ℓ`.
Ми знаємо обидва лише вибірково. Питання «у що бити, щоб шукати ті 45×» — це
питання, який із двох наборів коштує дорожче.

⛔ АДИТИВНИЙ РОЗКЛАД `e = Σ перенос(mp⊙δg) + Σ перенос(δc)` ВИРОДЖЕНИЙ і був
   першою (хибною) версією цього стенда. Алгебра:
       δc = δm_relu − mp·δg − δmp·ĝ      ⟹  доданок `mp·δg` СКОРОЧУЄТЬСЯ
   тобто це різання однієї малої величини на дві великі протилежні. Зміряно:
   кожна половина сама по собі у ~4.3× БІЛЬША за суму. Це той самий закон, що
   й [[finding-closure-is-a-balance-point-not-an-approximation]].

✅ ПРАВИЛЬНА ПОСТАНОВКА — КОНТРФАКТИЧНІ РЕКУРСІЇ. Проганяємо ту саму точну
   рекурсію, підмінюючи рівно один набір на істинний:

       обидва вибіркові → m̂₃₂ (пломба: тотожність)   похибка = e
       ТОЧНІ ворота + вибіркові c                    похибка = скільки лишиться
       вибіркові ворота + ТОЧНІ c                    похибка = скільки лишиться
       обидва точні   → m₃₂  (пломба: рівно 0)

   Тут нічого не гаситься штучно: кожен рядок — це справжній оцінювач, який ми
   могли б побудувати, якби знали відповідну половину.

📏 Плюс профіль «ціна одного шару»: точний набір ЛИШЕ на шарі L, решта вибіркові.

📣 ПЕРЕДБАЧЕНО: ворота — це частка з `n` семплів (шум ~1/√n, однаковий на всіх
   шарах), `c` — коваріація, шумніша. Але `δg` множиться на `m_{ℓ−1}W`, яке
   росте вглиб. Ставлю на ворота як на дорожчу половину.

🚨 Стенд БЕЗ обрізання мертвих і без λ-поправок перших 8 шарів: обрізання не
   змінює алгебру взагалі, а λ-поправки ловлять частину саме цих двох джерел на
   шарах 1..8 — тож тут міряється СИРИЙ розклад, до них.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 19   # істина: 2^LOGN ПАР
NPAIR = int(sys.argv[4]) if len(sys.argv) > 4 else 4096  # робоча вибірка, пар
DRAWS = int(sys.argv[5]) if len(sys.argv) > 5 else 4     # жеребів робочої вибірки
W_ = 256
D = 32
CH = 1 << 14   # пар у пачці

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1].split('🚨')[0], flush=True)
print(f'істина 2·2^{LOGN} = {2*(1<<LOGN)} семплів; робоча вибірка 2·{NPAIR} = '
      f'{2*NPAIR}; шум істини у {np.sqrt((1<<LOGN)/NPAIR):.1f}× менший\n', flush=True)


def _accum(W, z):
    """Пропустити пачку [z;−z] і повернути суми p, relu(p), gate по шарах."""
    x = torch.cat([z, -z], 0)
    Sp, Sr, Sg = [], [], []
    for L in range(D):
        p = x @ W[L]                       # 🚨 fp32: fp64 на P104 у 32× повільніше
        Sp.append(p.double().sum(0))
        Sg.append((p > 0).double().sum(0))
        x = torch.relu(p)
        Sr.append(x.double().sum(0))
    return Sp, Sr, Sg


def _gc(W, seed, npairs, whiten=False):
    """(g, c, mp, m) для вибірки з `npairs` пар. mp_ℓ = mean(p_ℓ) — ТОЧНО m_{ℓ−1}W_ℓ."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Sp = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    Sr = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    Sg = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        if whiten:
            # ZCA рівно як у сабміті: з антитетикою xᵀx = 2·zᵀz ⟹ cov = zᵀz/b
            cov = (z.T @ z) / float(b)
            ev, U = torch.linalg.eigh(cov.double())
            z = (z.double() @ (U * ev.clamp_min(1e-8).rsqrt()) @ U.T).float()
        a, r, gg = _accum(W, z)
        for L in range(D):
            Sp[L] += a[L]; Sr[L] += r[L]; Sg[L] += gg[L]
        done += b
    N = float(2 * npairs)
    mp = [s / N for s in Sp]
    m = [s / N for s in Sr]
    g = [s / N for s in Sg]
    c = [m[L] - mp[L] * g[L] for L in range(D)]
    return g, c, mp, m


KS = (4, 8, 16, 24, 28, 30, 31)
rows, seals, share, kmap = [], [], [], []
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    Wd = [w.double() for w in W]
    with torch.no_grad():
        g, c, mp, m = _gc(W, 91_000 + 13 * n, 1 << LOGN)                 # істина
        truth = m[-1]

        def rec(gs, cs):
            """Та сама ТОЧНА рекурсія m_ℓ = (m_{ℓ−1}W)⊙g + c, набори на вибір."""
            v = torch.zeros(W_, dtype=torch.float64, device=DEV)
            for L in range(D):
                v = (v @ Wd[L]) * gs[L] + cs[L]
            return v

        # 🚨 ВІДНОШЕННЯ СУМ, не середнє відношень: `‖e‖²` одного жереба саме
        #    флуктуує в рази, і ділення на нього роздуває викиди (зміряно: від
        #    0.22 до 10.8 при поділі пожеребно).
        acc = np.zeros(3)                 # d2, точні ворота, точні c
        accL = np.zeros((2, D))           # профіль по шарах
        accK = np.zeros((len(KS), 2))     # узгоджена підміна
        for d in range(DRAWS):
            gh, ch, mph, mh = _gc(W, 55_000 + 7 * n + 1013 * d, NPAIR, whiten=True)
            # ✅ПЛОМБИ: обидва вибіркові → m̂₃₂ ; обидва точні → m₃₂
            if d == 0:
                seals.append((float((rec(gh, ch) - mh[-1]).norm() / mh[-1].norm()),
                              float((rec(g, c) - truth).norm() / truth.norm())))
            acc += [float((mh[-1] - truth).pow(2).sum()),
                    float((rec(g, ch) - truth).pow(2).sum()),
                    float((rec(gh, c) - truth).pow(2).sum())]

            for L in range(D):
                a = list(gh); a[L] = g[L]
                b = list(ch); b[L] = c[L]
                accL[0, L] += float((rec(a, ch) - truth).pow(2).sum())
                accL[1, L] += float((rec(gh, b) - truth).pow(2).sum())

            for i, K in enumerate(KS):
                ta = list(gh); tb = list(ch)
                ha = list(gh); hb = list(ch)
                for L in range(D):
                    if L >= K:
                        ta[L] = g[L]; tb[L] = c[L]
                    else:
                        ha[L] = g[L]; hb[L] = c[L]
                accK[i, 0] += float((rec(ta, tb) - truth).pow(2).sum())
                accK[i, 1] += float((rec(ha, hb) - truth).pow(2).sum())

        rows.append(acc)
        share.append(accL)
        kmap.append(accK)
    del W, Wd
    torch.cuda.empty_cache()
    r = rows[-1]
    print(f'  … мережа {n+1}/{NETS}  MSE_наш={r[0]/DRAWS:.3e}  залишок: '
          f'ворота {r[1]/r[0]:.3f} / c {r[2]/r[0]:.3f}  '
          f'пломби {seals[-1][0]:.1e} {seals[-1][1]:.1e}', flush=True)

R = np.array(rows).sum(0)          # відношення СУМ по всіх мережах і жеребах
S = np.array(share).sum(0) / R[0]
KM = np.array(kmap).sum(0) / R[0]
SE = np.array(seals)
print(f'\n{NETS} мереж × {DRAWS} жеребів (відношення СУМ)\n')
print(f'✅ПЛОМБА 1 (обидва вибіркові → m̂₃₂): {np.median(SE[:,0]):.2e}')
print(f'✅ПЛОМБА 2 (обидва точні → m₃₂):     {np.median(SE[:,1]):.2e}')
print('   (обидві мусять бути ~1e-7 — це fp32 у матмулах, рекурсія ТОЧНА)\n')

print('залишок MSE (частка від нашої ‖e‖²), якщо ТОЧНО знати ОДИН шар:')
print(f'{"шар":>4s} {"ворота шару L":>15s} {"c шару L":>12s}')
print('-' * 34)
for L in range(D):
    best = min(S[0, L], S[1, L])
    star = ' <<<' if best < 0.90 else ''
    print(f'{L:4d} {S[0,L]:15.4f} {S[1,L]:12.4f}{star}')

print(f'\n{"":=^70s}')
print(f'ЗНАТИ ТОЧНО ВСІ ВОРОТА:  залишок MSE {R[1]/R[0]:.4f}  '
      f'⟹ виграш {R[0]/max(R[1],1e-30):.2f}×')
print(f'ЗНАТИ ТОЧНО ВСІ `c`:     залишок MSE {R[2]/R[0]:.4f}  '
      f'⟹ виграш {R[0]/max(R[2],1e-30):.2f}×')

print(f'\n🔑 УЗГОДЖЕНА підміна — ОБИДВА набори точні:')
print(f'{"K":>4s} {"хвіст ℓ≥K: залишок":>20s} {"виграш":>9s} '
      f'{"голова ℓ<K: залишок":>21s} {"виграш":>9s}')
print('-' * 68)
for i, K in enumerate(KS):
    print(f'{K:4d} {KM[i,0]:20.4f} {1/max(KM[i,0],1e-9):8.2f}× '
          f'{KM[i,1]:21.4f} {1/max(KM[i,1],1e-9):8.2f}×')
print('📊 БАЛ: 1.954e-07 — вимір структурний, прайсується картою (K,ε)')
