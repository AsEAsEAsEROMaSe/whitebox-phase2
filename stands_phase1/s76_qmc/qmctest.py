"""№3: QMC — ЗАМІНИТИ ЖЕРЕБ НА SCRAMBLED SOBOL. (С76)

════════ ЧОМУ ЦЕ ВЗАГАЛІ НЕ МІРЯНО ════════
Весь розрив до лідера — ДИСПЕРСІЯ (зміщення лише 7.6%). Шум має ранг ~2.5
(масштаб уздовж `m`) ⟹ ефективна розмірність інтегранда НИЗЬКА. Для таких
інтегралів квазівипадкові точки (Sobol) збігаються швидше за 1/√n — інколи
аж до 1/n. Клас «змінити РОЗПОДІЛ семплів» закритий (importance sampling
потребує відповіді), але це НЕ зміна розподілу: маргінально ті самі N(0,I),
змінюється лише СПІЛЬНЕ розташування точок. Кошик — «знімати в джерелі»
(як вибілювання, 1.98×), не «передбачати поширення».

════════ ЩО МІРЯЄМО ════════
База = рівно sub87 (run72 з усіма λ-каналами). Варіант = те саме, але
`z_ext` — scrambled Sobol → інверсія нормального CDF. Антитетика і
вибілювання ЛИШАЮТЬСЯ (x = [z,−z], бонусом Sobol уже майже вибілений).
Крос-ціль на двох незалежних істинах ПОВНОЇ мережі, MSE = ⟨d1,d2⟩.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 чесно: невідомо, 1.0–1.3×. Скептик: антитетика+вибілювання вже зняли
   легку структуру, а ⛔радіальний RB 0.995× натякає, що парний радіальний
   шум — не головний. Оптиміст: ранг шуму 2.5 ⟹ ефективна розмірність мала.
🟢 ГЕЙТ: ≥1.10× — жити далі (повний прогін 24 мереж), ≥1.5× — ПОРТУВАТИ
   в сабміт сьогодні. <1.05× — закрити клас.
🚨 ПЛОМБИ
  ✅ z_ext=randn(той самий генератор) мусить дати РІВНО базу (побітово);
  ✅ u кламповано від 0/1 (ndtri(0)=-inf);
  ✅ вирок лише на ≥24 мережах (малі прогони брешуть).
"""
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (closure_parts, run_side1 as run72, W_,               # noqa: E402
                    cut_and_budget)
from biassplit import truth_full                                         # noqa: E402
from varsub import S72, A72, Q72, G72                                    # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:1'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 16
# MODE 'p2': обидві гілки на 2^⌊log2(npair)⌋ точок — Sobol тримає
# рівномірність ЛИШЕ на степенях двійки, 4331 точка ламає баланс
MODE = sys.argv[4] if len(sys.argv) > 4 else ''
KSIG = 8


def sobol_z(npair, seed, dev):
    """Scrambled Sobol (256 вимірів) → N(0,1) через інверсію CDF."""
    eng = torch.quasirandom.SobolEngine(W_, scramble=True, seed=int(seed))
    u = eng.draw(npair, dtype=torch.float64)
    u = u.clamp(1e-12, 1.0 - 1e-12)
    return torch.special.ndtri(u).float().to(dev)


def main(nets=24, dev=DEV, ndraw=NDRAW):
    t0 = time.time()
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'№3 SOBOL ЗАМІСТЬ ЖЕРЕБУ · {nets} мереж × {ndraw} жеребів, '
          f'крос-ціль, {dev}')
    print('📣 ПЕРЕДБАЧЕНО: невідомо, 1.0–1.3×; гейт 1.10×, порт від 1.5×\n',
          flush=True)

    MB = np.zeros(nets)      # MSE бази по мережі
    MQ = np.zeros(nets)      # MSE Sobol по мережі
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            if MODE == 'p2':
                npair = 1 << int(np.floor(np.log2(npair)))
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, KSIG, dev, extras=True)
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)
            if n == 0:
                # ✅ПЛОМБА: z_ext з того ж генератора = РІВНО база
                g = torch.Generator(device=dev).manual_seed(71_000_000)
                zz = torch.randn(npair, W_, device=dev, generator=g)
                p1, _ = run72(wc, npair, s_cl, m_cl, dev, 71_000_000,
                              lam_s=S72, lam_a=A72, lam_q=Q72, lam_g=G72,
                              q_ext=q_cl, p_ext=f_cl)
                p2, _ = run72(wc, npair, s_cl, m_cl, dev, 0, z_ext=zz,
                              lam_s=S72, lam_a=A72, lam_q=Q72, lam_g=G72,
                              q_ext=q_cl, p_ext=f_cl)
                dd = float((p1 - p2).abs().max())
                print(f'✅ПЛОМБА z_ext: max|Δ| = {dd:.2e} '
                      f'{"✅" if dd < 1e-12 else "🛑 СТЕНД БИТИЙ"}', flush=True)
            ab = aq = 0.0
            for k in range(ndraw):
                sd = 71_000_000 + 977 * n + k
                pb, _ = run72(wc, npair, s_cl, m_cl, dev, sd,
                              lam_s=S72, lam_a=A72, lam_q=Q72, lam_g=G72,
                              q_ext=q_cl, p_ext=f_cl)
                zq = sobol_z(npair, sd, dev)
                pq, _ = run72(wc, npair, s_cl, m_cl, dev, 0, z_ext=zq,
                              lam_s=S72, lam_a=A72, lam_q=Q72, lam_g=G72,
                              q_ext=q_cl, p_ext=f_cl)
                d1, d2 = pb - T1, pb - T2
                e1, e2 = pq - T1, pq - T2
                ab += float((d1 * d2).sum())
                aq += float((e1 * e2).sum())
            MB[n], MQ[n] = ab / ndraw, aq / ndraw
        del W, wc
        torch.cuda.empty_cache()
        print(f'  {n+1:>2}/{nets}  база {MB[n]:.3e}  sobol {MQ[n]:.3e}  '
              f'({MB[n]/max(MQ[n],1e-30):.3f}×)  [{time.time()-t0:.0f}s]',
              flush=True)

    base, qmc = float(np.mean(MB)), float(np.mean(MQ))
    r = base / qmc
    wins = int(np.sum(MB > MQ))
    print(f'\nMSE база  {base:.5e}')
    print(f'MSE sobol {qmc:.5e}')
    print(f'\n🔑 ВИРОК: **{r:.4f}×**  (перемог по мережах {wins}/{nets})')
    print(f'   {"🟢 ПОРТУВАТИ" if r >= 1.5 else ("🟡 ЖИВЕ, міряти далі" if r >= 1.10 else "⛔ ЗАКРИТИ КЛАС")}')
    print(f'📊 БАЛ: 1.954e-07 → {1.954e-07/r:.4e} (якщо перенесеться)')
    print(f'   час {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
