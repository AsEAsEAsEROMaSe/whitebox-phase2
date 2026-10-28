"""№4: ЗАТВОР З УСІХ СЕМПЛІВ, А НЕ З ПЕРШИХ 1024. (С76)

════════ ФАКТ ІЗ КОДУ (`run_side1` / сабміт) ════════
    gate = (p[:GATE_N] > 0).sum(0) / GATE_N,  GATE_N = 1024
Активації `p` ВЖЕ пораховані для всіх 2·npair ≈ 8662 семплів — але маска
переносу оцінюється лише з перших 1024. Шум затвора множиться на `v`
МУЛЬТИПЛІКАТИВНО на КОЖНОМУ з 31 шару переносу. Використати всі рядки —
клас «вже оплачене, але викинуте»: доплата лише порівняння+сума
(~0.1–0.15% бюджету, частково є запас 0.03%).

════════ ЩО МІРЯЄМО ════════
ПАРНЕ порівняння (той самий seed ⟹ ті самі z): база GATE_N=1024 проти
GATE_N=2·npair (монкіпатч base72.GATE_N перед другим прогоном).
Крос-ціль, MSE = ⟨d1,d2⟩.

📣 ПЕРЕДБАЧЕННЯ: шум затвора на шар ~√(q(1−q)/1024)≈1.5% відн., по ланцюгу
   ~√31·1.5%≈8% відносної похибки масштабу `v`; зняття 2.9× цього шуму дасть
   1.00–1.05× (чесно не знаю, чи затвор — помітна частка). ГЕЙТ: ≥1.01×
   при ціні ≤0.15% бюджету. ⚠️пам'ять: «затвор НЕ горло» (радіальний RB) —
   але то був ІНШИЙ механізм (розподіл семплів), не шум маски.
🚨 ПЛОМБИ: ✅відновити base72.GATE_N після патчу · ✅той самий seed в обох
   гілках · ✅вирок ≥24 мереж.
"""
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import base72                                                            # noqa: E402
from base72 import (closure_parts, run_side1 as run72,                   # noqa: E402
                    cut_and_budget)
from biassplit import truth_full                                         # noqa: E402
from varsub import S72, A72, Q72, G72                                    # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:1'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 16
KSIG = 8


def main(nets=24, dev=DEV, ndraw=NDRAW):
    t0 = time.time()
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'№4 ПОВНИЙ ЗАТВОР · {nets} мереж × {ndraw} жеребів, парне '
          f'порівняння, {dev}')
    print('📣 ПЕРЕДБАЧЕНО: 1.00–1.05×; гейт ≥1.01× (ціна ≤0.15% бюджету)\n',
          flush=True)

    MB = np.zeros(nets)
    MG = np.zeros(nets)
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, KSIG, dev, extras=True)
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)
            ab = ag = 0.0
            for k in range(ndraw):
                sd = 71_000_000 + 977 * n + k
                pb, _ = run72(wc, npair, s_cl, m_cl, dev, sd,
                              lam_s=S72, lam_a=A72, lam_q=Q72, lam_g=G72,
                              q_ext=q_cl, p_ext=f_cl)
                base72.GATE_N = 2 * npair
                try:
                    pg, _ = run72(wc, npair, s_cl, m_cl, dev, sd,
                                  lam_s=S72, lam_a=A72, lam_q=Q72, lam_g=G72,
                                  q_ext=q_cl, p_ext=f_cl)
                finally:
                    base72.GATE_N = 1024
                d1, d2 = pb - T1, pb - T2
                e1, e2 = pg - T1, pg - T2
                ab += float((d1 * d2).sum())
                ag += float((e1 * e2).sum())
            MB[n], MG[n] = ab / ndraw, ag / ndraw
        del W, wc
        torch.cuda.empty_cache()
        print(f'  {n+1:>2}/{nets}  база {MB[n]:.3e}  повн.затвор {MG[n]:.3e}  '
              f'({MB[n]/max(MG[n],1e-30):.3f}×)  [{time.time()-t0:.0f}s]',
              flush=True)

    base, gf = float(np.mean(MB)), float(np.mean(MG))
    r = base / gf
    wins = int(np.sum(MB > MG))
    print(f'\nMSE база          {base:.5e}')
    print(f'MSE повний затвор {gf:.5e}')
    print(f'\n🔑 ВИРОК: **{r:.4f}×**  (перемог по мережах {wins}/{nets})')
    print(f'📊 БАЛ: 1.954e-07 → {1.954e-07/r:.4e} (до ціни флопів)')
    print(f'   час {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
