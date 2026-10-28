"""ЧИ Є МИГАННЯ ВАЖЕЛЕМ ПОВЕРХ v2 — ДРУГИЙ ДИЗАЙН. (ідея юзера)

ЧОМУ ДРУГИЙ. Перший (`blink_cv.py`) ПРОВАЛИВ власні контролі: f₀ дав 0.908
замість рівно 1.000, shuffle 0.123 замість 1.000. Причина знайдена: я підганяв
матрицю β 31×256 (з нев'язки мигання в похибку КОНКРЕТНИХ нейронів) на 26
рядках і застосовував до іншої мережі, де нейрон `j` — зовсім інший нейрон.
Між мережами такі коефіцієнти безглузді, і система недовизначена.

ЦЕЙ ДИЗАЙН
  ціль      скаляр `r = ⟨E, û⟩ / ‖a_true‖` — відносна похибка вздовж головного
            напрямку (там 98% усієї похибки), БЕЗРОЗМІРНА ⟹ порівнянна між мережами;
  регресори δf, згруповані по глибині (4 групи) — мало параметрів на багато точок;
  дані      8 мереж × 16 жеребів = 128 точок на 4 коефіцієнти;
  β         підганяється на ІНШИХ мережах (5 згорток по мережах).

КОНТРОЛІ
  * δf₀ друкуємо ЧИСЛОМ: антитетика робить f̄₀ = 0.5 тотожно, E[f₀] = 0.5 точно
    ⟹ мусить бути ~0. Це перевірка, що вибірка справді антитетична;
  * SHUFFLE регресорів по жеребах ⟹ R² мусить впасти до ~0;
  * друкуємо R² і на НАВЧАЛЬНИХ мережах — розрив каже, скільки з'їдає підгонка;
  * виграш перекладаємо в 1/(1−R²) — і це СТЕЛЯ, бо E[f] оракульне.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
from sub66_torch import (GATE_N, INV_SQRT_2PI, MIN_PAIRS,     # noqa: E402
                         PILOT_PAIRS, TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                              # noqa: E402
from blink_cv import oracle, run_one                          # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331, 5150, 60613, 727272, 8181,
         101, 22222, 33033, 444, 55, 666, 7007, 80808)
GROUPS = ((1, 8), (8, 16), (16, 24), (24, 32))


def main(nets=8, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    R, DF, DF0 = [], [], []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            a_true, f_true = oracle(wc, dev, 90001 + k, TRUTH_N)
            u = a_true / a_true.norm()
            rr, dd = [], []
            for sd in SEEDS:
                raw, v, fbar = run_one(wc, sd, dev)
                e = raw - v - a_true
                rr.append(float((e @ u) / a_true.norm()))
                d = (fbar - f_true).cpu().numpy()
                DF0.append(d[0])
                dd.append([float(d[a:b].mean()) for a, b in GROUPS])
            R.append(rr); DF.append(dd)
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    R = np.array(R)                       # (мережі, жереби)
    X = np.array(DF)                      # (мережі, жереби, групи)
    n = R.shape[0]
    fold = np.arange(n) % 5

    def run(Xs):
        pr_te, pr_tr = np.zeros_like(R), np.zeros_like(R)
        for c in range(5):
            tr, te = fold != c, fold == c
            A = Xs[tr].reshape(-1, Xs.shape[2])
            A = np.concatenate([A, np.ones((A.shape[0], 1))], 1)
            b = R[tr].reshape(-1)
            w, *_ = np.linalg.lstsq(A, b, rcond=None)
            for m, sel in ((pr_te, te), (pr_tr, tr)):
                B = Xs[sel].reshape(-1, Xs.shape[2])
                B = np.concatenate([B, np.ones((B.shape[0], 1))], 1)
                m[sel] = (B @ w).reshape(m[sel].shape)
        def r2(p, sel=None):
            return 1.0 - float(((R - p) ** 2).sum() / ((R - R.mean()) ** 2).sum())
        return r2(pr_te), r2(pr_tr)

    rs = np.random.default_rng(5)
    Xsh = X.copy()
    for i in range(n):
        Xsh[i] = X[i][rs.permutation(X.shape[1])]

    print(f'\n{n} мереж × {len(SEEDS)} жеребів = {R.size} точок, '
          f'{X.shape[2]} регресори, оракульне E[f] на {TRUTH_N}')
    print(f'δf₀ (мусить бути ~0): {np.mean(np.abs(DF0)):.3e}   '
          f'типова |δf| груп: {np.abs(X).mean():.3e}')
    print(f'відносна похибка v2 вздовж û: {np.abs(R).mean()*100:.3f}%')
    print(f'\n{"регресори":26s} {"R² тест":>9s} {"R² навч":>9s} {"1/(1−R²)":>10s}')
    print('-' * 58)
    for nm, Xs in (('δf по 4 групах глибини', X), ('SHUFFLE по жеребах', Xsh)):
        te, tr = run(Xs)
        print(f'{nm:26s} {te:+9.4f} {tr:+9.4f} '
              f'{(1.0/(1.0-te) if te < 0.999 else float("inf")):10.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
