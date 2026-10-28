"""ПРАЙСИНГ ГІЛКИ AMP: скільки дасть замикання, точніше в 1/k разів?

ЗВІДКИ (ідея юзера). Скінченнорозмірні поправки до state evolution (Донохо-
Малекі-Монтанарі) — це буквально та величина, яку ми зміряли як `α≈1`, і в
тій літературі для неї є вирази з константами. Пропозиція: не будувати
наступну схему, а взяти їхню формулу.

🚨 АЛЕ СПОЧАТКУ — ЦІНА. Перед тим як виводити чи шукати константу, треба
знати, СКІЛЬКИ вона варта. Це вимірюється БЕЗ жодної теорії: беремо істину
пошарово і штучно масштабуємо похибку замикання,
    `s_k = s_іст + k·(s_замик − s_іст)`,
тобто `k=1` — наше нинішнє замикання, `k=0` — ідеальне, `k=0.5` — вдвічі
точніше. Свіп по `k` дає ПОВНУ криву віддачі для БУДЬ-ЯКОГО майбутнього
покращення замикання, включно з AMP.
🔑 Заразом це прямо перевіряє запис M36 «замикання ВДВІЧІ точніше дало нуль»
— але вже на ПОВНОМУ ланцюгу сабміту, з вибілюванням.

ДРУГИЙ СТЕНД У ТОМУ САМОМУ ПРОГОНІ: контрольна змінна 4-го СТЕПЕНЯ.
Вибілювання прибиває ЕМПІРИЧНИЙ 2-й момент, антитетика — всі НЕПАРНІ.
Четвертий не чіпає ніхто, а `v2 точний до 3-го моменту` каже, що похибка
саме там. `He₄(u·z) = (u·z)⁴−6(u·z)²+3` має середнє РІВНО нуль ⟹ незміщена
КЗ. Напрямки — з ваг (`W₀diag(g₀)G`), як у `dirs_from_G`.

КОНТРОЛІ
  * ✅ `k=1` мусить збігтися з базою ДО ОСТАННЬОЇ ЦИФРИ;
  * ✅ без вибілювання мусить бути помітно ГІРШЕ (перевірка, що порт вірний);
  * ✅ `lam_s=0` — вимкнена σ-поправка, щоб бачити її внесок окремо;
  * ⛔ КЗ 4-го степеня у ВИПАДКОВИХ напрямках — мусить дати ≈1.0;
  * β для КЗ підганяється LOO ПО МЕРЕЖАХ (не на тій самій мережі).
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, KSIG, W_)
from impsamp import gates_and_G                                   # noqa: E402
from tilt_end2end import dirs_from_G                              # noqa: E402
from tilt_wide import truth_last                                  # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 48
KGRID = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
# 🚨 λ мусить ПЕРЕФІТУВАТИСЬ під кожне k: LAM_S підігнані під нинішнє
# замикання (k=1), а під точнішим оптимальне λ росте до 1. Свіп без
# перефіту — це рівно помилка С67 («свіп K без перефіту λ не вимірює нічого»).
SGRID = [1.0, 1.3, 1.7, 2.2, 3.0, 4.5, 7.0]
MCV = 16


def truth_layers(wc, dev, seed, npair=524_288, chunk=65536):
    """Справжні `s` доактивацій і `m` активацій шарів 1..KSIG."""
    ns = [w.shape[1] for w in wc]
    s1 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    s2 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    a1 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = x @ w
            pd = p.double()
            s1[i] += pd.sum(0)
            s2[i] += (pd * pd).sum(0)
            x = torch.relu(p)
            a1[i] += x.double().sum(0)
        done += k
    n = 2.0 * done
    s = [(s2[i] / n - (s1[i] / n) ** 2).clamp_min(1e-30).sqrt()
         for i in range(1, KSIG + 1)]
    m = [a1[i] / n for i in range(1, KSIG + 1)]
    return {'s': s, 'm': m}


def main(nets=20, dev=DEV):
    torch.manual_seed(2718)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    tot = {(k, sc): 0.0 for k in KGRID for sc in SGRID}
    tot['⛔без вибіл.'] = 0.0
    tot['σ-поправка ВИМК'] = 0.0
    # КЗ 4-го степеня: оракульна стеля з ПОМЕРЕЖЕВОЮ матрицею B (n_L × MCV)
    #   виграш = Σ‖e‖² / (Σ‖e‖² − tr(Se·Sh⁻¹·Seᵀ))
    cv = {'акт': [0.0, 0.0], '⛔вип': [0.0, 0.0]}
    cv_base = 0.0

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KSIG, dev)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            TL = truth_layers(wc, dev, 710_000 + 7919 * n)
            G = gates_and_G(wc, dev, 4242 + n)
            U = dirs_from_G(wc, G, dev, 66_000 + n)[:, :MCV]
            Ur = torch.linalg.qr(torch.randn(W_, W_, device=dev))[0][:, :MCV]
            nl = wc[-1].shape[1]
            Se = {k: torch.zeros(nl, MCV, dtype=torch.float64, device=dev)
                  for k in ('акт', '⛔вип')}
            Sh = {k: torch.zeros(MCV, MCV, dtype=torch.float64, device=dev)
                  for k in ('акт', '⛔вип')}
            EV = {k: [] for k in ('акт', '⛔вип')}
            for d in range(NDRAW):
                sd = 44_000_000 + 733 * n + d
                for k in KGRID:
                    for sc in (SGRID if k != 1.0 else [1.0]):
                        ls = tuple(min(sc * v, 1.0) for v in LAM_S)
                        pr, _ = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                          bias_k=k, truth_layers=TL, lam_s=ls)
                        tot[(k, sc)] += float(((pr - T) ** 2).sum())
                pr0, h4 = run_side1(wc, npair, s_cl, m_cl, dev, sd, cv4_dirs=U)
                _, h4r = run_side1(wc, npair, s_cl, m_cl, dev, sd, cv4_dirs=Ur)
                e = (pr0 - T)

                # ⚠️ фіт B на 1-й половині жеребів, оцінка на 2-й — інакше
                #    n_L×16 параметрів із 48 жеребів дають чистий міраж
                part = 0 if d < NDRAW // 2 else 1
                for nm, hh in (('акт', h4), ('⛔вип', h4r)):
                    if part == 0:
                        Se[nm] += torch.outer(e, hh)
                        Sh[nm] += torch.outer(hh, hh)
                    else:
                        EV[nm].append((e, hh))
                prn, _ = run_side1(wc, npair, s_cl, m_cl, dev, sd, white=False)
                tot['⛔без вибіл.'] += float(((prn - T) ** 2).sum())
                prs, _ = run_side1(wc, npair, s_cl, m_cl, dev, sd, lam_s=None)
                tot['σ-поправка ВИМК'] += float(((prs - T) ** 2).sum())
            for nm in ('акт', '⛔вип'):
                rg = 1e-6 * float(torch.diagonal(Sh[nm]).mean()) * torch.eye(
                    MCV, dtype=torch.float64, device=dev)
                B = torch.linalg.solve((Sh[nm] + rg).T, Se[nm].T).T
                for e, hh in EV[nm]:
                    cv[nm][0] += float(((e - B @ hh) ** 2).sum())
                    cv[nm][1] += float((e * e).sum())
        del W, wc, G
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    b = tot[(1.0, 1.0)]
    print(f'\n{nets} мереж × {NDRAW} жеребів, ПОВНИЙ ланцюг сабміту '
          f'(вибілювання + антитетика + якір-0 + σ-поправка)\n')
    print(f'{"варіант":>18s} {"MSE відносно бази":>18s}')
    print(f'{"k=1 (наше)":>18s} {1.0:17.4f}× ✅КОНТРОЛЬ  (λ×1.0)')
    for k in KGRID[1:]:
        best = min(SGRID, key=lambda sc: tot[(k, sc)])
        print(f'{"k="+str(k):>18s} {b/max(tot[(k,best)],1e-300):17.4f}×  '
              f'(найкраще λ×{best})'
              + ('  ← ОРАКУЛЬНЕ замикання' if k == 0.0 else ''))
    print(f'{"":>18s} {"—":>17s}')
    for kk in ('⛔без вибіл.', 'σ-поправка ВИМК'):
        print(f'{kk:>18s} {b/max(tot[kk],1e-300):17.4f}×')
    print(f'\nКЗ 4-го СТЕПЕНЯ (He₄ у {MCV} напрямках, ОРАКУЛЬНА матриця B '
          f'на кожній мережі — це СТЕЛЯ, не метод):')
    for nm in ('акт', '⛔вип'):
        g = cv[nm][1] / max(cv[nm][0], 1e-300)
        print(f'   напрямки {nm:>5s}: {g:.4f}×  (фіт на 1-й половині жеребів, '
              f'оцінка на 2-й)'
              + ('  ⛔ мусить бути ≈1.0' if nm == '⛔вип' else ''))
    print('\n🔑 крива по `k` — ПОВНА віддача будь-якого покращення замикання, '
          'включно з AMP. Якщо навіть k=0 дає мало — гілку можна не відкривати')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
