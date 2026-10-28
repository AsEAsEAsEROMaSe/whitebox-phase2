"""НАПРЯМОК ЧИ ТОЧНІСТЬ? — розсуд між `price31.py` (2.41×) і `mlmc_probe.py` (54×).

ЗВІДКИ (зауваження юзера, найточніше за дві сесії). Два МОЇ власні виміри
розходяться у 20 разів: ідеальне замикання на всіх 31 шарі = **2.41×**, а профіль
`ρ²(ℓ)` каже, що лінійна голова на активаціях шару 24 має `ρ²=0.981` ⟹ стеля
**54×**. Гіпотеза юзера: `ρ²` рахувався для голови з ОПТИМАЛЬНИМИ вагами МНК, а
наша схема несе поправку в «природних» координатах замикання ⟹ розрив тримає
**ВИБІР НАПРЯМКУ ПРОЄКЦІЇ**, а це ступінь свободи, якої нема в закритих гілках.
І окремо: як контрольна змінна `a_ℓ` лежить на тому самому прямому проході, тож
`γ=0`, і формула Гілза тут не та формула.

🎯 ОБИДВА РЕЗУЛЬТАТИ ЦІННІ. Рухається число до профілю — втрачений множник це
напрямок. Не рухається — профіль `ρ²` оракульний, і його треба ВИПРАВИТИ в
пам'яті, поки він не спокусив ще одну сесію.

ЩО РАХУЄМО. На ЗВІРЕНІЙ базі, для шару `ℓ` беремо СИРИЙ залишок
`r_ℓ = mean(a_ℓ) − m_ℓ` (без переносу, гачок `collect_means`) і питаємо, яку
частку похибки `e` він пояснює при РІЗНИХ дозволених напрямках:
  * **природний** — той єдиний вектор, який дає ланцюг воріт (`collect_anchors`);
    це рівно те, що робить наш μ-якір;
  * **ранг k = 1, 2, 4, 8** — вільна проєкція `r_ℓ` на топ-`k` головних напрямків
    із ПОЛОВИНИ жеребів, лінійна голова на другу половину.
Два джерела «відомого» середнього: **⚠️ОРАКУЛ** `E_іст[a_ℓ]` (верхня межа) і
**замикання** `m_замик` (те, що доступно).

І ТРЕТІЙ стовпець, що прямо локалізує розбіжність: та сама регресія, але цілю
береться не наша похибка `e`, а СИРА похибка Монте-Карло `mean(x_L) − істина`.
Якщо там `ρ²≈0.98`, а на `e` мало — відповідь «виграші КОНКУРУЮТЬ: `v` уже взяв».

КОНТРОЛІ
  * ✅ ТАВТОЛОГІЯ: фіт і оцінка на ТИХ САМИХ жеребах при великому `k` мусять
    дати майже нескінченний виграш (машинерія жива);
  * ⛔ НЕГ: `r_ℓ` із ПЕРЕСТАВЛЕНИХ жеребів (зсув `roll`, не `permutation`) —
    мусить дати ≈1.0;
  * 📊 сума · МЕДІАНА по мережах.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, W_)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import truth_layers31                                # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 128
K31 = 31
PROBE = [1, 4, 8, 16, 24, 28, 30]
RANKS = [1, 2, 4, 8]
RIDGE = 1e-6
LAM31 = tuple(LAM_S) + (0.0,) * (K31 - len(LAM_S))     # σ-поправка лише 1..8


def fit_eval(R, E, k, ridge=RIDGE):
    """Топ-`k` напрямків із ПЕРШОЇ половини жеребів, оцінка на ДРУГІЙ.

    🚨 БЕЗ центрування і БЕЗ вільного члена. Перша версія мала обидва — і вони
    поглинали ПОСТІЙНИЙ зсув `m_іст − m_замик`, через що стовпці «оракул» і
    «замикання» виходили побітово однакові (два різні варіанти з тим самим
    числом = вада стенда). Вільний член до того ж є ОРАКУЛЬНОЮ померéжевою
    поправкою на зміщення, якої в сабміті взяти нізвідки.
    """
    h = R.shape[0] // 2
    Rf, Ef, Rv, Ev = R[:h], E[:h], R[h:], E[h:]
    U = torch.linalg.svd(Rf, full_matrices=False)[2][:k].T         # (n_ℓ, k)
    Pf, Pv = Rf @ U, Rv @ U
    G = Pf.T @ Pf
    rg = ridge * float(torch.diagonal(G).mean().clamp_min(1e-300))
    B = torch.linalg.solve(G + rg * torch.eye(G.shape[0], dtype=R.dtype,
                                              device=R.device), Pf.T @ Ef)
    res = Ev - Pv @ B
    return float((Ev * Ev).sum()), float((res * res).sum())


def scalar_fit(A, E):
    """ПРИРОДНИЙ напрямок: те, що робить μ-якір — ОДИН скаляр на перенесений
    якір. Саме це, а не найкращий напрямок у просторі виходу."""
    h = A.shape[0] // 2
    Af, Ef, Av, Ev = A[:h], E[:h], A[h:], E[h:]
    beta = float((Af * Ef).sum()) / max(float((Af * Af).sum()), 1e-300)
    res = Ev - beta * Av
    return float((Ev * Ev).sum()), float((res * res).sum())


def main(nets=24, dev=DEV):
    torch.manual_seed(4242)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    keys = ([('оракул', 'прир.'), ('замик.', 'прир.')]
            + [(s, f'ранг{k}') for s in ('оракул', 'замик.') for k in RANKS]
            + [('оракул', '⛔НЕГ'), ('оракул', '✅тавтол.')])
    acc = {lay: {kk: [0.0, 0.0] for kk in keys} for lay in PROBE}
    per = {lay: {kk: [] for kk in keys} for lay in PROBE}
    raw = {lay: [0.0, 0.0] for lay in PROBE}          # ціль = СИРА похибка МК

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, K31, dev)
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            nl = wc[-1].shape[1]
            Rall = {lay: [] for lay in PROBE}
            Anat = {lay: {'оракул': [], 'замик.': []} for lay in PROBE}
            Es, Eraw = [], []

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                gg = torch.Generator(device=dev).manual_seed(int(sd))
                z = torch.randn(npair, W_, device=dev, generator=gg)
                pr, Ms = run_side1(wc, npair, s_cl, m_cl, dev, sd, z_ext=z,
                                   lam_s=LAM31, collect_means=True)
                Es.append(pr - T)
                for lay in PROBE:
                    Rall[lay].append(Ms[lay - 1])
                # природний напрямок = перенесений якір (те, що робить μ-якір)
                for src, mm in (('замик.', m_cl), ('оракул', TL['m'])):
                    _, A = run_side1(wc, npair, s_cl, mm, dev, sd, z_ext=z,
                                     lam_s=LAM31, collect_anchors=True)
                    for lay in PROBE:
                        Anat[lay][src].append(A[lay - 1])
                # СИРА оцінка Монте-Карло ТОГО САМОГО ланцюга (з вибілюванням):
                # це рівно `Ms[-1]`, тобто `pred + v`. Перша версія рахувала
                # окремий НЕвибілений прохід — інша вибірка, інший об'єкт.
                Eraw.append(Ms[-1] - T)

            E = torch.stack(Es)
            ER = torch.stack(Eraw)
            for lay in PROBE:
                Rl = torch.stack(Rall[lay])
                for src, mm in (('оракул', TL['m']), ('замик.', m_cl)):
                    R = Rl - mm[lay - 1].unsqueeze(0)
                    for k in RANKS:
                        a, b = fit_eval(R, E, k)
                        acc[lay][(src, f'ранг{k}')][0] += a
                        acc[lay][(src, f'ранг{k}')][1] += b
                        per[lay][(src, f'ранг{k}')].append(a / max(b, 1e-300))
                    An = torch.stack(Anat[lay][src])
                    a, b = scalar_fit(An, E)
                    acc[lay][(src, 'прир.')][0] += a
                    acc[lay][(src, 'прир.')][1] += b
                    per[lay][(src, 'прир.')].append(a / max(b, 1e-300))
                Ro = Rl - TL['m'][lay - 1].unsqueeze(0)
                a, b = fit_eval(torch.roll(Ro, 7, 0), E, 8)      # ⛔НЕГ
                acc[lay][('оракул', '⛔НЕГ')][0] += a
                acc[lay][('оракул', '⛔НЕГ')][1] += b
                per[lay][('оракул', '⛔НЕГ')].append(a / max(b, 1e-300))
                hh = Ro.shape[0] // 2                            # ✅ТАВТОЛОГІЯ
                a, b = fit_eval(torch.cat([Ro[:hh], Ro[:hh]]),
                                torch.cat([E[:hh], E[:hh]]), 16)
                acc[lay][('оракул', '✅тавтол.')][0] += a
                acc[lay][('оракул', '✅тавтол.')][1] += b
                per[lay][('оракул', '✅тавтол.')].append(a / max(b, 1e-300))
                a, b = fit_eval(Ro, ER, 8)                       # ціль = СИРА МК
                raw[lay][0] += a
                raw[lay][1] += b
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (nl={nl})', flush=True)

    print(f'\n{nets} мереж × {NDRAW} жеребів. Виграш = ‖e‖²/‖залишок‖² '
          f'(фіт на 1-й половині жеребів, оцінка на 2-й)\n')
    hdr = ['прир.'] + [f'ранг{k}' for k in RANKS]
    for src in ('оракул', 'замик.'):
        print(f'--- відоме середнє: {src} ---')
        print(f'{"шар":>5s} ' + ' '.join(f'{h:>9s}' for h in hdr))
        for lay in PROBE:
            row = []
            for h in hdr:
                a, b = acc[lay][(src, h)]
                row.append(f'{a / max(b, 1e-300):9.3f}')
            print(f'{lay:5d} ' + ' '.join(row))
        print()
    print('МЕДІАНИ ПО МЕРЕЖАХ (оракул):')
    print(f'{"шар":>5s} ' + ' '.join(f'{h:>9s}' for h in hdr))
    for lay in PROBE:
        print(f'{lay:5d} ' + ' '.join(
            f'{np.median(per[lay][("оракул", h)]):9.3f}' for h in hdr))
    print('\nКОНТРОЛІ (оракул, ранг 8/16):')
    print(f'{"шар":>5s} {"⛔НЕГ":>9s} {"✅тавтол.":>10s}   '
          f'{"ціль=СИРА МК":>13s}')
    for lay in PROBE:
        a, b = acc[lay][('оракул', '⛔НЕГ')]
        c, e = acc[lay][('оракул', '✅тавтол.')]
        g, h = raw[lay]
        print(f'{lay:5d} {a / max(b, 1e-300):9.3f} {c / max(e, 1e-300):10.1f}   '
              f'{g / max(h, 1e-300):13.3f}')
    print('\n🔑 читати так: якщо «ранг8» ПОМІТНО більший за «прир.» — втрачений '
          'множник це НАПРЯМОК, і 31 контрольна змінна лежить на прямому проході. '
          'Якщо вони близькі, а «ціль=СИРА МК» велика — виграші КОНКУРУЮТЬ: `v` '
          'уже забрав те, що обіцяв профіль ρ².')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
