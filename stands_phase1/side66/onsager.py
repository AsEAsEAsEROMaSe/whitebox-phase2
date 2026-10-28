"""ОНСАГЕР: наш перенос — це НАЇВНЕ СЕРЕДНЄ ПОЛЕ. Скільки коштує факторизація?

════════ ФІЗИКА, З ЯКОЇ ЦЕ ВЗЯТО ════════
Мережа = спінове скло Шеррінгтона–Кіркпатріка: iid гаусові зв'язки, кожен
нейрон бачить суму 256 випадкових внесків ⟹ ЦГТ ⟹ гаусове локальне поле.
Наше замикання Мелера K=2 — це рівняння середнього поля.
У сабміті похибка якоря переноситься так:

    v ← (v @ W_i) * gate_i,      gate_i = ЧАСТКА активних нейронів

Це **ДОБУТОК СЕРЕДНІХ** затворів. Правильно — **СЕРЕДНЄ ДОБУТКІВ**:
затвори сусідніх шарів КОРЕЛЮЮТЬ (нейрон, що спрацював, змінює поле
наступного, а той — назад). У фізиці рівно цю різницю виправляє
**реакційний член Онсагера** (поправка Таулесса–Андерсона–Палмера).

🔑 ЧОМУ ЦЕ НЕ ЗАКРИТИЙ КЛАС.
  * ⛔клас Стайна (`ρ²=0.002`) — це БЕЗКОШТОВНІ КЗ для ДИСПЕРСІЇ. Тут ми
    правимо ЗМІЩЕННЯ переносу — інша величина.
  * ⛔«затвор НЕ горло» (С76, повний затвор 1.0059×) міряло ТОЧНІСТЬ оцінки
    `gate` (усі рядки замість 1024). Факторизацію не міряв ніхто: навіть із
    ІДЕАЛЬНО точним `gate` добуток середніх ≠ середнє добутків.
  * 🗺️пам'ять: «2/3 похибки замикання робить ПЕРЕНОС» ⟹ ціль велика.

════════ ЩО МІРЯЄМО (усе з ОДНОГО жеребу, порівняння чисте) ════════
  FACT    — база sub87: `gate` з 1024 рядків          (наївне середнє поле)
  FULL    — `gate` з УСІХ рядків     ✅ПЛОМБА С76 ≈1.006×
  EXACT   — ПОСЕМПЛЬНИЙ перенос: несемо матрицю `D` (2n × width),
            `D ← (D @ W_i) * 1[p_i>0]`, далі `v = D.mean(0)`.
            Це `E[J]v₀` БЕЗ факторизації — стеля класу.
  ONS(α)  — реакційний член у найдешевшій формі: один скаляр на крок,
            `gate → α·gate`. Нуль параметрів у сабміті (α зашивається).
  ONS31   — той самий α, але свій на кожен шар (31 параметр, LOO).

🚨 EXACT — ЦЕ ДІАГНОСТИКА СТЕЛІ, НЕ ГОТОВИЙ ХІД: посемпльний перенос
   подвоює найдорожчий цикл (85% флопів) ⟹ у сабміт не влізе. Він потрібен,
   щоб знати, чи варто шукати дешеву форму. Дешева форма — це ONS.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 FULL   ≈ 1.005-1.010× (відтворення С76)
📣 EXACT  = 1.05-1.25×  — факторизація коштує, але перенос псує не лише вона
📣 ONS(α) = 1.00-1.05×, оптимальний α трохи БІЛЬШИЙ за 1
           (кореляція затворів ПОЗИТИВНА: активний нейрон піднімає поле
            наступного ⟹ добуток середніх ЗАНИЖУЄ перенос)
🟢 ГЕЙТИ: ONS(α) ≥1.02× ⟹ прямо в бал (множник на підлозі, 0 флопів).
          EXACT ≥1.10× ⟹ варто шукати дешеву апроксимацію.
          EXACT <1.05× ⟹ факторизація не горло, гілку закрити числом.

🚨 ПЛОМБИ
  ✅ ONS(α=1.0) мусить дати РІВНО 1.0000× (це і є FACT);
  ✅ EXACT з маскою, заміненою на середній затвор, мусить дати РІВНО FACT
     (той самий код, дві маски) — друкуємо як SEAL;
  ⛔ НЕГ: EXACT з масками СУСІДНЬОЇ мережі ⟹ мусить бути ГІРШЕ за FACT.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N)
from jitprof import truth_means                                    # noqa: E402
from jitanim import bank_net, LAM_S, LAM_A, LAM_Q, LAM_G           # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
KSIG = 8
# 🚨 множник іде в 31-й степінь ⟹ сітка мусить бути в масштабі 1/31 ≈ 0.003,
#    інакше вершина не роздільна (перша версія мала крок 0.02 і бачила лише схил)
ALPHAS = (0.9900, 0.9950, 0.9975, 0.9990, 1.0000,
          1.0010, 1.0025, 1.0050, 1.0100)


def _phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQRT_2PI


def run(wc, npair, cl, dev, seed, mask_src=None):
    """Один жереб. Повертає все, що треба для всіх варіантів переносу.

    mask_src — ⛔НЕГ: список масок ЧУЖОЇ мережі замість власних.
    Повертає dict:
      xbar   — вибіркове середнє виходу
      v0     — стартова похибка якоря (шар 0)
      g1024  — список `gate` з 1024 рядків (як у сабміті)
      gfull  — список `gate` з усіх рядків
      chan   — список λ-внесків шару (нуль поза 1..ksig)
      vex    — ПОСЕМПЛЬНИЙ перенос, уже усереднений (EXACT)
      vseal  — той самий код, але маска = середній затвор (✅SEAL = FACT)
    """
    s_cl, m_cl, P_cl, q_cl = cl
    d = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    y = z @ (wh @ wc[0])
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    denom = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v0 = x.double().sum(0) / denom - sg0 * INV_SQRT_2PI
    ks = min(KSIG, d - 1)

    # ПОСЕМПЛЬНИЙ перенос: кожен рядок несе СВОЮ копію v₀ крізь СВОЇ затвори.
    D = v0.unsqueeze(0).expand(int(denom), -1).contiguous()
    S = D.clone()                                   # ✅SEAL-двійник
    g1024, gfull, chan, masks = [], [], [], []
    for i in range(1, d):
        p = x @ wc[i]
        mk = (p > 0)
        gt_full = mk.to(p.dtype).sum(0).double() / denom
        gt_1024 = mk[:GATE_N].to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        use = mk if mask_src is None else mask_src[i - 1]
        D = (D @ wd) * use.double()
        S = (S @ wd) * gt_1024                      # ✅ та сама формула,
        #                                              маска → середній затвор
        masks.append(mk)
        x = torch.relu(p)
        ch = torch.zeros(wd.shape[1], dtype=torch.float64, device=dev)
        if i <= ks:
            mh = p.double().sum(0) / denom
            sh = (p.double().pow(2).sum(0) / denom - mh * mh).clamp_min(
                1e-30).sqrt()
            xm = x.double().sum(0) / denom
            xq = x.double().pow(2).sum(0) / denom
            ch = ((LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                  + LAM_A[i - 1] * (xm - m_cl[i - 1])
                  + LAM_Q[i - 1] * (xq - q_cl[i - 1])
                  + LAM_G[i - 1] * (gt_1024 - P_cl[i - 1]))
            D = D + ch.unsqueeze(0)
            S = S + ch.unsqueeze(0)
        g1024.append(gt_1024)
        gfull.append(gt_full)
        chan.append(ch)
    return {
        'xbar': x.double().sum(0) / denom,
        'v0': v0, 'g1024': g1024, 'gfull': gfull, 'chan': chan,
        'vex': D.sum(0) / denom, 'vseal': S.sum(0) / denom,
        'masks': masks,
    }


def main(nets=24, dev=DEV, ndraw=NDRAW):
    print('📣 ПЕРЕДБАЧЕНО: FULL 1.005-1.010× · EXACT 1.05-1.25× · '
          'ONS(α) 1.00-1.05×, α>1')
    print('🟢 ГЕЙТИ: ONS ≥1.02× (прямо в бал) · EXACT ≥1.10× '
          '(шукати дешеву форму)\n', flush=True)

    acc = {k: [] for k in ('FACT', 'FULL', 'EXACT', 'SEAL', 'NEG')}
    accA = {a: [] for a in ALPHAS}
    # для ONS31 копимо (похибка, матриця чутливостей) — фітуємо LOO по мережах
    per_net = []
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            mt, _ = truth_means(wc, dev, 310_000 + 877 * n)
            m31 = mt[d - 1]
            cl = closure_parts(wc, min(KSIG, d - 1), dev, extras=True)
            wd = [wc[i].double() for i in range(d)]
            prev_masks = None
            errs, dvs = [], []
            for k in range(ndraw):
                R = run(wc, npair, cl, dev, 93_000_000 + 977 * n + k)
                xb = R['xbar']

                def vec(gates, alpha=1.0, pl=None):
                    v = R['v0']
                    for i in range(len(gates)):
                        a = alpha if pl is None else pl[i]
                        v = (v @ wd[i + 1]) * (gates[i] * a) + R['chan'][i]
                    return v

                vf = vec(R['g1024'])
                acc['FACT'].append(float((xb - vf - m31).pow(2).sum()))
                acc['FULL'].append(
                    float((xb - vec(R['gfull']) - m31).pow(2).sum()))
                acc['EXACT'].append(
                    float((xb - R['vex'] - m31).pow(2).sum()))
                acc['SEAL'].append(
                    float((xb - R['vseal'] - m31).pow(2).sum()))
                for a in ALPHAS:
                    accA[a].append(
                        float((xb - vec(R['g1024'], a) - m31).pow(2).sum()))
                # 🔬 РОЗКЛАД: реакційний член систематичний чи шумний?
                #    err = похибка бази · dv = поправка Онсагера з 1 жеребу
                errs.append((xb - vf - m31).cpu().numpy())
                dvs.append((R['vex'] - vf).cpu().numpy())
                # ⛔НЕГ: посемпльний перенос по ЧУЖИХ масках (з попереднього
                #        жеребу цієї ж мережі — форми збігаються ⟹ чесно)
                if prev_masks is not None:
                    Rn = run(wc, npair, cl, dev, 93_000_000 + 977 * n + k,
                             mask_src=prev_masks)
                    acc['NEG'].append(
                        float((xb - Rn['vex'] - m31).pow(2).sum()))
                prev_masks = R['masks']
            E = np.array(errs)
            Dv = np.array(dvs)
            eb, db = E.mean(0), Dv.mean(0)              # зміщення по жеребах
            per_net.append((
                float((eb ** 2).sum()),                 # ‖зміщення FACT‖²
                float(((eb - db) ** 2).sum()),          # ‖зміщення EXACT‖²
                float(((E - eb) ** 2).sum(1).mean()),   # дисперсія FACT
                float(((E - Dv - eb + db) ** 2).sum(1).mean()))) # дисп EXACT
        del W, wc, wd
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    base = float(np.mean(acc['FACT']))
    print(f'\n{nets} мереж × {ndraw} жеребів · база FACT = {base:.5e}')
    print(f'{"варіант":>10} {"MSE":>13} {"× проти FACT":>14}')
    for k in ('FACT', 'FULL', 'EXACT', 'SEAL', 'NEG'):
        if not acc[k]:
            continue
        v = float(np.mean(acc[k]))
        print(f'{k:>10} {v:>13.5e} {base/v:>14.4f}')
    sealr = base / float(np.mean(acc['SEAL']))
    print(f'\n✅SEAL (маска→середній затвор) = {sealr:.6f}× '
          f'(мусить бути 1.000000)')

    print(f'\nОНСАГЕР, скаляр на затвор:')
    bestA = (0.0, None)
    for a in ALPHAS:
        r = base / float(np.mean(accA[a]))
        if r > bestA[0]:
            bestA = (r, a)
        tag = '  ✅ПЛОМБА = FACT' if a == 1.00 else ''
        print(f'   α={a:<6} {r:>8.4f}×{tag}')
    print(f'   🔑 найкраще α={bestA[1]} ⟹ {bestA[0]:.4f}× (🟢гейт 1.02)')

    P = np.array(per_net)
    bF, bE, vF, vE = P[:, 0].mean(), P[:, 1].mean(), P[:, 2].mean(), P[:, 3].mean()
    print(f'\n🔬 РОЗКЛАД MSE (реакційний член: сигнал чи шум?)')
    print(f'{"":>18} {"ЗМІЩЕННЯ²":>13} {"ДИСПЕРСІЯ":>13}')
    print(f'{"FACT (наївне)":>18} {bF:>13.5e} {vF:>13.5e}')
    print(f'{"EXACT (Онсагер)":>18} {bE:>13.5e} {vE:>13.5e}')
    print(f'   зміщення {bF/max(bE,1e-300):.4f}× · дисперсія '
          f'{vF/max(vE,1e-300):.4f}×')
    if bE < bF and vE > vF:
        print('   🔑 реакційний член РЕАЛЬНИЙ (зміщення падає), але тоне в '
              'дисперсії ⟹ шукати ЗГЛАДЖЕНУ дешеву форму')
    elif bE >= bF:
        print('   ⛔ реакційний член НЕ зменшує зміщення ⟹ гілку закрито '
              'за механізмом, не за ціною')

    ex = base / float(np.mean(acc['EXACT']))
    print(f'\n🔑 ВИРОК')
    print(f'   ONS(α)  {bestA[0]:.4f}×  ⟹ '
          f'{"🟢 ЖИВЕ, прямо в бал" if bestA[0] >= 1.02 else "⛔ нижче гейта"}')
    print(f'   EXACT   {ex:.4f}×  ⟹ '
          f'{"🟢 факторизація КОШТУЄ, шукати дешеву форму" if ex >= 1.10 else "⛔ факторизація НЕ горло"}')
    print(f'📊 БАЗОВИЙ БАЛ 1.954e-07 · множник на підлозі ⟹ виграш у MSE '
          f'іде в бал один до одного')
    print(f'   ONS(α) дав би {1.954e-07 / max(bestA[0], 1e-9):.4e}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
