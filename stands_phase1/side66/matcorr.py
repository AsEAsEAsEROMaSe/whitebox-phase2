"""ЧИ КОРЕЛЮЄ ХОЧ ЯКАСЬ МАТ-КОНСТРУКЦІЯ З НАШОЮ ПОХИБКОЮ? (ціну ІГНОРУЄМО)

Питання юзера: «чи є кореляція моменту максимальної точності з якимись мат
виведеними конструкціями? ціну поки не враховуй, просто виявити чи є».

Тому тут свідомо дорогі речі: замикання на ВСІ 31 шар, градієнти (зайвий
прохід), розбиття на фолди. Мета — НЕ канал у сабміт, а відповідь «бачить
хоч щось чи ні».

════════ ЩО САМЕ ПЕРЕВІРЯЄМО ════════
ЦІЛІ (обидві — ЗНАКОВІ, бо лише знак віднімається):
  `yP` = ⟨x̄₃₁ − m₃₁, u₁⟩        — похибка ЧИСТОГО середнього;
  `yO` = ⟨наш sub87 − m₃₁, u₁⟩   — що ЛИШИЛОСЬ після нашої поправки.

КОНСТРУКЦІЇ (усі рахуються з жеребу, істина НЕ використовується):
  T<l>  перенесений залишок замикання шару l: ⟨transport(x̄_l − m_cl,l), u₁⟩
        — та сама родина, що в сабміті, але до БУДЬ-ЯКОЇ глибини;
  D31   ПРЯМИЙ залишок на виході: ⟨x̄₃₁ − m_cl,31, u₁⟩ (без переносу);
  Q31   залишок ДРУГОГО моменту на виході: ⟨q̄₃₁ − q_cl,31, u₁⟩;
  ASYM  антитетична асиметрія: ⟨сер(relu(y)) − сер(relu(−y)), u₁⟩;
  FOLD  розбіжність половин вибірки: ⟨x̄_A − x̄_B, u₁⟩;
  LDET  Σlog власних чисел емпіричної коваріації входу (наскільки жереб
        нетиповий ДО вибілювання);
  RAD   сер‖z‖ − E[χ₂₅₆] (радіальна аномалія жеребу);
  SKEW  вибіркова скошеність проєкції виходу;
  STEIN тотожність Стайна: сер(Σᵢzᵢ·F) − сер(Σᵢ∂F/∂zᵢ), обидва боки рівні
        В СЕРЕДНЬОМУ ⟹ різниця = чистий шум ЦЬОГО жеребу.

🚨 ДВА ВИМІРИ НА КОЖНУ, І ЦЕ ПРИНЦИПОВО:
  «з вільним членом» — скільки СИГНАЛУ (стала похибка замикання знімається
     задарма, у сабміті так НЕ можна);
  «без вільного члена» — скільки МОЖНА ВЗЯТИ насправді.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 T<l> з вільним членом росте з глибиною до ~0.98 (це вже відома драбина),
   але БЕЗ нього має мати ГОРБ близько l=2..8 на рівні 0.03-0.05 і далі
   спадати, бо зміщення замикання росте швидше за сигнал.
📣 D31, Q31 — те саме, що T31, тобто з вільним членом багато, без нього ~0.
📣 ASYM, FOLD, LDET, RAD, SKEW, STEIN — усі ≈0 на ОБОХ цілях: це або
   відомі мертві канали, або чистий шум без зв'язку зі знаком.
🟢 ЖИВЕ, якщо якась конструкція дає БЕЗ вільного члена R² ≥ 0.10 на `yO`.
🚨 ПЛОМБИ: ⛔ROLL на кожній · ✅T31 з вільним членом ≈ 1.0 (тавтологія).
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N)
from jitprof import truth_means, transport, r2_oos                 # noqa: E402
from jitanim import bank_net, _run_sub87, LAM_S                    # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 256
T_L = (1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 31)
# E[‖z‖] для χ з 256 ст.св. — точно, через lgamma
E_CHI = float(np.exp(0.5 * np.log(2.0) + torch.special.gammaln(
    torch.tensor((W_ + 1) / 2.0)).item()
    - torch.special.gammaln(torch.tensor(W_ / 2.0)).item()))


def draw_feats(wc, npair, dev, seed, m_cl, q_cl, mt, u1):
    """Один жереб: усі конструкції + обидві цілі."""
    d = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x0 = torch.cat([z, -z], 0)
    cov = (x0.T @ x0) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xw = x0 @ wh
    denom = float(2 * npair)

    y = xw[:npair] @ wc[0]
    a = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    xb, gates, masks = [a.double().mean(0)], [None], [(a > 0).float()]
    for i in range(1, d):
        p = a @ wc[i]
        gates.append((p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N)
        a = torch.relu(p)
        masks.append((a > 0).float())
        xb.append(a.double().mean(0))
    q31 = a.double().pow(2).mean(0)

    f = {}
    for l in T_L:
        if l <= d - 1:
            ref = m_cl[l - 1].double() if l >= 1 else None
            f[f'T{l}'] = float(transport(xb[l] - ref, wc, gates,
                                         l, d - 1) @ u1)
    f['D31'] = float((xb[d - 1] - m_cl[d - 2].double()) @ u1)
    f['Q31'] = float((q31 - q_cl[d - 2].double()) @ u1)
    h = npair
    f['ASYM'] = float((a[:h].double().mean(0) - a[h:].double().mean(0)) @ u1)
    f['FOLD'] = float((a[:h // 2].double().mean(0)
                       - a[h:h + h // 2].double().mean(0)) @ u1)
    f['LDET'] = float(torch.log(ev.clamp_min(1e-12)).sum())
    f['RAD'] = float(xw.double().norm(dim=1).mean()) - E_CHI
    pr = (a.double() @ u1)
    f['SKEW'] = float(((pr - pr.mean()) ** 3).mean()
                      / max(float(pr.std()) ** 3, 1e-30))
    # STEIN: E[Σᵢ zᵢ F] = E[Σᵢ ∂F/∂zᵢ]; градієнт точний через маски relu
    gv = u1.float().unsqueeze(0).expand(a.shape[0], -1)
    for i in range(d - 1, -1, -1):
        gv = (gv * masks[i]) @ wc[i].T
    F = (a.double() @ u1)
    zs = xw.double().sum(1)
    f['STEIN'] = float((zs * F).mean() - gv.double().sum(1).mean())

    yP = float((xb[d - 1] - mt[d - 1]) @ u1)
    return f, yP


def main(nets=12, dev=DEV, ndraw=NDRAW):
    print('📣 ПЕРЕДБАЧЕНО: T<l> без вільного члена — горб 0.03-0.05 на l=2..8, '
          'решта ≈0')
    print('🟢 ЖИВЕ при R²(без вільн.чл.) ≥ 0.10 на цілі `yO`\n', flush=True)
    keys = None
    RI, RN, RR = {}, {}, {}
    RIO, RNO = {}, {}
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            mt, _ = truth_means(wc, dev, 310_000 + 877 * n)
            s_cl, m_cl, P_cl, q_cl = closure_parts(wc, d - 1, dev, extras=True)
            cl8 = (s_cl[:8], m_cl[:8], P_cl[:8], q_cl[:8])
            m31 = mt[d - 1]
            u1 = (m31 / m31.norm())          # напрямок відповіді
            FT, YP, YO = [], [], []
            for k in range(ndraw):
                sd = 61_000_000 + 977 * n + k
                f, yP = draw_feats(wc, npair, dev, sd, m_cl, q_cl, mt, u1)
                pr = _run_sub87(wc, npair, cl8, dev, sd)
                FT.append(f)
                YP.append(yP)
                YO.append(float((pr - m31) @ u1))
            keys = list(FT[0].keys())
            # 🚨🚨 НЕ СТАНДАРТИЗУВАТИ! Центрування віднімає СЕРЕДНЄ ознаки,
            #    тобто протягує вільний член туди, де я його заборонив, і
            #    «без вільн.чл.» ставало тим самим, що «з ним» (перша версія
            #    дала фальшиві 0.97 замість справжніх ~0.03). Масштаб для
            #    одноознакової регресії й так поглинається коефіцієнтом.
            X = np.array([[t[k] for k in keys] for t in FT])
            yP = np.array(YP)
            yO = np.array(YO)
            for j, k in enumerate(keys):
                c = X[:, j:j + 1]
                RI.setdefault(k, []).append(r2_oos(c, yP))
                RN.setdefault(k, []).append(r2_oos(c, yP, intercept=False))
                RIO.setdefault(k, []).append(r2_oos(c, yO))
                RNO.setdefault(k, []).append(r2_oos(c, yO, intercept=False))
                RR.setdefault(k, []).append(
                    r2_oos(np.roll(c, 1, axis=0), yP))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж × {ndraw} жеребів, npair={npair}\n')
    print('ЦІЛЬ `yP` = похибка ЧИСТОГО середнього · '
          '`yO` = ЗАЛИШОК після sub87')
    print(f'{"конструкція":>12} {"yP з в.ч.":>10} {"yP без":>8} '
          f'{"yO з в.ч.":>10} {"yO без":>8} {"⛔ROLL":>8}')
    rows = sorted(keys, key=lambda k: -np.mean(RNO[k]))
    for k in rows:
        taut = ' ✅тавтологія' if k == f'T{31}' else ''
        print(f'{k:>12} {np.mean(RI[k]):>10.4f} {np.mean(RN[k]):>8.4f} '
              f'{np.mean(RIO[k]):>10.4f} {np.mean(RNO[k]):>8.4f} '
              f'{np.mean(RR[k]):>8.4f}{taut}')
    best = max(np.mean(RNO[k]) for k in keys if not k.startswith('T3'))
    bk = [k for k in keys if not k.startswith('T3')
          and np.mean(RNO[k]) == best][0]
    print(f'\n🔑 НАЙКРАЩА КОНСТРУКЦІЯ БЕЗ ВІЛЬНОГО ЧЛЕНА на `yO`: '
          f'{bk} = {best:.4f}')
    print(f'   ВИРОК: {"ЖИВЕ" if best >= 0.10 else "МЕРТВО"} '
          f'(🟢гейт 0.10)')
    print('📊 БАЛ: 1.954e-07 (діагностика, ціна свідомо ігнорується)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 12)
