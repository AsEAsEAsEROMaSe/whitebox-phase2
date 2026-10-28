"""ВАРІАЦІЙНЕ ЗАМИКАННЯ ВСЕРЕДИНІ **СПРАВЖНЬОГО** САБМІТУ (`sub72`, 4 канали).

════════ ЗВІДКИ ════════
`arena62/variational.py`: якщо в рекурсії замикання підставляти не той гаус,
що збігається з істиною ЗА МОМЕНТАМИ, а трохи ВУЖЧИЙ (`cp → α²·cp`), то
RMS-похибка замикання падає у **1.63–1.68×** з ОДНИМ спільним скаляром
`α ≈ 0.998`, і це ПЕРЕНОСИТЬСЯ на чужі мережі (на них навіть краще).
Напрямок `α<1` передбачено ДО прогону з важких хвостів преактивацій.

🚨 МІРЯЄМО НА `base72`, НЕ НА `base_real`. Урок відкликаного каналу `â²`:
`base_real` — порт УРІЗАНОГО `side1` (лише σ-канал), а відправлений `sub72`
має ЧОТИРИ родини (δs, δx̄, δE[x²], δмаски). Точніше замикання годує ВСІ
чотири, тож на урізаній базі виграш був би завищений.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
λ сабміту підігнані під СТАРУ похибку замикання. Канали віднімають
`(вибіркова статистика − замикання)`; коли замикання точніше, ця різниця
ближча до чистого шуму вибірки ⟹ ОПТИМАЛЬНИЙ λ РОСТЕ. Отже:
📣 наївна підстановка з РІДНИМИ λ дасть майже нуль (**1.00–1.02×**), а весь
   виграш, якщо він є, вимагатиме БІЛЬШИХ λ ⟹ дивимось скан масштабу `μ`.
📣 `s_cl` краще віддавати НЕ масштабованим: `α` калібрувалась під `m`, а
   `s_cl` — це прогноз істинної σ доактивацій, інша величина.
🟢 ЖИВИЙ при ≥1.03× на сумі та медіані одночасно (12% — поріг приладу).

🚨 ПЛОМБИ
  ✅ `α=1, μ=1` — це РІВНО діючий sub72, референс 1.000×;
  ⛔ КОНТРОЛЬ ЗНАКУ: `α>1` (ширший пакет) мусить бути ГІРШЕ — інакше це не
     варіаційний ефект, а довільна ручка;
  🚨 КРОС-ЦІЛЬ (дві незалежні істини) — шум істини інакше занижує різниці.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, _phi, _Phi                   # noqa: E402
from base72 import run_side1 as run72                              # noqa: E402
from tilt_wide import truth_last                                   # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 64
K = 8
FRM = 1
ALPHAS = (0.9960, 0.9970, 0.9980, 0.9990, 1.0000, 1.0020, 1.0050)
# 🚨 ПЕРША СПРОБА ПРОВАЛИЛАСЬ (0.275×) І ПОКАЗАЛА ЧОМУ: `α` калібрувалась
# під `m`, а підставлялась одразу в УСІ ЧОТИРИ опори (`s`,`m`,`q`,`f`).
# Канали віднімають РІЗНИЦЮ `(вибіркова − замикання)`, яка сама ~0.5%, тож
# зсув опори на 0.4% від АБСОЛЮТНОГО значення — це зсув самого каналу на
# десятки відсотків. ⟹ підставляємо лише туди, де калібрували, і скануємо
# множник САМЕ того λ, чию опору змінили.
# 🔑 ДРУГА СПРОБА показала, ЧОМУ мало: `m_cl` входить у прогноз ЛИШЕ через
# `λ_a = −0.007` (✅пломба: при μ=0 усі α дали ОДНЕ число 1.009×), тобто
# канал μ-якоря майже вимкнений. А опори каналів `q` і `g` мають λ у 5–6×
# більші (0.042, 0.027) ⟹ пробуємо СВІЙ `α` на КОЖНУ опору окремо.
WHERE = ('m', 's', 'q', 'g')
MUS = (1.0, 2.0, 4.0)

# ── коефіцієнти ДІЮЧОГО САБМІТУ `estimator_sub72.py` ──────────────────
S72 = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
       0.0088285, 0.0007922, 0.0015044, 0.0001062)
A72 = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
       -0.0025822, -0.0030778, 0.0006822, 0.0068061)
Q72 = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
       0.0189783, 0.0250559, 0.0340802, 0.0415433)
G72 = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
       -0.0146129, -0.0100612, 0.0167223, 0.0168109)


def closure_var(wc, ksig, dev, alpha=1.0, s_raw=True):
    """`closure_parts` з ВАРІАЦІЙНОЮ шириною: `cp → α²·cp` з кроку `FRM`.

    s_raw=True — назовні віддаємо НЕ масштабоване `s` (прогноз істинної σ
    доактивацій), а `α` лишається внутрішнім трюком рекурсії для `m`.
    """
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    s_o, m_o, f_o, q_o = [], [], [], []
    for i in range(ksig + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        a2 = (alpha * alpha) if i >= FRM else 1.0
        cpe = cp * a2
        se = cpe.diagonal().clamp_min(1e-20).sqrt()
        if i >= 1:
            s_o.append(s if s_raw else se)
        mp = m @ w
        t = mp / se
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + se * ph
        C = cpe * torch.outer(Ph, Ph) \
            + (0.5 * cpe * cpe) * torch.outer(ph / se, ph / se)
        d = (mp * mp + se * se) * Ph + mp * se * ph - m * m
        C = C + torch.diag(d - C.diagonal())
        if i >= 1:
            m_o.append(m)
            f_o.append(Ph)
            q_o.append(d + m * m)
    return s_o, m_o, f_o, q_o


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: рідні λ дадуть 1.00–1.02×; виграш (якщо є) '
          f'вимагає БІЛЬШИХ λ')
    print(f'🟢 ЖИВИЙ при ≥1.03× на сумі І медіані.  '
          f'мереж {nets} × {ndraw} жеребів, КРОС-ЦІЛЬ\n')

    keys = [(a, mu, wh) for a in ALPHAS for mu in MUS for wh in WHERE]
    acc = {k: [] for k in keys}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)
            # 🚨 s_raw=False — інакше `α` взагалі не доходить до `s_cl`
            # і режим 's' мовчки міряв би базу проти бази
            CLS = {a: closure_var(wc, K, dev, a, s_raw=False)
                   for a in ALPHAS}
            BASE = CLS[1.0]
            run = {k: 0.0 for k in keys}
            for d in range(ndraw):
                sd = 77_000_000 + 733 * n + d
                for (a, mu, wh) in keys:
                    # варіаційна опора ЛИШЕ в одному каналі; `μ` множить
                    # λ САМЕ того каналу, чию опору змінили
                    s_cl, m_cl, f_cl, q_cl = BASE
                    ls, la, lq, lg = S72, A72, Q72, G72
                    if wh == 'm':
                        m_cl = CLS[a][1]
                        la = tuple(mu * v for v in A72)
                    elif wh == 's':
                        s_cl = CLS[a][0]
                        ls = tuple(mu * v for v in S72)
                    elif wh == 'q':
                        q_cl = CLS[a][3]
                        lq = tuple(mu * v for v in Q72)
                    else:
                        f_cl = CLS[a][2]
                        lg = tuple(mu * v for v in G72)
                    p, _ = run72(wc, npair, s_cl, m_cl, dev, sd,
                                 lam_s=ls, lam_a=la, lam_q=lq, lam_g=lg,
                                 q_ext=q_cl, p_ext=f_cl)
                    run[(a, mu, wh)] += float(((p - T1) * (p - T2)).sum())
            for k in keys:
                acc[k].append(run[k] / ndraw)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    ref = np.array(acc[(1.0, 1.0, 'm')])
    print(f'\n{nets} мереж × {ndraw} жеребів. КРОС-ЦІЛЬ. '
          f'✅референс = діючий sub72 (α=1, μ=1)\n')
    print(f'{"α":>8s} {"μ(λ_a)":>7s} {"куди":>5s} {"СУМА":>9s} '
          f'{"МЕДІАНА":>9s} {"краще":>7s}')
    print('-' * 52)
    best = None
    for wh in WHERE:
        for a in ALPHAS:
            for mu in MUS:
                v = np.array(acc[(a, mu, wh)])
                s = ref.sum() / v.sum()
                md = float(np.median(ref / v))
                frac = float((v < ref).mean())
                mark = ''
                if a == 1.0 and mu == 1.0 and wh == 'm':
                    mark = ' ✅пломба'
                if best is None or s > best[0]:
                    best = (s, md, a, mu, wh, frac)
                print(f'{a:8.4f} {mu:7.1f} {wh:>5s} {s:8.3f}× '
                      f'{md:8.3f}× {frac * 100:6.0f}%{mark}')
        print('-' * 52)

    s, md, a, mu, sr, frac = best
    up = [k for k in ALPHAS if k > 1.0]
    ctl = max(ref.sum() / np.array(acc[(k, mu, sr)]).sum() for k in up)
    print(f'\n⛔КОНТРОЛЬ ЗНАКУ: найкраще серед α>1 (ширший пакет) = '
          f'{ctl:.3f}× (мусить бути гірше за {s:.3f}×)')
    print(f'\n🔑 НАЙКРАЩЕ: α={a:.4f}, μ={mu:.1f}, s_cl={"сирий" if sr else "α·s"}'
          f'  ⟹ сума {s:.3f}×, медіана {md:.3f}×, краще на {frac*100:.0f}% мереж')
    print(f'🔑 ВИРОК: {"ЖИВИЙ" if (s >= 1.03 and md >= 1.03) else "МЕРТВИЙ"} '
          f'(🟢гейт ≥1.03× на сумі І медіані)')
    print(f'📊 БАЛ: 2.05e-07 → {2.05e-07 / max(s, 1e-9):.3e} '
          f'(якщо переноситься на грейдер)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
