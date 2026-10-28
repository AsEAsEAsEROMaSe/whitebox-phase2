"""ПРАЙСИНГ ГЛИБИНИ: відношення ЗМІЩЕННЯ/ШУМ у кожній родині на кожному шарі.

════════ ЗВІДКИ ПИТАННЯ ════════
`biassplit.py` (С75): з чотирьох родин `λ` живе РІВНО ОДНА — `Q` (**1.20×**),
а `S`,`A`,`G` дають 0.999/1.003/1.006×. `jitprof.py` (паралельна лінія):
зміщення замикання переганяє шум вибірки **рівно на шарі 2**, на шарі 8 —
у **2.37×**, і це пояснює мізерні `λ≈0.005` та ⛔провал `KSIG` 8→24.
🔑 Але тоді ЧОМУ `Q` працює? Значить, у різних родин РІЗНЕ відношення, і саме
воно, а не глибина сама по собі, вирішує, доки родину можна тягнути.

════════ ЩО МІРЯЄТЬСЯ І ЧОМУ ЦЕ ДЕШЕВО ════════
Кожна ознака має вигляд `f = (вибіркова статистика) − (замикання)`. Отже
    E_жереб[f] = (істинне значення) − (замикання) = ПОХИБКА ЗАМИКАННЯ
    ст.в._жереб[f] = ШУМ ВИБІРКИ цієї статистики
🔑 **ІСТИНА ТУТ НЕ ПОТРІБНА ВЗАГАЛІ** — середнє по жеребах саме і є оцінкою
істини. Тож вимір коштує лише прогонів, без жодного `truth_*`.

📐 Оцінка незміщена (та сама формула, що в `biassplit`):
    ‖b‖² = (D·‖сер_d f‖² − сер_d‖f‖²) / (D−1),   шум² = сер_d‖f−сер‖²
Відношення `r = ‖b‖/шум` і є ціною: `r≪1` — ознака чиста; `r≫1` — ознака є
майже стала й лише вливає зміщення, і регресія ОБОВ'ЯЗКОВО зіб'є `λ` до нуля.

════════ ЩО ЦЕ ВИРІШУЄ ════════
Шар, де `r` перетинає 1 для НАЙКРАЩОЇ родини, і є максимально корисним `KSIG`.
Дізнатись це можна ДО того, як платити за рекурсію `C` на глибині
(0.025% бюджету/шар при запасі під підлогою 0.03%).

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 `A` (δсереднього) на шарі 8: `r ≈ 2.4` (незалежна звірка з `jitprof`);
📣 `Q` мусить мати НАЙМЕНШЕ `r` — інакше не пояснити, чому саме вона працює;
📣 перетин `r=1` для найкращої родини — на шарі **≤12** ⟹ напрямок «читати
   глибше» здебільшого мертвий. Якщо перетин ≥16 — напрямок ЖИВИЙ.

🚨 ПЛОМБИ
  ✅ ‖b‖² ≥ 0 у кожній клітинці (незміщена оцінка може піти в мінус лише на шум);
  ✅ оцінка на ДВОХ НЕЗАЛЕЖНИХ половинах жеребів мусить збігтися (друкується);
  ✅ `r` для `S` на шарі 1 ≈ 0: після вибілювання+антитетики `ŝ₁` збігається з
     замиканням ТОТОЖНО (обидва = `diag(W₁ᵀW₁)^½`) ⟹ ознака мертва за побудовою.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (cut_and_budget, closure_parts, W_, GATE_N,        # noqa: E402
                    INV_SQRT_2PI, _phi)

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
KCL = 28
FAM = ('S: δs', 'A: δE[x]', 'Q: δE[x²]', 'G: δмаска')


def feats(wc, npair, cl, dev, seed, kcl):
    """Один жереб: усі 4 ознаки на шарах 1..kcl. Без істини, без поправок."""
    s_cl, m_cl, f_cl, q_cl = cl
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cov = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xa = torch.relu(x @ (wh @ wc[0]))
    out = []
    for i in range(1, min(kcl, depth - 1) + 1):
        p = xa @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        pd = p.double()
        mh = pd.mean(0)
        sh = (pd.pow(2).mean(0) - mh * mh).clamp_min(1e-30).sqrt()
        xa = torch.relu(p)
        xd = xa.double()
        out.append((sh - s_cl[i - 1],
                    xd.mean(0) - m_cl[i - 1],
                    xd.pow(2).mean(0) - q_cl[i - 1],
                    gate - f_cl[i - 1]))
    # решта шарів — лише прохід, ознак нема
    for i in range(min(kcl, depth - 1) + 1, depth):
        xa = torch.relu(xa @ wc[i])
    return out


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: `A` на шарі 8 → r≈2.4 (звірка з jitprof) · `Q` — '
          f'НАЙМЕНШЕ `r` · перетин r=1 на шарі ≤12 ⟹ «глибше» МЕРТВЕ')
    print(f'🟢 НАПРЯМОК ЖИВИЙ, якщо перетин r=1 у найкращої родини ≥16')
    print(f'   {nets} мереж × {ndraw} жеребів, замикання до шару {KCL}, '
          f'БЕЗ ІСТИНИ\n')

    # акумулятори: [шар][родина] -> суми по мережах
    S1 = {}    # ‖сер_d f‖²   (сума по мережах)
    S2 = {}    # сер_d ‖f‖²
    HL = {}    # те саме на двох половинах жеребів (пломба)
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            kcl = min(KCL, d - 1)
            cl = closure_parts(wc, kcl, dev, extras=True)
            sm = sq = hs = None
            for dr in range(ndraw):
                fv = feats(wc, npair, cl, dev, 88_000_000 + 733 * n + dr, kcl)
                if sm is None:
                    sm = [[torch.zeros_like(t) for t in ly] for ly in fv]
                    sq = [[0.0 for _ in ly] for ly in fv]
                    hs = [[[torch.zeros_like(t) for t in ly] for ly in fv]
                          for _ in range(2)]
                h = 0 if dr < ndraw // 2 else 1
                for li, ly in enumerate(fv):
                    for qi, t in enumerate(ly):
                        sm[li][qi] += t
                        hs[h][li][qi] += t
                        sq[li][qi] += float(t.pow(2).sum())
            for li in range(len(sm)):
                for qi in range(4):
                    mean = sm[li][qi] / ndraw
                    m2 = float(mean.pow(2).sum())
                    a2 = sq[li][qi] / ndraw
                    S1.setdefault((li, qi), 0.0)
                    S2.setdefault((li, qi), 0.0)
                    S1[(li, qi)] += m2
                    S2[(li, qi)] += a2
                    hh = [float((hs[k][li][qi] / (ndraw // 2)).pow(2).sum())
                          for k in range(2)]
                    HL.setdefault((li, qi), [0.0, 0.0])
                    HL[(li, qi)][0] += hh[0]
                    HL[(li, qi)][1] += hh[1]
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    D = ndraw
    nl = max(k[0] for k in S1) + 1
    print(f'\n`r` = ‖зміщення замикання‖ / ‖шум вибірки‖.  '
          f'r<1 ознака ЧИСТА, r>1 — вливає зміщення\n')
    print(f'{"шар":>4s} ' + ' '.join(f'{f:>11s}' for f in FAM))
    print('-' * 52)
    cross = {q: None for q in range(4)}
    rows = {}
    for li in range(nl):
        cells = []
        for qi in range(4):
            m2, a2 = S1[(li, qi)], S2[(li, qi)]
            b2 = (D * m2 - a2) / (D - 1.0)
            nz = a2 - m2                       # шум² (сер_d‖f−сер‖²)
            r = (max(b2, 0.0) / max(nz, 1e-300)) ** 0.5
            rows[(li, qi)] = r
            if cross[qi] is None and r > 1.0:
                cross[qi] = li + 1
            cells.append(f'{r:11.3f}')
        if li < 12 or (li + 1) % 4 == 0:
            print(f'{li+1:4d} ' + ' '.join(cells))

    print(f'\n{"перетин r=1":>12s}: ' + ' '.join(
        f'{FAM[q].split(":")[0]}={cross[q] if cross[q] else ">"+str(nl)}'
        for q in range(4)))
    # ✅ПЛОМБА ПОЛОВИН
    dev_h = []
    for k, (h0, h1) in HL.items():
        if h0 + h1 > 0:
            dev_h.append(abs(h0 - h1) / (0.5 * (h0 + h1)))
    print(f'✅ПЛОМБА ПОЛОВИН (дві незалежні половини жеребів): '
          f'медіанна розбіжність {np.median(dev_h):.3f}')
    r_s1 = rows[(0, 0)]
    print(f'✅ПЛОМБА `S` НА ШАРІ 1: r={r_s1:.4f} '
          f'(після вибілювання ŝ₁≡замикання ⟹ мусить бути ≈0)')
    print(f'📣ЗВІРКА З jitprof: `A` на шарі 8 → r={rows[(7,1)]:.3f} '
          f'(передбачено ≈2.4)')

    best = max(range(4), key=lambda q: cross[q] if cross[q] else nl + 1)
    kbest = cross[best] if cross[best] else nl
    print(f'\n🔑 НАЙГЛИБША ЧИСТА РОДИНА: {FAM[best]} — тримається до шару '
          f'{kbest}')
    print(f'🔑 ВИРОК: напрямок «читати глибше» '
          f'{"ЖИВИЙ" if kbest >= 16 else "МЕРТВИЙ"} '
          f'(🟢гейт: перетин ≥16, зараз {kbest}; нинішній KSIG=8)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
