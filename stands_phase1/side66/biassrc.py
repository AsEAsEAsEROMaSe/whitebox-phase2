"""ЗВІДКИ 46% ЗМІЩЕННЯ — ЗАМИКАННЯ ЧИ ОБРІЗАННЯ? (С74)

ЗВІДКИ. sub87 (321788) дав ПРЯМИЙ вимір розкладу MSE: вибірка ↓3.75%, MSE
зросла лише на 2.09% замість 3.90% ⟹ **шум 54%, ЗМІЩЕННЯ 46%**.
Прибрати зміщення = MSE ×0.54 = **1.85×** — найбільший важіль, що лишився.

🎯 ДВА ПІДОЗРЮВАНІ, І ВОНИ ЛІКУЮТЬСЯ ПО-РІЗНОМУ:
  1. **ЗАМИКАННЯ** — `m_cl` неточне, канали `(x̄ − m_cl)` несуть його похибку.
     Клас вичерпано (стеля локальної формули 1.111×) ⟹ якщо винне воно, важіль
     МЕРТВИЙ;
  2. **ОБРІЗАННЯ** — пілот бачить 256 семплів і оголошує нейрон мертвим за
     `max(h) < 0`. Живі серед них губляться СИСТЕМАТИЧНО, і така похибка НЕ
     падає з `n` узагалі ⟹ якщо винне воно, важіль ЖИВИЙ (ми ПІД підлогою,
     0.3% флопів безкоштовні).

ЯК МІРЯЄМО. Один і той самий прогін на ТИХ САМИХ семплах, двічі:
  A — з обрізанням (як у сабміті);
  B — БЕЗ обрізання (повні матриці 256×256), решта ідентична.
Різниця MSE — ЧИСТИЙ внесок обрізання (шум вибірки скорочується тотожно, бо
`z` той самий).
🔑 заразом рахуємо, СКІЛЬКИ нейронів обрізано і скільки з них ЖИВІ на великій
вибірці — це прямий діагноз, а не здогад.

ПЛОМБИ
  ✅ якщо жодного нейрона не обрізано, A і B мусять збігтися БІТ У БІТ;
  ✅ `n_pairs` ОДНАКОВИЙ в обох (інакше міряємо бюджет, а не обрізання);
  📊 8 мереж × 24 жереби, істина — НЕЗАЛЕЖНИЙ жереб.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (run_side1, closure_parts, W_, KSIG,           # noqa: E402
                    B_BUDGET, MIN_PAIRS, SEED)

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = 12
UTIL = 0.0908                # як у sub87
PILOT = 128

S72 = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
       0.0088285, 0.0007922, 0.0015044, 0.0001062)
A72 = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
       -0.0025822, -0.0030778, 0.0006822, 0.0068061)
Q72 = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
       0.0189783, 0.0250559, 0.0340802, 0.0415433)
G72 = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
       -0.0146129, -0.0100612, 0.0167223, 0.0168109)


THR = (0.0, 0.10, 0.15, 0.20, 0.28)   # звужена сітка навколо оптимуму


def pilot_alive_thr(W, dev, thr, pilot=PILOT):
    """`alive` з ПОМ'ЯКШЕНИМ критерієм: `max(h) ≥ −thr·std(h)`.

    🔑 нейрон, що відкривається РІДКО, майже напевно не відкриється на 256
    семплах пілота — і буде викинутий. А саме рідко перемикні несуть похибку.
    Пом'якшений поріг зберігає СУМНІВНИХ, платячи флопами лише за них."""
    rng = np.random.default_rng(SEED)
    zp = torch.tensor(rng.standard_normal((pilot, W_)).astype(np.float32),
                      device=dev)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(len(W)):
        h = a @ W[i]
        if i < len(W) - 1:
            sd = h.std(0).clamp_min(1e-30)
            alive.append(h.max(0).values >= -thr * sd)
        a = torch.relu(h)
    return alive


def pilot_alive(W, dev, pilot=PILOT):
    rng = np.random.default_rng(SEED)
    zp = torch.tensor(rng.standard_normal((pilot, W_)).astype(np.float32),
                      device=dev)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(len(W)):
        h = a @ W[i]
        if i < len(W) - 1:
            alive.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    return alive


def truly_alive(W, dev, seed, npair=65_536, chunk=8192):
    """Хто відкривається хоч раз на ВЕЛИКІЙ вибірці (еталон «живості»)."""
    ever = [torch.zeros(W[i].shape[1], dtype=torch.bool, device=dev)
            for i in range(len(W) - 1)]
    rng = np.random.default_rng(seed)
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(len(W)):
            h = a @ W[i]
            if i < len(W) - 1:
                ever[i] |= (h.max(0).values >= 0.0)
            a = torch.relu(h)
        done += k
    return ever


def truth_last(W, dev, seed, npair=131_072, chunk=16384):
    acc = torch.zeros(W[-1].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(seed)
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(L_):
            a = torch.relu(a @ W[i])
        acc += a.double().sum(0)
        done += k
    return acc / float(2 * npair)


def main(nets=8, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    res = {'A з обрізанням': [], 'B БЕЗ обрізання': []}
    SW = {t: {'mse': [], 'fb': [], 'cut': [], 'wrong': []} for t in THR}
    CUT, WRONG, LOST = [], [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            alive = pilot_alive(W, dev)
            ever = truly_alive(W, dev, 330_000 + 7919 * n)
            ncut = sum(int((~a).sum()) for a in alive)
            nwrong = sum(int((ever[i] & ~alive[i]).sum())
                         for i in range(len(alive)))
            CUT.append(ncut)
            WRONG.append(nwrong)
            LOST.append(100.0 * nwrong / max(ncut, 1))

            wc = []
            for i in range(L_):
                wi = W[i]
                if i > 0:
                    wi = wi[alive[i - 1], :]
                if i < L_ - 1:
                    wi = wi[:, alive[i]]
                wc.append(wi.contiguous())
            per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1])
                                for w in wc)
            used = 2.0 * PILOT * (2.0 * L_ * W_ * W_)
            npair = max(MIN_PAIRS,
                        int((UTIL * B_BUDGET - used) / (2.0 * per_cut)))
            T = truth_last(W, dev, 260_000 + 7919 * n)

            for nm, ws in (('A з обрізанням', wc), ('B БЕЗ обрізання', W)):
                ks = min(KSIG, len(ws) - 1)
                s_cl, m_cl, f_cl, q_cl = closure_parts(ws, ks, dev, extras=True)
                a2 = 0.0
                for d in range(NDRAW):
                    # 🔑 ТОЙ САМИЙ сід ⟹ ті самі `z` ⟹ шум скорочується
                    pr, _ = run_side1(ws, npair, s_cl, m_cl, dev,
                                      63_000_000 + 977 * n + d,
                                      lam_s=S72[:ks], lam_a=A72[:ks],
                                      lam_q=Q72[:ks], lam_g=G72[:ks],
                                      q_ext=q_cl, p_ext=f_cl)
                    a2 += float(((pr - T) ** 2).mean())
                res[nm].append(a2 / NDRAW)
            # ── СВІП ПОРОГУ: скільки коштує зберігати сумнівних ──
            for t in THR:
                al = pilot_alive_thr(W, dev, t)
                ws = []
                for i in range(L_):
                    wi = W[i]
                    if i > 0:
                        wi = wi[al[i - 1], :]
                    if i < L_ - 1:
                        wi = wi[:, al[i]]
                    ws.append(wi.contiguous())
                pc = 2.0 * sum(float(w.shape[0]) * float(w.shape[1])
                               for w in ws)
                np_t = max(MIN_PAIRS,
                           int((UTIL * B_BUDGET - used) / (2.0 * pc)))
                F = used + 2.0 * np_t * pc
                for i in range(min(KSIG, len(ws) - 1) + 1):
                    aa, bb = ws[i].shape
                    F += 2.0 * aa * aa * bb + 2.0 * aa * bb * bb
                ks = min(KSIG, len(ws) - 1)
                s_cl, m_cl, f_cl, q_cl = closure_parts(ws, ks, dev, extras=True)
                a2 = 0.0
                for d in range(NDRAW):
                    pr, _ = run_side1(ws, np_t, s_cl, m_cl, dev,
                                      63_000_000 + 977 * n + d,
                                      lam_s=S72[:ks], lam_a=A72[:ks],
                                      lam_q=Q72[:ks], lam_g=G72[:ks],
                                      q_ext=q_cl, p_ext=f_cl)
                    a2 += float(((pr - T) ** 2).mean())
                SW[t]['mse'].append(a2 / NDRAW)
                SW[t]['fb'].append(F / B_BUDGET)
                SW[t]['cut'].append(sum(int((~q).sum()) for q in al))
                SW[t]['wrong'].append(sum(int((ever[i] & ~al[i]).sum())
                                          for i in range(len(al))))
                del ws
                torch.cuda.empty_cache()
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (обрізано {ncut}, з них живих {nwrong})',
              flush=True)

    md = np.median
    print(f'\n{nets} мереж × {NDRAW} жеребів, ТІ САМІ семпли, `n_pairs` спільний\n')
    print(f'{"обрізано нейронів (медіана)":>34s} {md(CUT):10.0f}')
    print(f'{"з них ЖИВІ на 131k семплів":>34s} {md(WRONG):10.0f}  '
          f'({md(LOST):.1f}% помилкових)')
    print()
    print(f'{"варіант":>20s} {"MSE":>12s} {"відношення":>12s}')
    print('-' * 48)
    a = md(res['A з обрізанням'])
    b = md(res['B БЕЗ обрізання'])
    for nm in ('A з обрізанням', 'B БЕЗ обрізання'):
        print(f'{nm:>20s} {md(res[nm]):12.4e} {a/md(res[nm]):11.4f}×')
    print('-' * 48)
    print(f'⟹ внесок ОБРІЗАННЯ у MSE: {100*(a-b)/a:+.2f}%\n')
    print(f'📐 СВІП ПОРОГУ `max(h) ≥ −thr·std(h)`  (λR/B=0.0059, підлога 0.1)\n')
    print(f'{"thr":>6s} {"обрізано":>9s} {"з них живі":>11s} {"F/B":>7s} '
          f'{"множник":>8s} {"MSE":>11s} {"БАЛ":>11s} {"виграш":>8s}')
    print('-' * 78)
    b0 = None
    for t in THR:
        w = SW[t]
        fb = md(w['fb'])
        mult = max(0.1, fb + 0.0059)
        sc = md(w['mse']) * mult
        if b0 is None:
            b0 = sc
        print(f'{t:6.2f} {md(w["cut"]):9.0f} {md(w["wrong"]):11.0f} '
              f'{fb:7.4f} {mult:8.4f} {md(w["mse"]):11.4e} {sc:11.4e} '
              f'{b0/sc:7.4f}×')
    print('-' * 78)
    print('   thr=0 — діючий сабміт (пломба: мусить збігтися з рядком A)\n')
    print('🎯 ГЕЙТ: якщо B помітно КРАЩА — зміщення від ОБРІЗАННЯ, важіль ЖИВИЙ\n'
          '   (ми під підлогою, точніший пілот майже безкоштовний).\n'
          '   Якщо A≈B — зміщення від ЗАМИКАННЯ, а той клас вичерпано.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
