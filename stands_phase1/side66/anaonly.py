"""ЩО БУДЕ, ЯКЩО ВИКИНУТИ СЕМПЛИ ЗОВСІМ — ЛИШЕ АНАЛІТИКА? (С74, питання юзера)

ІДЕЯ. `pred = m_замик` останнього шару — чиста аналітика, ЖОДНОГО семпла.
Ціна майже нульова ⟹ множник упаде на ПІДЛОГУ `max(0.1, C/B)`.

📐 ЧОМУ ЦЕ ВАЖЛИВЕ ПИТАННЯ НЕЗАЛЕЖНО ВІД ВІДПОВІДІ. Ми зараз на множнику
**0.1034** при підлозі **0.1** ⟹ до «безкоштовної зони» лишилось **3.4%**.
Тобто вся подальша економія флопів обмежена 3.4%, а нижче підлоги економія
не дає НІЧОГО. Цей стенд міряє обидві межі одразу.

ЩО РАХУЄМО
  1. MSE чистого замикання (глибина 31) проти незалежної істини;
  2. його ФЛОПИ (рекурсія `Wᵀ C W` — два матмули `n³` на шар);
  3. бал `MSE × max(0.1, C/B)` і порівняння з діючим sub86 (1.972e-07);
  4. 🔑 СКІЛЬКИ БЮДЖЕТУ ЛИШИЛОСЬ ДО ПІДЛОГИ і що дасть віддати його у вибірку.

ПЛОМБИ
  ✅ похибка замикання мусить лягти біля **0.9115%** (`k3rec.py`, шар 30);
  ✅ MSE = (rel·nrm)² мусить збігтися з прямим виміром.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import cut_and_budget, W_, _phi, _Phi, TARGET_UTIL    # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
B_BUD = 272_000_000_000
SUB86 = 1.972e-07          # діючий бал
MULT86 = 0.1034            # його множник


def closure_full(wc, dev):
    """Замикання на ВСІ шари; повертає `m` останнього + лічильник флопів."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    fl = 0.0
    for i in range(len(wc)):
        w = wc[i].double()
        a, b = w.shape
        cp = w.T @ (C @ w)
        fl += 2.0 * a * a * b + 2.0 * a * b * b       # C@w, потім wᵀ@(·)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        fl += 2.0 * a * b
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        fl += 6.0 * b * b
        d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(d - C.diagonal())
    return m, fl


def truth_last(wc, dev, seed, npair=131_072, chunk=16384):
    acc = torch.zeros(wc[-1].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(seed)
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(L_):
            a = torch.relu(a @ wc[i])
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

    mse, rel, FL, NRM = [], [], [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            # 🚨 БЕЗ семплів обрізання неможливе ⟹ ваги ПОВНІ
            wc = [w for w in W]
            m, fl = closure_full(wc, dev)
            T = truth_last(wc, dev, 260_000 + 7919 * n)
            e = m - T
            mse.append(float((e * e).mean()))
            NRM.append(float((T ** 2).mean().sqrt()))
            rel.append(100.0 * float((e * e).mean().sqrt()) / NRM[-1])
            FL.append(fl)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    md = np.median
    F = md(FL)
    fb = F / B_BUD
    mult = max(0.1, fb)
    sc = md(mse) * mult
    print(f'\n{nets} мереж, істина 262144 семплів\n')
    print(f'✅ПЛОМБА: похибка замикання {md(rel):.4f}%  '
          f'(`k3rec.py` дав 0.9115% на шарі 30)')
    print(f'   nrm = {md(NRM):.4f}\n')
    print(f'{"величина":>28s} {"чиста аналітика":>18s} {"sub86":>14s}')
    print('-' * 64)
    print(f'{"MSE":>28s} {md(mse):18.4e} {1.91e-06:14.4e}')
    print(f'{"флопи F":>28s} {F:18.4e} {0.0975*B_BUD:14.4e}')
    print(f'{"F/B":>28s} {fb:18.5f} {0.0975:14.5f}')
    print(f'{"МНОЖНИК = max(0.1, ·)":>28s} {mult:18.4f} {MULT86:14.4f}')
    print(f'{"БАЛ":>28s} {sc:18.4e} {SUB86:14.4e}')
    print('-' * 64)
    print(f'⟹ чиста аналітика **у {sc/SUB86:.1f}× ГІРША** за діючий сабміт\n')

    # ── 🔑 ГОЛОВНЕ: скільки бюджету лишилось до ПІДЛОГИ ──
    print('🔑 АРИФМЕТИКА ПІДЛОГИ (те, що це питання виявляє насправді)\n')
    fb86, rb86 = 0.0975, MULT86 - 0.0975
    room = 0.1 - rb86                      # скільки F/B дозволено при підлозі
    print(f'   sub86: F/B={fb86:.4f} + λR/B={rb86:.4f} = {MULT86:.4f}')
    print(f'   підлога діє, поки F/B ≤ {room:.4f} ⟹ до неї лишилось '
          f'{100*(fb86-room)/fb86:.1f}% флопів')
    print(f'   💰 економія ДО підлоги дає {MULT86/0.1:.4f}× і ЖОДНОГО множника '
          f'далі')
    # віддати зекономлене у вибірку: MSE ∝ 1/n, n ∝ F
    for cut in (0.039, 0.061, 0.10):
        f_new = fb86 * (1.0 - cut)
        if f_new <= room:
            # можна ДОДАТИ семплів до межі підлоги
            grow = room / f_new
            print(f'   економія {100*cut:.1f}% ⟹ F/B={f_new:.4f} ≤ {room:.4f}: '
                  f'можна збільшити вибірку в {grow:.3f}× ⟹ бал '
                  f'{SUB86 * (0.1/MULT86) / grow:.4e} '
                  f'({SUB86/(SUB86*(0.1/MULT86)/grow):.3f}×)')
        else:
            print(f'   економія {100*cut:.1f}% ⟹ F/B={f_new:.4f} > {room:.4f}: '
                  f'ще НАД підлогою, бал {SUB86*(f_new+rb86)/MULT86:.4e} '
                  f'({MULT86/(f_new+rb86):.3f}×)')
    print('\n🎯 ВИСНОВОК ЧИТАТИ ТАК: якщо ми вже майже на підлозі, то далі '
          'вигідно\n   НЕ економити, а ВІДДАВАТИ зекономлене у ВИБІРКУ — '
          'доки множник не 0.1.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
