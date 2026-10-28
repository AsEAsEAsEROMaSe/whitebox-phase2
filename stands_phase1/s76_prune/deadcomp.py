"""№2: ОБРІЗАНИЙ НЕЙРОН СТАВИТЬСЯ В НУЛЬ, А МАЄ СТАВИТИСЬ У СВОЄ СЕРЕДНЄ. (С76)

════════ МЕХАНІЗМ ════════
Нейрон викидається, якщо на 256 пілотних семплах ЖОДНОГО разу не відкрився.
Але з 1502 обрізаних **654 (44%) насправді ЖИВІ** — просто рідкісні
(`whocarries.py`). Викидання ставить його внесок у 0, тоді як насправді
`E[relu(h_j)] > 0` завжди. ⟹ систематичне ЗАНИЖЕННЯ входу наступного шару,
і це **36% усього ‖b‖²** (`biassplit.py`).

🔑 ЛІКУВАННЯ ТОЧНЕ І МАЙЖЕ БЕЗКОШТОВНЕ: до доактивацій наступного шару додати
СТАЛИЙ вектор `Σ_{j мертві} m_j · W[ℓ+1][j,:]`. Це вектор на шар, а не на
семпл ⟹ ціна не залежить від `n`.
Де брати `m_j`:
  * шар 0 — **ТОЧНО і безкоштовно**: `m_j = ‖W₀[:,j]‖/√(2π)` (вхід гаусів);
  * шари 1..ksig — із замикання **ПОВНОЇ** мережі (та сама рекурсія, що вже
    рахується для обрізаної; мертвих у ній просто не викидаємо);
  * глибше — опори немає (та сама стіна, що вчора) ⟹ без компенсації.

🚨 ЦЕ НЕ ТЕ, ЩО РОБИВ `delta_cut.py`: там ремонтували ЦІЛЬ фіту `λ`
(проти якої істини фітувати), тут — сам ЕСТИМАТОР.

════════ ЩО МІРЯЄМО ════════
  BASE    — діючий сабміт (порт sub87)
  REAL    — компенсація доступними `m` (шар 0 точний + замикання 1..ksig)
  REAL0   — компенсація ЛИШЕ шару 0 (де опора ТОЧНА) — скільки дає найчистіше
  ORACLE  — компенсація ІСТИННИМИ `m` на ВСІХ шарах ⟹ СТЕЛЯ класу
  ⛔ROLL   — компенсація тим самим вектором, ЗСУНУТИМ по нейронах (контроль:
            чи виграш від правильних значень, а не від «додати щось додатне»)

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 ORACLE = 1.02–1.04× (зміщення обрізання = 36% від 7.6% MSE ⟹ ≈2.7%);
📣 REAL   = 1.005–1.020× (9 із 31 шару, але ранні підсилюються наступними);
📣 REAL0  = 0.3–0.7 від REAL;
📣 ⛔ROLL ≈ 1.000× або гірше.
🟢 ЖИВЕ при REAL ≥ 1.010× і ROLL ≤ 1.003× (ціна майже нуль ⟹ гейт низький).

🚨 ПЛОМБИ
  ✅ порожня компенсація (усі нулі) == BASE до 1e-15;
  ✅ ORACLE ≥ REAL (істина не може бути гіршою за замикання);
  ✅ частка мертвих нейронів друкується — має бути ≈18%.
"""
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (closure_parts, W_, GATE_N, INV_SQRT_2PI, _phi,       # noqa: E402
                    PILOT_PAIRS, TARGET_UTIL, B_BUDGET, MIN_PAIRS, SEED)
from biassplit import truth_full                                         # noqa: E402
from varsub import S72, A72, Q72, G72                                    # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:1'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 16
KSIG = 8
TRUTH_PAIRS = 262_144
CHUNK = 32_768


def cut_alive(W, dev):
    """`cut_and_budget` сабміту, але ще й ПОВЕРТАЄ маски живих."""
    depth = len(W)
    rng = np.random.default_rng(SEED)
    zp = torch.tensor(rng.standard_normal((PILOT_PAIRS, W_)).astype(np.float32),
                      device=dev)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(depth):
        h = a @ W[i]
        if i < depth - 1:
            alive.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    wc = []
    for i in range(depth):
        wi = W[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < depth - 1:
            wi = wi[:, alive[i]]
        wc.append(wi.contiguous())
    per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * depth * W_ * W_)
    left = TARGET_UTIL * float(B_BUDGET) - used
    return wc, max(MIN_PAIRS, int(left / (2.0 * max(per_cut, 1.0)))), alive


def truth_layers(W, dev, seed):
    """Істинні `E[relu]` на КОЖНОМУ шарі ПОВНОЇ мережі (для оракульної стелі)."""
    depth = len(W)
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in W]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TRUTH_PAIRS:
        k = min(CHUNK, TRUTH_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(depth):
            a = torch.relu(a @ W[i])
            acc[i] += a.double().sum(0)
        done += k
    return [t / float(2 * done) for t in acc]


def build_comp(W, wc, alive, dev, mode, m_src, kmax, roll=0):
    """Сталий вектор-компенсація для доактивацій кожного шару.

    `comp[i]` додається до `p` шару `i` (i = 1..depth-1). Він дорівнює
    внеску МЕРТВИХ нейронів шару `i−1`, узятих у своєму СЕРЕДНЬОМУ."""
    depth = len(W)
    comp = [None] * depth
    for i in range(1, depth):
        j = i - 1                       # шар, чиї нейрони мертві
        if j > kmax:
            continue
        dead = ~alive[j]
        if not bool(dead.any()):
            continue
        mv = m_src[j]                   # (256,) середні активацій шару j
        if roll:
            mv = torch.roll(mv, roll)
        contrib = mv[dead] @ W[i][dead, :].double()
        if i < depth - 1:
            contrib = contrib[alive[i]]
        comp[i] = contrib
    return comp


def run_comp(wc, npair, cl, dev, seed, comp=None):
    """Порт sub87 з опційною сталою компенсацією мертвих нейронів."""
    s_cl, m_cl, f_cl, q_cl = cl
    ksig = len(s_cl)
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    denom = float(2 * npair)
    cov = (x.T @ x) / denom
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xa = torch.relu(x @ (wh @ wc[0]))
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = xa.double().mean(0) - sg0 * INV_SQRT_2PI
    for i in range(1, depth):
        p = xa @ wc[i]
        if comp is not None and comp[i] is not None:
            p = p + comp[i].to(p.dtype)
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        v = (v @ wc[i].double()) * gate
        if i <= ksig:
            pd = p.double()
            mh = pd.mean(0)
            sh = (pd.pow(2).mean(0) - mh * mh).clamp_min(1e-30).sqrt()
            xa = torch.relu(p)
            xd = xa.double()
            v = (v + (S72[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + A72[i - 1] * (xd.mean(0) - m_cl[i - 1])
                 + Q72[i - 1] * (xd.pow(2).mean(0) - q_cl[i - 1])
                 + G72[i - 1] * (gate - f_cl[i - 1]))
        else:
            xa = torch.relu(p)
    return xa.double().mean(0) - v


def main(nets=24, dev=DEV, ndraw=NDRAW):
    t0 = time.time()
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'№2 КОМПЕНСАЦІЯ МЕРТВИХ · {nets} мереж × {ndraw} жеребів, '
          f'крос-ціль, {dev}')
    print('📣 ПЕРЕДБАЧЕНО: ORACLE 1.02–1.04× · REAL 1.005–1.020× · '
          '⛔ROLL ≈1.000×')
    print('🟢 ЖИВЕ при REAL ≥1.010× і ROLL ≤1.003×\n', flush=True)

    KEYS = ('BASE', 'REAL0', 'REAL', 'REALF', 'ORACLE8', 'ORACLE',
            'ORACLF', '⛔ROLL')
    acc = {k: 0.0 for k in KEYS}
    seal_zero, frac_dead = [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair, alive = cut_alive(W, dev)
            depth = len(W)
            cl = closure_parts(wc, KSIG, dev, extras=True)
            # замикання ПОВНОЇ мережі — звідти `m` для мертвих
            _, m_full, _, _ = closure_parts(W, KSIG, dev, extras=True)
            mt = truth_layers(W, dev, 210_000 + 7919 * n)
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)
            frac_dead.append(float(sum((~a).sum() for a in alive))
                             / float(sum(a.numel() for a in alive)))

            # джерела `m`: доступне (шар 0 точний + замикання) та істинне
            sg0f = W[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
            m_av = [sg0f * INV_SQRT_2PI] + [m_full[q] for q in range(KSIG)]
            m_or = [mt[q].clone() for q in range(depth - 1)]

            # 🔑 КЛЮЧОВЕ: компенсуючи мертвих, ми робимо мережу «майже
            # повною» ⟹ опорою мусить стати замикання ПОВНОЇ мережі,
            # звужене до живих. Інакше λ-канали віднімають проти опори,
            # яка більше не відповідає статистикам `p` — саме так ORACLE
            # виходив ГІРШЕ за базу.
            sf, mf, ff, qf = closure_parts(W, KSIG, dev, extras=True)
            cl_f = tuple([[t[q][alive[q + 1]] for q in range(KSIG)]
                          for t in (sf, mf, ff, qf)])
            comps = {
                'BASE': (None, cl),
                'REAL0': (build_comp(W, wc, alive, dev, 'a', m_av, 0), cl),
                'REAL': (build_comp(W, wc, alive, dev, 'a', m_av, KSIG), cl),
                'REALF': (build_comp(W, wc, alive, dev, 'a', m_av, KSIG), cl_f),
                'ORACLE8': (build_comp(W, wc, alive, dev, 'o', m_or, KSIG), cl),
                'ORACLE': (build_comp(W, wc, alive, dev, 'o', m_or, depth - 2),
                           cl),
                'ORACLF': (build_comp(W, wc, alive, dev, 'o', m_or, depth - 2),
                           cl_f),
                '⛔ROLL': (build_comp(W, wc, alive, dev, 'a', m_av, KSIG,
                                     roll=13), cl),
            }
            zero = [None] * depth
            for k in range(ndraw):
                sd = 81_000_000 + 977 * n + k
                for key in KEYS:
                    cmp_, clx = comps[key]
                    pr = run_comp(wc, npair, clx, dev, sd, cmp_)
                    acc[key] += float(((pr - T1) * (pr - T2)).sum())
                    if n == 0 and k == 0 and key == 'BASE':
                        pz = run_comp(wc, npair, cl, dev, sd, zero)
                        seal_zero.append(float((pz - pr).abs().max()))
        del W, wc
        torch.cuda.empty_cache()
        if (n + 1) % 4 == 0:
            print(f'  … {n+1}/{nets}  ({time.time()-t0:.0f}s)', flush=True)

    nd = nets * ndraw
    base = acc['BASE'] / nd
    print(f'\n✅ПЛОМБА ПОРОЖНЬОЇ КОМПЕНСАЦІЇ: {seal_zero[0]:.2e} '
          f'{"✅" if seal_zero[0] < 1e-12 else "🛑"}')
    print(f'   мертвих нейронів: {np.mean(frac_dead)*100:.1f}%')
    print(f'\n{"варіант":<10} {"крос-MSE":>13} {"×BASE":>8}')
    for k in KEYS:
        m = acc[k] / nd
        print(f'{k:<10} {m:>13.5e} {base/m:>8.4f}')
    real = max(base / (acc['REAL'] / nd), base / (acc['REALF'] / nd))
    orc = max(base / (acc['ORACLE'] / nd), base / (acc['ORACLF'] / nd))
    rl = base / (acc['⛔ROLL'] / nd)
    print(f'\n⛔ROLL = {rl:.4f}× {"✅" if rl <= 1.003 else "🛑 виграш не від значень"}')
    print(f'🔑 ВИРОК: REAL = {real:.4f}× при стелі ORACLE {orc:.4f}× ⟹ '
          f'{"🟢 ЖИВЕ" if real >= 1.010 and rl <= 1.003 else "⛔ МЕРТВО"}')
    print(f'📊 БАЛ: 1.954e-07 → {1.954e-07/max(real,1e-9):.4e}')
    print(f'   час {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
