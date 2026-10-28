"""ДВІ ФАЗИ БЕЗ ЖОДНОЇ ОПЕРАЦІЇ FLOPSCOPE = 47.5% RESIDUAL. ЧИ ЗНІМАЄТЬСЯ? (С73)

ЗВІДКИ. `phases.py` (пломба Σфаз==R, Δ=4 мкс) розклав residual сабміту:
    10 звільнення локальних  3.32 мс (27.0%)  — 0 операцій flopscope
    3  обрізання ваг         2.52 мс (20.5%)  — 0 операцій flopscope
Обидві — чистий python/numpy ПОЗА інструментацією, тобто повністю в residual.
Разом 5.83 мс ⟹ `λR/B = 0.00215` ⟹ 2.1% множника.

ГІПОТЕЗА ПРО МЕХАНІЗМ (перевіряється тут, а не називається):
  * обрізання робить ДВІ фенсі-індексації на шар (`wi[rows,:][:,cols]`) ⟹ дві
    копії; `fnp.ix_` дає ОДНУ (перевірено: біт у біт, дельта 0.0);
  * relu алокує НОВИЙ масив `x` 7.7 МБ на кожній із 31 ітерацій, хоча `p` уже
    не потрібен (`mh`,`sh` рахуються ДО нього) ⟹ `out=p` знімає 31 алокацію
    І 31 звільнення (перевірено: `out=` підтримується, повертає ТОЙ САМИЙ буфер).

🚨 ЦЕ НЕ ЗМІНА ВИБІРКИ І НЕ ЗМІНА ТОЧНОСТІ: обидві правки зобов'язані дати
   вихід БІТ У БІТ. Пломба на це — жорстка (`max|Δ| == 0.0` рівно).
🚨 РІШУЧЕ ПРАВИЛО ВИМІРНОСТІ: розкид R ~0.6 мс ⟹ правка, що дає <0.6 мс,
   НЕВИМІРНА і сабмітити її не варто ([[finding-residual-is-python-time-and-measurable]]).
"""
import glob
import statistics as st
import sys

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)

import numpy as np                                      # noqa: E402
import flopscope as flops                               # noqa: E402
import flopscope.numpy as fnp                           # noqa: E402
from whestbench import MLP                              # noqa: E402

B = 272_000_000_000
LAM = 1e11

_TARGET_UTIL, _MIN_PAIRS, _SEED, _PILOT_PAIRS = 0.0942, 128, 12345, 128
_GATE_N, _INV_GATE = 1024, 1.0 / 1024
_INV_SQRT_2PI, _INV_SQRT_2 = 0.3989422804014327, 0.7071067811865476
_KSIG = 8
_LAM_S = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
          0.0088285, 0.0007922, 0.0015044, 0.0001062)
_LAM_A = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
          -0.0025822, -0.0030778, 0.0006822, 0.0068061)
_LAM_Q = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
          0.0189783, 0.0250559, 0.0340802, 0.0415433)
_LAM_G = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
          -0.0146129, -0.0100612, 0.0167223, 0.0168109)
_A1, _A2, _A3, _A4, _A5, _P = (0.254829592, -0.284496736, 1.421413741,
                               -1.453152027, 1.061405429, 0.3275911)


def _phi(x):
    return fnp.exp(-0.5 * x * x) * _INV_SQRT_2PI


def _Phi(x):
    ax = fnp.abs(x) * _INV_SQRT_2
    t = 1.0 / (1.0 + _P * ax)
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    er = 1.0 - poly * fnp.exp(-ax * ax)
    return 0.5 * (1.0 + fnp.sign(x) * er)


def _closure_parts(w_cut, ksig, width):
    m = fnp.zeros(width, dtype=fnp.float32)
    C = fnp.eye(width, dtype=fnp.float32)
    out, mo, qo, po = [], [], [], []
    for i in range(ksig + 1):
        w = w_cut[i]
        cp = fnp.matmul(w.T, fnp.matmul(C, w))
        s = fnp.sqrt(fnp.maximum(fnp.diagonal(cp), 1e-20))
        if i >= 1:
            out.append(s)
        mp = fnp.matmul(m, w)
        t = mp / s
        ph = _phi(t)
        Ph = _Phi(t)
        if i >= 1:
            po.append(Ph)
        m = mp * Ph + s * ph
        C = cp * fnp.outer(Ph, Ph) + (0.5 * (cp * cp)) * fnp.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + fnp.diag(d - fnp.diagonal(C))
        if i >= 1:
            mo.append(m)
            qo.append(fnp.diagonal(C) + m * m)
    return out, mo, qo, po


def predict(mlp, budget, ix=False, inplace=False, free=False, take=False):
    """sub72; `ix` — обрізання одним `ix_`, `inplace` — relu на місці,
    `free` — ранній `del` великих проміжних, `take` — обрізання по int-індексах."""
    width, depth = mlp.width, len(mlp.weights)
    per_sample_full = 2.0 * depth * width * width
    rng = fnp.random.default_rng(_SEED)

    zp = fnp.array(rng.standard_normal((_PILOT_PAIRS, width)).astype(fnp.float32))
    xp = fnp.concatenate([zp, -zp], axis=0)
    alive = []
    a = xp
    for i in range(depth):
        h = fnp.matmul(a, mlp.weights[i])
        if i < depth - 1:
            alive.append(fnp.max(h, axis=0) >= 0.0)
        a = fnp.maximum(h, 0.0)

    w_cut = []
    for i in range(depth):
        wi = mlp.weights[i]
        if ix:
            # 🔑 ОДНА копія замість двох (значення тотожні: та сама підматриця)
            if i > 0 and i < depth - 1:
                wi = wi[fnp.ix_(alive[i - 1], alive[i])]
            elif i > 0:
                wi = wi[alive[i - 1], :]
            elif i < depth - 1:
                wi = wi[:, alive[i]]
        elif take:
            # 🔑 булеву маску — в int-індекси ОДИН раз, далі `take` (швидкий шлях)
            if i > 0:
                wi = fnp.take(wi, fnp.flatnonzero(alive[i - 1]), axis=0)
            if i < depth - 1:
                wi = fnp.take(wi, fnp.flatnonzero(alive[i]), axis=1)
        else:
            if i > 0:
                wi = wi[alive[i - 1], :]
            if i < depth - 1:
                wi = wi[:, alive[i]]
        w_cut.append(wi)
    per_sample_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in w_cut)

    used = 2.0 * _PILOT_PAIRS * per_sample_full
    left = _TARGET_UTIL * float(budget) - used
    n_pairs = max(_MIN_PAIRS, int(left / (2.0 * max(per_sample_cut, 1.0))))

    z = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
    x = fnp.concatenate([z, -z], axis=0)
    if free:
        # 🔑 `z` — 6.3 МБ і живе до кінця `predict`, хоча потрібен лише тут.
        #    Те саме `zp`,`xp`,`a`,`h` з пілота (вони теж у кадрі).
        del z, zp, xp, a, h
    cov = fnp.matmul(x.T, x) / float(2 * n_pairs)
    evals, evecs = fnp.linalg.eigh(cov)
    inv_sqrt = 1.0 / fnp.sqrt(fnp.maximum(evals, 1e-8))
    whiten = fnp.matmul(evecs * inv_sqrt, evecs.T)
    if free:
        del cov, evals, evecs, inv_sqrt

    denom = float(2 * n_pairs)
    zrow = fnp.zeros(width, dtype=fnp.float32)
    out = []

    ksig = _KSIG if _KSIG <= depth - 1 else depth - 1
    if ksig >= 1:
        s_cl, m_cl, q_cl, P_cl = _closure_parts(w_cut, ksig, width)
    else:
        s_cl, m_cl, q_cl, P_cl = [], [], [], []

    w0 = fnp.matmul(whiten, w_cut[0])
    x = fnp.maximum(fnp.matmul(x, w0), 0.0)
    if free:
        del whiten, w0
    out.append(zrow if depth > 1 else fnp.sum(x, axis=0) / denom)

    sg0 = fnp.sqrt(fnp.maximum(fnp.sum(w_cut[0] * w_cut[0], axis=0), 1e-30))
    v = fnp.sum(x, axis=0) / denom - sg0 * _INV_SQRT_2PI

    for i in range(1, depth):
        p = fnp.matmul(x, w_cut[i])
        gate = fnp.sum((p[:_GATE_N] > 0.0).astype(p.dtype), axis=0) * _INV_GATE
        v = fnp.matmul(v, w_cut[i]) * gate
        if i <= ksig:
            mh = fnp.sum(p, axis=0) / denom
            sh = fnp.sqrt(fnp.maximum(
                fnp.sum(p * p, axis=0) / denom - mh * mh, 1e-30))
            # 🚨 `mh`,`sh` УЖЕ пораховані ⟹ `p` більше не потрібен
            x = fnp.maximum(p, 0.0, out=p) if inplace else fnp.maximum(p, 0.0)
            xm = fnp.sum(x, axis=0) / denom
            xq = fnp.sum(x * x, axis=0) / denom
            v = (v + (_LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + _LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + _LAM_Q[i - 1] * (xq - q_cl[i - 1])
                 + _LAM_G[i - 1] * (gate - P_cl[i - 1]))
        else:
            x = fnp.maximum(p, 0.0, out=p) if inplace else fnp.maximum(p, 0.0)
        out.append(fnp.sum(x, axis=0) / denom if i == depth - 1 else zrow)

    out[-1] = out[-1] - v
    return fnp.stack(out, axis=0)


# 🚨 A′ — ДУБЛІКАТ A під іншим іменем. Якщо A і A′ розходяться більше, ніж
#    варіанти між собою, стенд міряє ДРЕЙФ, а не правку.
VAR = [('A базовий sub72', dict()),
       ('E ранній del великих', dict(free=True)),
       ('F обрізання take(int)', dict(take=True)),
       ('G E+F', dict(free=True, take=True)),
       ("⛔A′ дублікат A (пломба)", dict())]


def main(reps=11, warm=3, nets=1):
    wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = set(int(v) for v in sd)
    fs = [q for q in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(q)['seed']) in have][:nets]

    # 🚨 ОБЛІК ПО-МЕРЕЖНИЙ: `npair` (а з ним і R) різний на різних мережах ⟹
    #    об'єднувати сирі R у спільний пул НЕ МОЖНА. Медіана береться по
    #    повторах усередині мережі, порівняння — по МЕРЕЖАХ.
    acc = {k: {'R': [], 'F': []} for k, _ in VAR}
    per = {k: [] for k, _ in VAR}
    for f in fs:
        s0 = int(np.load(f)['seed'])
        j = int(np.where(sd == s0)[0][0])
        Wn = [fnp.array(np.asarray(wz[j][k], np.float32)) for k in range(32)]
        mlp = MLP(width=256, depth=32, weights=Wn, seed=s0)

        ref = None
        for _ in range(warm):
            with flops.BudgetContext(flop_budget=B, quiet=True):
                predict(mlp, B)
        last, loc = {}, {k: [] for k, _ in VAR}
        # 🔑 ЧЕРГУВАННЯ: усі варіанти всередині кожного повтору ⟹ дрейф
        #    (нагрівання, фрагментація купи) б'є по ВСІХ однаково, а не по
        #    останньому в списку.
        for _ in range(reps):
            for name, kw in VAR:
                with flops.BudgetContext(flop_budget=B, quiet=True) as c:
                    last[name] = predict(mlp, B, **kw)
                d = c.summary_dict()
                acc[name]['R'].append(d['residual_wall_time_s'])
                acc[name]['F'].append(c.flops_used)
                loc[name].append((d['residual_wall_time_s'], c.flops_used))
        for name, _ in VAR:
            R = st.median([q[0] for q in loc[name]])
            F = st.median([q[1] for q in loc[name]])
            per[name].append((R, max(0.1, F / B + LAM * R / B)))
        for name, kw in VAR:
            g = np.asarray(last[name])
            if ref is None:
                ref = g
            dm = float(np.abs(g - ref).max())
            acc[name]['D'] = dm
            # 🔑 `inplace` зобов'язаний бути БІТ У БІТ (той самий буфер, ті самі
            #    операції). `ix_` — ні: значення підматриць тотожні на всіх 32
            #    шарах (перевірено окремо), але двокрокове обрізання лишає 31 з
            #    32 матриць НЕ C-contiguous, а `ix_` робить їх суцільними ⟹
            #    інший порядок підсумовування в BLAS. Це float32-eps, не інша
            #    математика; поріг — 1e-06 при нашій RMS-похибці 4.5e-04.
            lim = 1e-06 if (kw.get('ix') or kw.get('take')) else 0.0
            assert dm <= lim, f'🛑{name}: вихід змінився, max|Δ|={dm:.3e}'
        print(f'  ✅ПЛОМБА виходу на мережі {s0}: '
              + '  '.join(f'{k}={acc[k].get("D", 0.0):.1e}' for k, _ in VAR))

    a0 = VAR[0][0]
    print(f'\n{nets} мереж × {reps} прогонів (прогрів {warm}); медіана по '
          f'повторах усередині мережі, потім по мережах\n')
    print(f'{"варіант":26s} {"R, мс":>8s} {"ΔR, мс":>8s} {"виграш":>9s} '
          f'{"мереж кращих":>13s}')
    print('-' * 72)
    for name, _ in VAR:
        dR = [per[name][i][0] - per[a0][i][0] for i in range(nets)]
        gn = [per[a0][i][1] / per[name][i][1] for i in range(nets)]
        win = sum(1 for q in gn if q > 1.0)
        print(f'{name:26s} {st.median([q[0] for q in per[name]])*1000:7.3f} '
              f'{st.median(dR)*1000:+8.3f} {st.median(gn):8.4f}× '
              f'{win:8d}/{nets}')
    print('-' * 72)
    lo = min(min(acc[k]['R']) for k, _ in VAR)
    hi = max(max(acc[k]['R']) for k, _ in VAR)
    R0 = st.median([q[0] for q in per[a0]])
    print(f'розкид R по ВСІХ прогонах: {lo*1000:.2f}–{hi*1000:.2f} мс ⟹ '
          f'🚨вимірні лише зміни ≳{0.05*R0*1000:.2f} мс')
    print('⛔B relu `out=` (+4.5 мс) і ⛔C обрізання `ix_` (+1.8 мс) уже '
          'відкинуті прогоном 15×1: ОБИДВА ПОГІРШУЮТЬ.')


if __name__ == '__main__':
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 11,
         nets=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
