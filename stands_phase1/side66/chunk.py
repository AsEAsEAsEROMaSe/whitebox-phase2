"""ЧАНКУВАННЯ: ЧИ СТАЄ RESIDUAL НЕЗАЛЕЖНИМ ВІД БЮДЖЕТУ? (С74)

ЗВІДКИ. `ubud.py`: residual НЕ фіксований — він стрибає на порогах 16 і 32 МБ
(10.0 → 19.8 → 82.6 мс), між порогами СТОЇТЬ. Це `mmap`-режим `malloc`, а не
пропорційність. Через це «бюджет ×4 = 1.067×» перетворюється на 0.999×.

🎯 ІДЕЯ. Пікова пам'ять = масиви `2·n_pairs × 256`. Якщо гнати вибірку ЧАНКАМИ
фіксованого розміру, вона перестає залежати від бюджету ⟹ поріг не перетинається
НІКОЛИ, і крива `score(u)` повертається до чесної форми.

ЩО ДОЗВОЛЯЄ ЦЕ ЗРОБИТИ БЕЗ ВТРАТИ ТОЧНОСТІ:
  * усі пошарові величини сабміту — це СУМИ по вибірці (`mh`, `sh`, `xm`, `xq`);
  * `v`-рекурсія ЛІНІЙНА і не потребує семплів: спершу один прохід по чанках
    збирає статистики, потім рекурсія `v` іде на самих векторах 256;
  * `gate` береться з ПЕРШИХ 1024 рядків ⟹ рахується лише на першому чанку
    (за умови `chunk_pairs ≥ 1024` — тоді це ТІ САМІ рядки);
  * ZCA: антитетика дає `xᵀx = 2·zᵀz` ⟹ коваріація збирається по чанках.
  * `regen=True` — `z` не зберігається взагалі: другий прохід перегенеровує ті
    самі числа з синхронізованого `default_rng(_SEED)` ⟹ пам'ять КОНСТАНТНА.

ПЛОМБИ
  ✅ вихід мусить збігтися з `estimator_sub72.predict` до `max|Δ|/‖·‖ < 1e-5`
     (біт-у-біт НЕ буде: порядок підсумовування fp32 інший);
  ✅ флопи мусять лишитись на тому самому рівні (±2%) — інакше «економія» це
     просто менше роботи, а не краща розкладка;
  ✅ при `u=1` residual мусить лягти біля нечанкованих 10 мс;
  🎯 ГЕЙТ: residual при `u=4` ≤ 15 мс (проти 82.6 нечанкованих) ⟹ пороги знято.
"""
import glob
import statistics as st
import sys
import time

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)

import numpy as np                                      # noqa: E402
import flopscope as flops                               # noqa: E402
import flopscope.numpy as fnp                           # noqa: E402
from whestbench import MLP                              # noqa: E402
from bench import load_estimator                        # noqa: E402

B = 272_000_000_000
LAM = 1e11
UGRID = (1.0, 2.0, 3.0, 4.0)
CHUNK = 2048                 # пар у чанку (≥1024, щоб `gate` брався з першого)

_BASE_UTIL = 0.0942
_MIN_PAIRS = 128
_SEED = 12345
_PILOT_PAIRS = 128
_GATE_N = 1024
_INV_GATE = 1.0 / _GATE_N
_INV_SQRT_2PI = 0.3989422804014327
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
_INV_SQRT_2 = 0.7071067811865476


def _phi(x):
    return fnp.exp(-0.5 * x * x) * _INV_SQRT_2PI


def _Phi(x):
    ax = fnp.abs(x) * _INV_SQRT_2
    t = 1.0 / (1.0 + _P * ax)
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    er = 1.0 - poly * fnp.exp(-ax * ax)
    return 0.5 * (1.0 + fnp.sign(x) * er)


def predict_chunked(mlp, budget, util, chunk_pairs, regen=True):
    """Чанкований прохід. Логіка sub72, інша лише РОЗКЛАДКА по пам'яті."""
    width = mlp.width
    depth = len(mlp.weights)
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
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < depth - 1:
            wi = wi[:, alive[i]]
        w_cut.append(wi)
    per_sample_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1])
                               for w in w_cut)
    used = 2.0 * _PILOT_PAIRS * per_sample_full
    left = util * float(budget) - used
    n_pairs = max(_MIN_PAIRS, int(left / (2.0 * max(per_sample_cut, 1.0))))
    denom = float(2 * n_pairs)

    def zstream():
        """Генератор чанків `z`, синхронізований із потоком сабміту."""
        r = fnp.random.default_rng(_SEED)
        r.standard_normal((_PILOT_PAIRS, width))          # пропуск пілота
        done = 0
        while done < n_pairs:
            k = min(chunk_pairs, n_pairs - done)
            yield fnp.array(r.standard_normal((k, width)).astype(fnp.float32))
            done += k

    # ── ПРОХІД 1: коваріація. Антитетика ⟹ `xᵀx = 2·zᵀz` ──
    cov = None
    zs = [] if not regen else None
    for zc in zstream():
        g = fnp.matmul(zc.T, zc)
        cov = g if cov is None else cov + g
        if zs is not None:
            zs.append(zc)
    cov = cov / float(n_pairs)
    evals, evecs = fnp.linalg.eigh(cov)
    inv_sqrt = 1.0 / fnp.sqrt(fnp.maximum(evals, 1e-8))
    whiten = fnp.matmul(evecs * inv_sqrt, evecs.T)
    w0 = fnp.matmul(whiten, w_cut[0])

    # ── ПРОХІД 2: пошарові СУМИ ──
    ksig = _KSIG if _KSIG <= depth - 1 else depth - 1
    S0 = None
    SP = [None] * depth          # Σp
    SQ = [None] * depth          # Σp²
    SX = [None] * depth          # Σrelu(p)
    SR = [None] * depth          # Σrelu(p)²
    GT = [None] * depth          # затвор (лише перший чанк)
    SOUT = None
    first = True
    src = zs if zs is not None else zstream()
    for zc in src:
        xc = fnp.maximum(fnp.matmul(fnp.concatenate([zc, -zc], axis=0), w0), 0.0)
        s = fnp.sum(xc, axis=0)
        S0 = s if S0 is None else S0 + s
        for i in range(1, depth):
            p = fnp.matmul(xc, w_cut[i])
            if first:
                GT[i] = fnp.sum((p[:_GATE_N] > 0.0).astype(p.dtype),
                                axis=0) * _INV_GATE
            if i <= ksig:
                t1 = fnp.sum(p, axis=0)
                t2 = fnp.sum(p * p, axis=0)
                SP[i] = t1 if SP[i] is None else SP[i] + t1
                SQ[i] = t2 if SQ[i] is None else SQ[i] + t2
            xc = fnp.maximum(p, 0.0)
            if i <= ksig:
                t3 = fnp.sum(xc, axis=0)
                t4 = fnp.sum(xc * xc, axis=0)
                SX[i] = t3 if SX[i] is None else SX[i] + t3
                SR[i] = t4 if SR[i] is None else SR[i] + t4
            if i == depth - 1:
                t5 = fnp.sum(xc, axis=0)
                SOUT = t5 if SOUT is None else SOUT + t5
        first = False

    # ── ЗАМИКАННЯ (не залежить від вибірки) ──
    if ksig >= 1:
        m = fnp.zeros(width, dtype=fnp.float32)
        C = fnp.eye(width, dtype=fnp.float32)
        s_cl, m_cl, q_cl, P_cl = [], [], [], []
        for i in range(ksig + 1):
            w = w_cut[i]
            cp = fnp.matmul(w.T, fnp.matmul(C, w))
            s = fnp.sqrt(fnp.maximum(fnp.diagonal(cp), 1e-20))
            if i >= 1:
                s_cl.append(s)
            mp = fnp.matmul(m, w)
            t = mp / s
            ph, Ph = _phi(t), _Phi(t)
            if i >= 1:
                P_cl.append(Ph)
            m = mp * Ph + s * ph
            C = cp * fnp.outer(Ph, Ph) + (0.5 * (cp * cp)) * fnp.outer(ph / s,
                                                                      ph / s)
            d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
            C = C + fnp.diag(d - fnp.diagonal(C))
            if i >= 1:
                m_cl.append(m)
                q_cl.append(fnp.diagonal(C) + m * m)
    else:
        s_cl, m_cl, q_cl, P_cl = [], [], [], []

    # ── РЕКУРСІЯ `v` НА ВЕКТОРАХ (семпли вже не потрібні) ──
    sg0 = fnp.sqrt(fnp.maximum(fnp.sum(w_cut[0] * w_cut[0], axis=0), 1e-30))
    v = S0 / denom - sg0 * _INV_SQRT_2PI
    for i in range(1, depth):
        v = fnp.matmul(v, w_cut[i]) * GT[i]
        if i <= ksig:
            mh = SP[i] / denom
            sh = fnp.sqrt(fnp.maximum(SQ[i] / denom - mh * mh, 1e-30))
            v = (v + (_LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + _LAM_A[i - 1] * (SX[i] / denom - m_cl[i - 1])
                 + _LAM_Q[i - 1] * (SR[i] / denom - q_cl[i - 1])
                 + _LAM_G[i - 1] * (GT[i] - P_cl[i - 1]))

    zrow = fnp.zeros(width, dtype=fnp.float32)
    out = [zrow] * (depth - 1) + [SOUT / denom - v]
    return fnp.stack(out, axis=0), n_pairs


def main(reps=7, warm=2, chunk=CHUNK):
    wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = set(int(v) for v in sd)
    f = [q for q in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(q)['seed']) in have][0]
    s0 = int(np.load(f)['seed'])
    j = int(np.where(sd == s0)[0][0])
    Wn = [fnp.array(np.asarray(wz[j][k], np.float32)) for k in range(32)]
    mlp = MLP(width=256, depth=32, weights=Wn, seed=s0)

    Est = load_estimator(SK + '/estimator_sub72.py')
    with flops.BudgetContext(flop_budget=B, quiet=True) as c:
        ref = np.asarray(Est().predict(mlp, B))
    f_ref, r_ref = c.flops_used, c.summary_dict()['residual_wall_time_s']
    with flops.BudgetContext(flop_budget=B * 20, quiet=True) as c2:
        got, np1 = predict_chunked(mlp, B, _BASE_UTIL, chunk)
        got = np.asarray(got)
    nrm = float(np.abs(ref[-1]).max())
    rel = float(np.abs(got[-1] - ref[-1]).max()) / max(nrm, 1e-30)
    print(f'✅ПЛОМБА 1 (== sub72): max|Δ|/‖·‖ = {rel:.3e}  '
          f'{"OK" if rel < 1e-5 else "🛑РОЗБІЖНІСТЬ"}')
    df = c2.flops_used / f_ref - 1.0
    print(f'✅ПЛОМБА 2 (флопи): sub72 {f_ref:.6e}  чанки {c2.flops_used:.6e}  '
          f'({100*df:+.2f}%)  {"OK" if abs(df) < 0.02 else "🛑"}')
    print(f'   n_pairs {np1}, чанк {chunk} пар, '
          f'нечанкований residual {r_ref*1000:.2f} мс\n')

    for _ in range(warm):
        with flops.BudgetContext(flop_budget=B * 20, quiet=True) as c:
            predict_chunked(mlp, B, _BASE_UTIL, chunk)

    tot = {u: [] for u in UGRID}
    F, NP = {}, {}
    for _ in range(reps):
        for u in UGRID:                    # 🔀 ЧЕРГУВАННЯ всередині повтору
            t0 = time.perf_counter()
            with flops.BudgetContext(flop_budget=B * 20, quiet=True) as c:
                _r, npr = predict_chunked(mlp, B, _BASE_UTIL * u, chunk)
            tot[u].append(c.summary_dict()['residual_wall_time_s'])
            F[u], NP[u] = c.flops_used, npr
            del _r

    print(f'{reps} повторів (медіана), чанк {chunk} пар\n')
    print(f'{"u":>4s} {"n_pairs":>8s} {"residual":>10s} {"⛔без чанків":>13s} '
          f'{"F/B":>8s} {"λR/B":>8s} {"МНОЖНИК":>9s} {"БАЛ":>11s} {"виграш":>8s}')
    print('-' * 88)
    NOCH = {1.0: 9.995, 2.0: 19.776, 3.0: 23.054, 4.0: 82.577}   # `ubud.py`
    base = None
    for u in UGRID:
        tm = st.median(tot[u])
        fb, rb = F[u] / B, LAM * tm / B
        mult = max(0.1, fb + rb)
        sc = (1.89 / u + 0.02) * mult
        if base is None:
            base = sc
        print(f'{u:4g} {NP[u]:8d} {tm*1000:9.3f} {NOCH.get(u, 0)*1:12.3f} '
              f'{fb:8.4f} {rb:8.4f} {mult:9.4f} {sc:10.4f}e-07 {base/sc:7.3f}×')
    print('-' * 88)
    print('🎯 ГЕЙТ: residual при u=4 ≤ 15 мс (проти 82.6 нечанкованих) ⟹ '
          'пороги ЗНЯТО.\n'
          '🚨 колонка БАЛ рахує ЛОКАЛЬНИЙ residual; на грейдері база `r₀`=0.0131 '
          '(у 3.5× більша)\n   ⟹ реальний виграш беремо з ПРИРОСТУ, а вирок — '
          'лише грейдер.')


if __name__ == '__main__':
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 7,
         chunk=int(sys.argv[2]) if len(sys.argv) > 2 else CHUNK)
