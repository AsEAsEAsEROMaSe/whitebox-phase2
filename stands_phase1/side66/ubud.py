"""ЧИ РОЗМИВАЄТЬСЯ RESIDUAL ПРИ БІЛЬШОМУ БЮДЖЕТІ? — гейт лінії «бюджет ×4». (С74)

ЗВІДКИ. Крива `score(u) = (1.89/u + 0.02)e−06 × max(0.1, 0.0942u + r)` обіцяє
1.067× при `u=4` ЛИШЕ тому, що `r` (residual-частка, 0.0131) вважається
ФІКСОВАНОЮ накладною: при вчетверо більшому `u` вона розмивається з 12% до 3.4%
множника. Це ПРИПУЩЕННЯ, і воно ніколи не перевірялось.

🚨 РИЗИК, ЯКИЙ ЛАМАЄ ВСЕ. Фазова карта С73 показала, що 27% residual — це
`звільнення локальних` (3.32 мс), а розмір локальних масивів ПРЯМО пропорційний
`n_pairs`. Якщо residual росте ∝ `u`, множник = `0.1073u`, а score =
`0.2028 + 0.00215u` — РОСТЕ ⟹ лінія не просто мертва, вона шкідлива.

📐 ОЧІКУВАННЯ ДО ПРОГОНУ (гейт). Фази `обрізання ваг`, `замикання` (матриці
256×256), `склейка` від `n_pairs` НЕ залежать — це ~половина residual. Решта
(генерація, ZCA, звільнення) залежить. ⟹ прогноз: residual 12.5 мс → **20–30 мс**
при ×4, тобто СУБЛІНІЙНО.
    r(×4) ≤ 0.0286 (residual ≤ 28 мс)  ⟹ виграш ≥1.03× — ЛІНІЯ ЖИВА
    r(×4) >  0.0286                    ⟹ ЗАКРИТО за ціною

ПРИЛАД (той самий, що в `phases.py`): residual = wall − backend − overhead,
пофазно через накопичувальні лічильники `BudgetContext`. Множник рахується
НАПРЯМУ: `F/B + λR/B` — без моделі, без кривої.

ПЛОМБИ
  ✅ при `u=1` вихід РІВНО == `estimator_sub72.predict` (max|Δ|=0) і множник
     мусить лягти біля грейдерних 0.1073;
  ✅ Σ фазових residual == загальний residual (розбіжність < 50 мкс);
  ✅ флопи мусять рости РІВНО в `u` разів (детерміновано) — якщо ні, `u` не
     доїхав до вибірки;
  🔀 усі `u` ЧЕРГУЮТЬСЯ всередині кожного повтору (без цього стенд міряє ДРЕЙФ —
     спіймано в С73 на `cheap.py`).
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
UGRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

# ── константи sub72 (звірено з estimator_sub72.py) ──
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


class Probe:
    """Знімок накопичувальних лічильників між фазами."""

    def __init__(self, ctx):
        self.c = ctx
        self.res, self.flp, self.ops = {}, {}, {}
        self._t = time.perf_counter()
        self._b = ctx.flopscope_backend_time_s
        self._o = ctx.flopscope_overhead_time_s
        self._f = ctx.flops_used
        self._n = len(ctx._op_log)

    def mark(self, name):
        t = time.perf_counter()
        b = self.c.flopscope_backend_time_s
        o = self.c.flopscope_overhead_time_s
        f = self.c.flops_used
        n = len(self.c._op_log)
        self.res[name] = (self.res.get(name, 0.0) + (t - self._t)
                          - (b - self._b) - (o - self._o))
        self.flp[name] = self.flp.get(name, 0) + (f - self._f)
        self.ops[name] = self.ops.get(name, 0) + (n - self._n)
        self._t, self._b, self._o, self._f, self._n = t, b, o, f, n


def predict_probed(mlp, budget, pr, util):
    """ДОСЛІВНА копія `predict` sub72 + `pr.mark(...)`; `util` — цільова частка."""
    width = mlp.width
    depth = len(mlp.weights)
    per_sample_full = 2.0 * depth * width * width
    rng = fnp.random.default_rng(_SEED)

    zp = fnp.array(rng.standard_normal((_PILOT_PAIRS, width)).astype(fnp.float32))
    xp = fnp.concatenate([zp, -zp], axis=0)
    pr.mark('1 пілот: генерація z')
    alive = []
    a = xp
    for i in range(depth):
        h = fnp.matmul(a, mlp.weights[i])
        if i < depth - 1:
            alive.append(fnp.max(h, axis=0) >= 0.0)
        a = fnp.maximum(h, 0.0)
    pr.mark('2 пілот: прохід 32 шари')

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
    pr.mark('3 обрізання ваг')

    used = 2.0 * _PILOT_PAIRS * per_sample_full
    left = util * float(budget) - used
    n_pairs = max(_MIN_PAIRS, int(left / (2.0 * max(per_sample_cut, 1.0))))

    z = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
    x = fnp.concatenate([z, -z], axis=0)
    pr.mark('4 генерація вибірки')
    cov = fnp.matmul(x.T, x) / float(2 * n_pairs)
    evals, evecs = fnp.linalg.eigh(cov)
    inv_sqrt = 1.0 / fnp.sqrt(fnp.maximum(evals, 1e-8))
    whiten = fnp.matmul(evecs * inv_sqrt, evecs.T)
    pr.mark('5 ZCA-вибілювання')

    denom = float(2 * n_pairs)
    zrow = fnp.zeros(width, dtype=fnp.float32)
    out = []

    ksig = _KSIG if _KSIG <= depth - 1 else depth - 1
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
            ph = _phi(t)
            Ph = _Phi(t)
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
    pr.mark('6 замикання (не залежить від n)')

    w0 = fnp.matmul(whiten, w_cut[0])
    x = fnp.maximum(fnp.matmul(x, w0), 0.0)
    out.append(zrow if depth > 1 else fnp.sum(x, axis=0) / denom)
    sg0 = fnp.sqrt(fnp.maximum(fnp.sum(w_cut[0] * w_cut[0], axis=0), 1e-30))
    v = fnp.sum(x, axis=0) / denom - sg0 * _INV_SQRT_2PI
    pr.mark('7 шар 0 + КЗ входу')

    for i in range(1, depth):
        p = fnp.matmul(x, w_cut[i])
        pr.mark('8a цикл: matmul x·W')
        gate = fnp.sum((p[:_GATE_N] > 0.0).astype(p.dtype), axis=0) * _INV_GATE
        pr.mark('8b цикл: затвор')
        v = fnp.matmul(v, w_cut[i]) * gate
        pr.mark('8c цикл: перенос v')
        if i <= ksig:
            mh = fnp.sum(p, axis=0) / denom
            sh = fnp.sqrt(fnp.maximum(
                fnp.sum(p * p, axis=0) / denom - mh * mh, 1e-30))
            x = fnp.maximum(p, 0.0)
            xm = fnp.sum(x, axis=0) / denom
            xq = fnp.sum(x * x, axis=0) / denom
            v = (v + (_LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + _LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + _LAM_Q[i - 1] * (xq - q_cl[i - 1])
                 + _LAM_G[i - 1] * (gate - P_cl[i - 1]))
            pr.mark('8d цикл: 4 канали (шари 1..8)')
        else:
            x = fnp.maximum(p, 0.0)
            pr.mark('8e цикл: relu (шари 9..31)')
        out.append(fnp.sum(x, axis=0) / denom if i == depth - 1 else zrow)
        pr.mark('8f цикл: вихід')

    out[-1] = out[-1] - v
    r = fnp.stack(out, axis=0)
    pr.mark('9 склейка')
    return r, n_pairs


def main(reps=7, warm=2):
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

    # ── ПЛОМБА 1: копія при u=1 == справжній сабміт, БІТ У БІТ ──
    Est = load_estimator(SK + '/estimator_sub72.py')
    with flops.BudgetContext(flop_budget=B, quiet=True) as c:
        ref = np.asarray(Est().predict(mlp, B))
    f_ref = c.flops_used
    with flops.BudgetContext(flop_budget=B, quiet=True) as c2:
        got, np1 = predict_probed(mlp, B, Probe(c2), _BASE_UTIL)
        got = np.asarray(got)
    dmax = float(np.abs(got - ref).max())
    print(f'✅ПЛОМБА 1 (копія == sub72):  max|Δ| = {dmax:.3e}  '
          f'{"OK" if dmax == 0.0 else "🛑РОЗБІЖНІСТЬ"}')
    print(f'✅ПЛОМБА 3 (флопи):  довідка {f_ref:.6e}  копія {c2.flops_used:.6e}  '
          f'{"OK" if f_ref == c2.flops_used else "🛑"}')
    print(f'   n_pairs при u=1: {np1}\n')

    for _ in range(warm):
        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            predict_probed(mlp, B, Probe(c), _BASE_UTIL)

    runs = {u: [] for u in UGRID}
    tot = {u: [] for u in UGRID}
    seal = {u: [] for u in UGRID}
    NP, F = {}, {}
    for _ in range(reps):
        for u in UGRID:                       # 🔀 ЧЕРГУВАННЯ всередині повтору
            with flops.BudgetContext(flop_budget=B * 20, quiet=True) as c:
                pr = Probe(c)
                _r, npr = predict_probed(mlp, B, pr, _BASE_UTIL * u)
                # 🔑 звільнення ЛОКАЛЬНИХ — на `return`, ВСЕРЕДИНІ контексту
                pr.mark('10 звільнення локальних')
            runs[u].append(dict(pr.res))
            tot[u].append(c.summary_dict()['residual_wall_time_s'])
            seal[u].append(sum(pr.res.values()))
            F[u], NP[u] = c.flops_used, npr

    for u in UGRID:
        ms, mt = st.median(seal[u]), st.median(tot[u])
        print(f'✅ПЛОМБА 2 (u={u:g}): Σфаз {ms*1000:.3f} проти R {mt*1000:.3f} мс'
              f'   Δ={abs(ms-mt)*1e6:.0f} мкс  '
              f'{"OK" if abs(ms - mt) < 50e-6 else "🛑"}')
    r0 = F[1.0] / B
    print(f'✅ПЛОМБА 4 (флопи ∝ u): ' + '  '.join(
        f'u={u:g}→{F[u]/F[1.0]:.3f}×' for u in UGRID))

    print(f'\n{reps} повторів (медіана), мережа 256×32\n')
    names = list(runs[UGRID[0]][0].keys())
    hdr = ' '.join(f'{"u=" + format(u, "g"):>9s}' for u in UGRID)
    print(f'{"фаза, мс":34s} {hdr}  {"×4/×1":>7s}')
    print('-' * 78)
    med = {u: {n: st.median([r[n] for r in runs[u]]) for n in names}
           for u in UGRID}
    for n in sorted(names, key=lambda q: -med[UGRID[-1]][q]):
        cells = ' '.join(f'{med[u][n]*1000:9.3f}' for u in UGRID)
        rt = med[UGRID[-1]][n] / max(med[UGRID[0]][n], 1e-9)
        print(f'{n:34s} {cells}  {rt:6.2f}×')
    print('-' * 78)
    tm = {u: st.median(tot[u]) for u in UGRID}
    cells = ' '.join(f'{tm[u]*1000:9.3f}' for u in UGRID)
    print(f'{"РАЗОМ residual":34s} {cells}  '
          f'{tm[UGRID[-1]]/tm[UGRID[0]]:6.2f}×')
    print(f'{"n_pairs":34s} ' + ' '.join(f'{NP[u]:9d}' for u in UGRID))

    print(f'\n{"u":>4s} {"F/B":>8s} {"λR/B":>8s} {"МНОЖНИК":>9s} '
          f'{"MSE-модель":>11s} {"БАЛ":>11s} {"виграш":>8s}')
    print('-' * 66)
    base_score = None
    for u in UGRID:
        fb = F[u] / B
        rb = LAM * tm[u] / B
        mult = max(0.1, fb + rb)
        mse = (1.89 / u + 0.02)          # e-06, крива з виміряним зміщенням
        sc = mse * mult                  # e-07 після ×1e-06×10
        if base_score is None:
            base_score = sc
        print(f'{u:4g} {fb:8.4f} {rb:8.4f} {mult:9.4f} {mse:10.4f}e-06 '
              f'{sc:10.4f}e-07 {base_score/sc:7.3f}×')
    print('-' * 66)
    print('🎯 ГЕЙТ: residual при u=4 ≤ 28 мс ⟹ виграш ≥1.03×, лінія ЖИВА.\n'
          '   Якщо residual росте ∝ u — множник теж ∝ u ⟹ бал РОСТЕ, '
          'лінія ШКІДЛИВА.\n'
          '🚨 MSE тут — МОДЕЛЬ (крива), не вимір. Реальний MSE при u=4 і '
          'перефічених `λ` — окремий стенд.')
    al = sum(tot.values(), [])
    print(f'розкид R по всіх u: {min(al)*1000:.2f}–{max(al)*1000:.2f} мс')


if __name__ == '__main__':
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 7)
