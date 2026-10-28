"""ДЕ САМЕ СИДЯТЬ 20 мс RESIDUAL — пофазний профіль ДІЮЧОГО САБМІТУ. (С73)

ЗВІДКИ. `finding-residual-is-python-time-and-measurable`: ціна грейдера
`C = F + λ·R`, λ=1e11 за секунду, і зміряно `F/B=0.1018` + `λR/B=0.0067..0.0092`
⟹ множник 0.1085 ≈ грейдерний 0.1073. **8.5% множника — це чистий python-час.**
Але АДРЕСИ немає: цикл по 31 шару = лише 2.2 мс з 25, решта не локалізована.
⛔Профіль через проміжки між записами `op_log` дав НЕДОСТОВІРНІ дані (сума
проміжків −17.5 мс — операції перекриваються, бо `op_log` пише час ПІСЛЯ).

ПРИЛАД (перевірено читанням `flopscope/_budget.py`):
    residual = wall − backend − overhead                     (`_timing_summary`)
    `_total_flopscope_backend_time` / `_total_flopscope_overhead_time` —
    НАКОПИЧУВАЛЬНІ поля BudgetContext, доступні як властивості під час роботи.
⟹ пофазний residual = Δperf_counter − Δbackend − Δoverhead. Вкладені
BudgetContext НЕ потрібні (і були б неправильні: вони обнулили б бюджет).

ПЛОМБИ
  ✅ вихід інструментованої копії РІВНО == `estimator_sub72.predict` (max|Δ|=0);
  ✅ Σ фазових residual == загальний residual (тотожність, розбіжність < 50 мкс);
  ✅ Σ фазових флопів == `flops_used`.
🚨 ПРОГРІВ 2 прогони; медіана з ≥7; розкид residual 17.4–25.9 мс ⟹ зміни <5%
   НЕВИМІРНІ (див. finding).
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

# ── константи sub72 (скопійовані ДОСЛІВНО; звірено з estimator_sub72.py) ──
_TARGET_UTIL = 0.0942
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
        self.res = {}
        self.flp = {}
        self.ops = {}
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
        self.res[name] = self.res.get(name, 0.0) + (t - self._t) - (b - self._b) - (o - self._o)
        self.flp[name] = self.flp.get(name, 0) + (f - self._f)
        self.ops[name] = self.ops.get(name, 0) + (n - self._n)
        self._t, self._b, self._o, self._f, self._n = t, b, o, f, n


def predict_probed(mlp, budget, pr, v82=False):
    """ДОСЛІВНА копія `predict` + `pr.mark(...)`; `v82` — правки sub82."""
    width = mlp.width
    depth = len(mlp.weights)
    per_sample_full = 2.0 * depth * width * width
    rng = fnp.random.default_rng(_SEED)

    # 1) ПІЛОТ
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

    # 2) обрізані ваги
    w_cut = []
    if v82:
        idx = [fnp.flatnonzero(mk) for mk in alive]
    for i in range(depth):
        wi = mlp.weights[i]
        if v82:
            if i > 0:
                wi = fnp.take(wi, idx[i - 1], axis=0)
            if i < depth - 1:
                wi = fnp.take(wi, idx[i], axis=1)
        else:
            if i > 0:
                wi = wi[alive[i - 1], :]
            if i < depth - 1:
                wi = wi[:, alive[i]]
        w_cut.append(wi)
    if v82:
        del zp, xp, a, h, idx
    per_sample_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in w_cut)
    pr.mark('3 обрізання ваг')

    # 3) основна вибірка
    used = 2.0 * _PILOT_PAIRS * per_sample_full
    left = _TARGET_UTIL * float(budget) - used
    n_pairs = max(_MIN_PAIRS, int(left / (2.0 * max(per_sample_cut, 1.0))))

    z = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
    x = fnp.concatenate([z, -z], axis=0)
    if v82:
        del z
    pr.mark('4 генерація вибірки')
    cov = fnp.matmul(x.T, x) / float(2 * n_pairs)
    evals, evecs = fnp.linalg.eigh(cov)
    inv_sqrt = 1.0 / fnp.sqrt(fnp.maximum(evals, 1e-8))
    whiten = fnp.matmul(evecs * inv_sqrt, evecs.T)
    if v82:
        del cov, evals, evecs, inv_sqrt
    pr.mark('5 ZCA-вибілювання')

    denom = float(2 * n_pairs)
    zrow = fnp.zeros(width, dtype=fnp.float32)
    out = []

    ksig = _KSIG if _KSIG <= depth - 1 else depth - 1
    if ksig >= 1:
        # ── _closure_parts, розгорнуте для пофазного розрізу ──
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
            pr.mark('6a замикання: mat + φΦ')
            m = mp * Ph + s * ph
            C = cp * fnp.outer(Ph, Ph) + (0.5 * (cp * cp)) * fnp.outer(ph / s, ph / s)
            d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
            C = C + fnp.diag(d - fnp.diagonal(C))
            if i >= 1:
                m_cl.append(m)
                q_cl.append(fnp.diagonal(C) + m * m)
            pr.mark('6b замикання: оновлення C')
    else:
        s_cl, m_cl, q_cl, P_cl = [], [], [], []
    pr.mark('6c замикання: решта')

    w0 = fnp.matmul(whiten, w_cut[0])
    x = fnp.maximum(fnp.matmul(x, w0), 0.0)
    if v82:
        del whiten, w0
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
    return r


def main(reps=9, warm=2):
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

    # ── ПЛОМБА 1: інструментована копія == справжній сабміт, БІТ У БІТ ──
    Est = load_estimator(SK + '/estimator_sub72.py')
    with flops.BudgetContext(flop_budget=B, quiet=True) as c:
        ref = np.asarray(Est().predict(mlp, B))
    f_ref, r_ref = c.flops_used, c.summary_dict()['residual_wall_time_s']
    with flops.BudgetContext(flop_budget=B, quiet=True) as c2:
        p2 = Probe(c2)
        got = np.asarray(predict_probed(mlp, B, p2))
    dmax = float(np.abs(got - ref).max())
    print(f'✅ПЛОМБА 1 (копія == sub72):  max|Δ| = {dmax:.3e}   '
          f'{"OK" if dmax == 0.0 else "🛑РОЗБІЖНІСТЬ"}')
    print(f'✅ПЛОМБА 3 (флопи):  довідка {f_ref:.6e}  копія {c2.flops_used:.6e}  '
          f'{"OK" if f_ref == c2.flops_used else "🛑"}')
    print(f'   (довідковий residual одного прогону: {r_ref*1000:.2f} мс)\n')

    # ── прогріви ──
    for _ in range(warm):
        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            predict_probed(mlp, B, Probe(c))

    # 🔀 ОБИДВА варіанти в кожному повторі (без чергування стенд міряє ДРЕЙФ)
    runs = {False: [], True: []}
    tot = {False: [], True: []}
    seal = {False: [], True: []}
    OPS, F = {}, {}
    for _ in range(reps):
        for v82 in (False, True):
            with flops.BudgetContext(flop_budget=B, quiet=True) as c:
                pr = Probe(c)
                _r = predict_probed(mlp, B, pr, v82=v82)
                # 🔑 звільнення ЛОКАЛЬНИХ масивів стається на `return`, тобто
                #    ПІСЛЯ останнього mark, але ВСЕРЕДИНІ контексту
                pr.mark('10 звільнення локальних')
            d = c.summary_dict()
            runs[v82].append(dict(pr.res))
            tot[v82].append(d['residual_wall_time_s'])
            seal[v82].append(sum(pr.res.values()))
            F[v82], OPS[v82] = c.flops_used, dict(pr.ops)

    for v82 in (False, True):
        ms, mt = st.median(seal[v82]), st.median(tot[v82])
        print(f'✅ПЛОМБА 2 ({"sub82" if v82 else "sub72"}): Σфаз {ms*1000:.3f} '
              f'проти R {mt*1000:.3f} мс   Δ={abs(ms-mt)*1e6:.0f} мкс  '
              f'{"OK" if abs(ms-mt) < 50e-6 else "🛑"}')
    m0, m1 = st.median(tot[False]), st.median(tot[True])
    print(f'\n{reps} повторів (медіана), мережа 256×32, F/B={F[False]/B:.4f}\n')
    print(f'{"фаза":34s} {"sub72":>8s} {"sub82":>8s} {"Δ, мс":>8s} '
          f'{"% sub82":>8s} {"опер.":>7s}')
    print('-' * 80)
    names = list(runs[False][0].keys())
    rows = sorted(((n, st.median([r[n] for r in runs[False]]),
                    st.median([r[n] for r in runs[True]])) for n in names),
                  key=lambda q: -q[2])
    for n, a, b in rows:
        print(f'{n:34s} {a*1000:7.3f} {b*1000:8.3f} {(b-a)*1000:+8.3f} '
              f'{100*b/m1:7.1f}% {OPS[True][n]:7d}')
    print('-' * 80)
    print(f'{"РАЗОМ":34s} {m0*1000:7.3f} {m1*1000:8.3f} {(m1-m0)*1000:+8.3f} '
          f'{100.0:7.1f}% {sum(OPS[True].values()):7d}')
    print(f'\nмножник: sub72 {max(0.1, F[False]/B + LAM*m0/B):.5f}  →  '
          f'sub82 {max(0.1, F[True]/B + LAM*m1/B):.5f}   '
          f'({max(0.1, F[False]/B+LAM*m0/B)/max(0.1, F[True]/B+LAM*m1/B):.4f}×)')
    al = tot[False] + tot[True]
    print(f'розкид R: {min(al)*1000:.2f}–{max(al)*1000:.2f} мс ⟹ '
          f'🚨зміни менші за {0.05*m0*1000:.2f} мс НЕВИМІРНІ')


if __name__ == '__main__':
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 9)
