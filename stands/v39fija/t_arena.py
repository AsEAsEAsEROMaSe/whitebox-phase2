# 18.10.2026 white-box-21: юніт-тест арен _Strassen (V45A) проти щільного matmul і проти старих пулів, малі форми
import importlib.util, sys, os
import numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
def load(arena):
    os.environ['V45A_ARENA'] = arena
    spec = importlib.util.spec_from_file_location('e' + arena, sys.argv[1])
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
import flopscope as flops, flopscope.numpy as fnp
rng = np.random.default_rng(0)
cases = [(256, 256, 256), (256, 224, 256), (256, 192, 224), (256, 256, 256), (224, 160, 256), (256, 224, 224), (192, 256, 192), (256, 256, 256)]
BATCH = [3, 5, 2, 3, 6, 1, 4, 5]
data = [(rng.standard_normal((bb, 1, m, kd)).astype(np.float32), rng.standard_normal((bb, 1, kd, w)).astype(np.float32),
         rng.standard_normal((bb, 1, w, kd)).astype(np.float32)) for (m, kd, w), bb in zip(cases, BATCH)]
res = {}
for arena in ('0', '1'):
    mod = load(arena)
    with flops.BudgetContext(flop_budget=10**13, quiet=True) if hasattr(flops, 'BudgetContext') else open(os.devnull) as _:
        S = mod._Strassen(fnp.float32, 6)
        out = []
        for (m, kd, w), (X, Y, Yh) in zip(cases, data):
            Xf, Yf, Yhf = fnp.asarray(X), fnp.asarray(Y), fnp.asarray(Yh)
            o = fnp.reshape(fnp.zeros((X.shape[0] * m * w,), dtype=fnp.float32), (X.shape[0], 1, m, w))
            S.mm(Xf, Yf, o, 3)
            h = fnp.reshape(fnp.zeros((m * w,), dtype=fnp.float32), (1, m, w))
            S.hub(Xf, Yhf, h, 3)
            out.append((np.asarray(o).copy(), np.asarray(h).copy()))
        res[arena] = out
        print('arena', arena, 'ключів пулу', len(S.pool), 'арен', len(S.arena))
for i, ((m, kd, w), (X, Y, Yh)) in enumerate(zip(cases, data)):
    ref = X @ Y; refh = np.einsum('kpij,kpcj->pic', X, Yh)
    e0 = np.abs(res['0'][i][0] - ref).max(); e1 = np.abs(res['1'][i][0] - ref).max()
    h0 = np.abs(res['0'][i][1] - refh).max(); h1 = np.abs(res['1'][i][1] - refh).max()
    same = np.array_equal(res['0'][i][0], res['1'][i][0]) and np.array_equal(res['0'][i][1], res['1'][i][1])
    print(cases[i], 'mm err old %.2e new %.2e · hub err old %.2e new %.2e · біт у біт %s' % (e0, e1, h0, h1, same))
