"""П3: sub91 при w=0 і util=0.0908 мусить бути БІТ У БІТ = sub87. (С77)"""
import glob, sys
SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)
import numpy as np, flopscope as flops, flopscope.numpy as fnp
from whestbench import MLP
import importlib.util
B = 272_000_000_000
def lm(p):
    s = importlib.util.spec_from_file_location(p.split('/')[-1][:-3], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m87 = lm(SK + '/estimator_sub87.py'); m91 = lm(SK + '/estimator_sub91.py')
m91._W_BLEND = 0.0; m91._TARGET_UTIL = m87._TARGET_UTIL
wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = set(int(v) for v in sd)
fs = [q for q in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(q)['seed']) in have][24:28]
dm = 0.0
for f in fs:
    zz = np.load(f); s0 = int(zz['seed']); j = int(np.where(sd == s0)[0][0])
    Wn = [fnp.array(np.asarray(wz[j][k], np.float32)) for k in range(32)]
    mlp = MLP(width=256, depth=32, weights=Wn, seed=s0)
    with flops.BudgetContext(flop_budget=B, quiet=True):
        a = np.asarray(m87.Estimator().predict(mlp, B))[-1].astype(np.float64)
    with flops.BudgetContext(flop_budget=B, quiet=True):
        b = np.asarray(m91.Estimator().predict(mlp, B))[-1].astype(np.float64)
    d = float(np.abs(a - b).max()); dm = max(dm, d)
    print(f'  {s0}: max|Δ| = {d:.3e}')
print(f'\nП3: max|Δ| по 4 мережах = {dm:.3e}  ' +
      ('✅ БІТ У БІТ' if dm == 0.0 else '🚨 РОЗБІЖНІСТЬ'))
