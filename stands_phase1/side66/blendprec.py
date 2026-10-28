"""fp32 (як у сабміті) проти fp64: чи псує точність замикання накопичення. (С77)"""
import glob
import sys

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)
import numpy as np                                      # noqa: E402
import flopscope as flops                               # noqa: E402
import flopscope.numpy as fnp                           # noqa: E402
from whestbench import MLP                              # noqa: E402
import importlib.util                                   # noqa: E402
from scipy.special import erf                           # noqa: E402

B = 272_000_000_000


def closure64(Wl):
    m = np.zeros(Wl[0].shape[0], dtype=np.float64)
    C = np.eye(Wl[0].shape[0], dtype=np.float64)
    for w in Wl:
        cp = w.T @ C @ w
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        t = mp / s
        ph = np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
        Ph = 0.5 * (1 + erf(t / np.sqrt(2)))
        m = mp * Ph + s * ph
        C = cp * np.outer(Ph, Ph) + 0.5 * (cp * cp) * np.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diag(C))
    return m


spec = importlib.util.spec_from_file_location('s91', SK + '/estimator_sub91.py')
m91 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m91)
m91._W_BLEND = 1.0

wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = set(int(v) for v in sd)
fs = [q for q in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(q)['seed']) in have][:int(sys.argv[1]) if len(sys.argv) > 1 else 4]

r32, r64, dif = [], [], []
for f in fs:
    zz = np.load(f)
    s0 = int(zz['seed'])
    truth = np.asarray(zz['a_true'])[-1].astype(np.float64)
    j = int(np.where(sd == s0)[0][0])
    Wn = [fnp.array(np.asarray(wz[j][k], np.float32)) for k in range(32)]
    mlp = MLP(width=256, depth=32, weights=Wn, seed=s0)
    with flops.BudgetContext(flop_budget=B, quiet=True):
        p1 = np.asarray(m91.Estimator().predict(mlp, B))[-1].astype(np.float64)
    M32 = p1 / (1.0 + m91._ALPHA)

    # ті самі ОБРІЗАНІ ваги — відтворюємо пілот сабміту
    rng = np.random.default_rng(m91._SEED)
    zp = rng.standard_normal((m91._PILOT_PAIRS, 256)).astype(np.float32)
    xp = np.concatenate([zp, -zp], 0)
    W = [np.asarray(wz[j][k], np.float32) for k in range(32)]
    alive, a = [], xp
    for i in range(32):
        h = a @ W[i]
        if i < 31:
            alive.append(h.max(0) >= 0.0)
        a = np.maximum(h, 0.0)
    wc = []
    for i in range(32):
        wi = W[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < 31:
            wi = wi[:, alive[i]]
        wc.append(wi.astype(np.float64))
    M64 = closure64(wc)
    n = np.linalg.norm(truth)
    r32.append(np.linalg.norm(M32 - truth) / n)
    r64.append(np.linalg.norm(M64 - truth) / n)
    dif.append(np.linalg.norm(M32 - M64) / n)
    print(f'{s0}: fp32 {r32[-1]*100:.4f}%  fp64 {r64[-1]*100:.4f}%  '
          f'‖fp32−fp64‖ {dif[-1]*100:.4f}%', flush=True)

rms = lambda v: float(np.sqrt((np.array(v) ** 2).mean()) * 100)
print(f'\nRMS: fp32 {rms(r32):.4f}%  fp64 {rms(r64):.4f}%  розбіжність {rms(dif):.4f}%')
