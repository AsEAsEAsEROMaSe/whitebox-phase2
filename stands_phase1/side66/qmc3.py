"""qmc3: ВИПРАВЛЕНЕ вирівнювання (100%, не 94%) + контроль ВИПАДКОВОЮ ротацією Хаара.

Наш власний вирок [[finding-qmc-sobol-loses-to-iid]] стояв на 3 МЕРЕЖАХ (нижче нашого
ж порогу 24) і сам називав, куди ще не ходили: «лише через рандомізовані ротації
(align u₁ з першою координатою) або lattice rules». Тут — саме це, на 24 мережах.

МІРЯЄМО ПРИ РІВНОМУ N І БЕЗ МОДЕЛІ ЦІНИ: якщо точковий набір не виграє ЗАДАРМА,
гілка мертва без жодних розмов про флопи.

📣 ПЕРЕДБАЧЕНО: усе ≈1.00× при вибілюванні УВІМК. Дві причини:
   (1) інтегранд РІВНО `‖z‖·f(ẑ)` (позитивна однорідність) — радіальна частина
       одновимірна й гладка, кутова живе на S^255 без виділених напрямків
       ⟹ ефективна розмірність ≈255, найгірший випадок для QMC;
   (2) антитетика знімає непарні степені, вибілювання — 2-й; QMC бере саме
       низькі порядки ⟹ ми вже забрали те, що взяв би QMC (evaaaz: «RQMC and
       exact-mean control variates are substitutes»).
   При вибілюванні ВИМК очікую малий виграш — і це буде доказом «замінники».
🟢 ЖИВЕ, якщо якийсь варіант ≥1.15× проти `iid` при вибілюванні УВІМК,
   і на сумі, і на медіані, і вище пломби `iid′`.
⛔ ПЛОМБА `iid′` — той самий жереб під іншим ім'ям: підлога роздільності стенда.
"""
import sys

import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import qmc

NETS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NP2 = 12                      # 🎓 степінь двійки ОБОВ'ЯЗКОВА для Sobol (0.64→0.84)
N = 1 << NP2                  # пар; семплів 2N = 8192 ≈ як у сабміті (npair≈4.3k)
D = 256
DEV = 'cuda:1'                # 🚨 P104, НЕ монітор

W_ALL = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SEEDS = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')

_PRIMES = []
_c = 2
while len(_PRIMES) < D:
    if all(_c % p for p in _PRIMES if p * p <= _c):
        _PRIMES.append(_c)
    _c += 1
_GEN = np.array([np.sqrt(p) % 1.0 for p in _PRIMES])       # Kronecker/Roberts


def _u2z(u):
    """[0,1)^d → N(0,I), з відступом від країв (ndtri(0)=-inf)."""
    return ndtri(np.clip(u, 1e-12, 1 - 1e-12))


def points(kind, rep_seed, rot):
    rng = np.random.default_rng(rep_seed)
    if kind in ('iid', "iid'"):
        return rng.standard_normal((N, D))
    if kind == 'sobol':
        z = _u2z(qmc.Sobol(d=D, scramble=True, seed=rep_seed).random(N))
    elif kind == 'latt':
        k = np.arange(N)[:, None]
        z = _u2z((k * _GEN[None, :] + rng.random(D)[None, :]) % 1.0)
    elif kind == 'sobol+rot':
        z = _u2z(qmc.Sobol(d=D, scramble=True, seed=rep_seed).random(N)) @ rot.T
    elif kind == 'latt+rot':
        k = np.arange(N)[:, None]
        z = _u2z((k * _GEN[None, :] + rng.random(D)[None, :]) % 1.0) @ rot.T
    elif kind == 'latt+haar':
        k = np.arange(N)[:, None]
        z = _u2z((k * _GEN[None, :] + rng.random(D)[None, :]) % 1.0)
        H, R = np.linalg.qr(rng.standard_normal((D, D)))
        z = z @ (H * np.sign(np.diag(R)))
    else:
        raise ValueError(kind)
    return z


def dominant_dir(W):
    """Напрямок у z-просторі, уздовж якого вихід відгукується найсильніше:
    E[∇_z Σ_j o_j] — зворотний прохід одиничного вектора крізь СЕРЕДНІ ворота."""
    g = torch.randn(4096, D, device=DEV)
    hs = []
    h = g
    for L in range(32):
        p = h @ W[L]
        hs.append((p > 0).float())
        h = torch.relu(p)
    b = torch.ones(4096, D, device=DEV)
    for L in range(31, -1, -1):
        b = (b * hs[L]) @ W[L].T
    b = b.mean(0)
    return (b / b.norm()).double().cpu().numpy()


def rot_from(bhat):
    """Ортогональна Q з b̂ у ПЕРШОМУ рядку ⟹ z@Q кладе домінантний напрямок
    у координату 0, куди Sobol/lattice дає найкращу рівномірність."""
    A = np.eye(D)
    A[0] = bhat
    Q, R = np.linalg.qr(A.T)
    return (Q * np.sign(np.diag(R)))          # (D,D), стовпці ортонормовані


def run(W, z_np, whiten):
    z = torch.from_numpy(z_np).float().to(DEV)
    z = torch.cat([z, -z], 0)                          # антитетика: середнє РІВНО 0
    if whiten:
        C = (z.T @ z) / z.shape[0]
        ev, V = torch.linalg.eigh(C.double())
        z = (z.double() @ (V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T)).float()
    h = z
    for L in range(32):
        h = torch.relu(h @ W[L])
    return h.double().mean(0).cpu().numpy()


KINDS = ['iid', "iid'", 'latt', 'sobol+rot', 'latt+rot', 'latt+haar']
have = {int(v): i for i, v in enumerate(SEEDS)}
import glob                                                          # noqa: E402
files = [q for q in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(q)['seed']) in have][:NETS]

se = {(k, w): [] for k in KINDS for w in (True, False)}
for ni, f in enumerate(files):
    zz = np.load(f)
    s0 = int(zz['seed'])
    truth = np.asarray(zz['a_true'])[-1].astype(np.float64)          # 🚨 саме [-1]
    W = [torch.from_numpy(np.asarray(W_ALL[have[s0]][k], np.float32)).to(DEV)
         for k in range(32)]
    torch.manual_seed(s0)
    rot = rot_from(dominant_dir(W))
    for k in KINDS:
        for w in (True, False):
            acc = 0.0
            for r in range(REPS):
                sd = s0 * 1000 + r * 7 + (11 if k == "iid'" else 0)
                p = run(W, points(k, sd, rot), w)
                acc += float(((p - truth) ** 2).mean())
            se[(k, w)].append(acc / REPS)
    print(f'  … {ni+1}/{len(files)} {s0}  iid {se[("iid",True)][-1]:.3e}  '
          f'latt {se[("latt",True)][-1]:.3e}  latt+rot {se[("latt+rot",True)][-1]:.3e}',
          flush=True)

print(f'\n{len(files)} мереж × {REPS} повторів, N={N} пар (2N={2*N} семплів), d={D}')
for w in (True, False):
    print(f'\n{"=== ВИБІЛЮВАННЯ УВІМК (наша конфігурація) ===" if w else "=== ВИБІЛЮВАННЯ ВИМК ==="}')
    print(f'{"набір":12s} {"MSE (сума)":>12s} {"×iid сума":>10s} {"×iid медіана":>13s} {"кращих":>8s}')
    print('-' * 62)
    b = se[('iid', w)]
    for k in KINDS:
        a = se[(k, w)]
        rat = [b[i] / a[i] for i in range(len(a))]
        print(f'{k:12s} {sum(a):12.4e} {sum(b)/sum(a):9.4f}× '
              f'{float(np.median(rat)):12.4f}× {sum(1 for q in rat if q > 1):4d}/{len(a)}')
print(f'\n⛔ПЛОМБА: `iid′` має бути 1.00× — усе, що ближче до неї, ніж до 1.15×, є ШУМ')
