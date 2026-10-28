"""КАРТИНА АКТИВНОСТІ ЯК КОНТРОЛЬНА ЗМІННА. (С86)

## ЗВІДКИ — ІДЕЯ ЮЗЕРА
«На кожну вагу поставити імовірність спрацювання, і наші 6к семплів мають
створити картину активності включно з негативами або мертвими.»

## ЩО ВЖЕ Є В САБМІТІ (щоб не міряти вдруге)
`gate` = частка активних НА НЕЙРОН — стоїть тричі: як канал `_LAM_G`, як
множник переносу похибки, і як фільтр мертвих у пілоті. Опора — `Φ(t)`.
🔑 АЛЕ `gate` — це середнє ПО ЗРАЗКАХ. Тут перевіряємо принципово інше:

## 🎯 ЩО НОВЕ: `n_act(z)` — ПОЗРАЗКОВА кількість відкритих нейронів
    `n_act_ℓ(z) = Σ_i 1{h_ℓi(z) > 0}`      одне число на зразок на шар
Це ТА САМА прихована змінна, що робить форму: «відкрилося багато входів ⟹
`h` великий». `gate` її не бачить, бо усереднений по `z`.
📐 звʼязок: `mean_z n_act = Σ_i gate_i` ⟹ СЕРЕДНЄ тотожне сумі `gate`
   і нічого не додає; уся нова інформація — у ВАРІАЦІЇ `n_act` по зразках.

## ЩО МІРЯЄМО — ЧИСТУ ТОЧНІСТЬ, БЕЗ ЦІНИ (вказівка ЮЗЕРА)
`R²` регресії `a_L(z)` на `n_act_ℓ(z)` ПО ЗРАЗКАХ ⟹ виграш дисперсії
`1/(1−R²)`. Це СТЕЛЯ каналу при ідеальній опорі, тобто верхня межа гілки.
Варіанти: кожен шар окремо · усі 32 разом · ⛔контроль: випадковий вектор.

⛔ПЛОМБИ:
 (0) шар 1: `n_act` від `h₁=zᵀW₀` — теж має сигнал, це НЕ нуль (контроль сенсу);
 (к) ⛔випадковий регресор тієї ж розмірності — `R²` мусить бути ~32/N ≈ 0.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: `n_act` глибоких шарів сильно корелює з `a_L`
   (обидва ростуть, коли мережа «широко відкрита»), `R²` одного шару 0.2–0.5,
   усіх 32 разом 0.5–0.75 ⟹ виграш ×2–4. Ризик: цей сигнал може ВЖЕ бути
   знятий блендом і каналами — тоді стеля є, а приросту над сабмітом нема.
   Шанс, що `R²(усі)` ≥ 0.5: **60%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/actcv.py [cuda:N] [NETS] [LOG2N]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NS = 1 << LOG2N
BATCH = 1 << 13
L_, W_ = 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж, вибірка 2^{LOG2N} = {NS}, {DEV}\n', flush=True)


def collect(W, dev, ns, batch, seed):
    """Повертає `A` (ns×256 виходів) і `NA` (ns×32 кількостей активних)."""
    A = torch.empty(ns, W_, dtype=torch.float32, device=dev)
    NA = torch.empty(ns, L_, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    d = 0
    while d < ns:
        b = min(batch, ns - d)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            h = a @ W[i]
            NA[d:d+b, i] = (h > 0).to(torch.float32).sum(1)
            a = torch.relu(h)
        A[d:d+b] = a
        d += b
    return A, NA


def r2(Y, X):
    """Множинний `R²` регресії кожного стовпця Y на X (з константою).

    Повертає СЕРЕДНІЙ по стовпцях Y, зважений дисперсією — це саме те, що
    входить у MSE: `виграш = 1/(1−R²)`.
    """
    Xc = (X - X.mean(0)).double()
    Yc = (Y - Y.mean(0)).double()
    G = Xc.T @ Xc
    G = G + torch.eye(G.shape[0], dtype=G.dtype, device=G.device) * 1e-9 * G.diagonal().mean()
    B = torch.linalg.solve(G, Xc.T @ Yc)
    res = Yc - Xc @ B
    ss_r = (res * res).sum()
    ss_t = (Yc * Yc).sum()
    return float(1.0 - ss_r / ss_t)


ROWS = []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        A, NA = collect(W, DEV, NS, BATCH, 3300 + n)
        per = [r2(A, NA[:, i:i+1]) for i in range(L_)]
        allr = r2(A, NA)
        tail = r2(A, NA[:, -8:])
        head = r2(A, NA[:, :8])
        rnd = torch.randn(NS, L_, device=DEV,
                          generator=torch.Generator(device=DEV).manual_seed(7))
        ctrl = r2(A, rnd)
        ROWS.append(per + [allr, head, tail, ctrl])
    print(f'  … {n+1}/{len(FS)} seed {sd0} · R²(усі 32) = {allr:.4f} · '
          f'⛔контроль {ctrl:.5f}', flush=True)
    del W, A, NA
    torch.cuda.empty_cache()

M = np.array(ROWS)
np.save('/mnt/win_d/llm/White-Box/arena62/logs/actcv.npy', M)
print(f'\n⛔ПЛОМБА(к) випадковий регресор 32 стовпці: R² = {M[:,-1].mean():.6f} '
      f'(теорія 32/{NS} = {32/NS:.6f})')
print(f'\n{"шар":>5s} {"R² одного шару":>15s} {"виграш ×":>10s}')
print('─' * 34)
for i in list(range(6)) + [7, 11, 15, 19, 23, 27, 31]:
    r = M[:, i].mean()
    print(f'{i+1:5d} {r:15.4f} {1/(1-r):10.3f}')
print('─' * 34)
for k, nm in ((-4, 'УСІ 32 шари разом'), (-3, 'лише шари 1..8'),
              (-2, 'лише шари 25..32')):
    r = M[:, k].mean()
    print(f'{nm:<22s} R² = {r:.4f}   виграш дисперсії ×{1/(1-r):.3f}')
print('\n🎯 це СТЕЛЯ при ідеальній опорі. Зміщення опори НЕ враховано.')
