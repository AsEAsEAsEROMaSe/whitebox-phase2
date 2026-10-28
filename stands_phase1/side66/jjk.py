"""ЧЛЕН `(j,j,k)` — ЧИ ВІН ГОЛОВНИЙ У `κ₃(h)`. (С86)

## ЗВІДКИ — ФОРМУЛЮВАННЯ ЮЗЕРА
«Все вирішиться в залежності вага мінус або плюс.» Точна тотожність:
    `κ₃(h_i) = Σ_{j,k,l} W_ji·W_ki·W_li · κ₃(a_j,a_k,a_l)`
Розбиття за збігом індексів, і ключ — У ЯКОМУ СТЕПЕНІ входить вага:
  (j,j,j)  n членів,   `W³` НЕПАРНИЙ  ⟹ знаки скасовуються  ⟹ ~0.04
  (j,j,k)  n² членів,  `W²·W` — `W²` БЕЗ ЗНАКУ, `ΣW²=2` детерміновано ⟹ ~8.5
  (j,k,l)  n³ членів,  всі непарні, скасування тричі         ⟹ ~2.8
🚨 Наша рекурсія (`cumrec.py`, С85) бере ЛИШЕ ПЕРШИЙ рядок ⟹ 0.4% від суми.
   Зміряний промах тієї рекурсії — **280×**, тобто 0.36%. ЗБІГ.
⟹ гіпотеза: викинуто саме той член, де вага стоїть у ПАРНОМУ степені.

## 🔑 ЧОМУ ЦЕ ВАЖЛИВО ПРАКТИЧНО
`(j,j,k)` містить не тензор `n³`, а МАТРИЦЮ `S_jk = E[ã_j²·ã_k]` (n²), і
згортка `3·diag((W²)ᵀ S W)` коштує `n³` на шар. Проти 151% бюджету за повний
co-skewness — тут ~2.6% на всі 32 шари.

## ⛔ ЩО САМЕ МІРЯЄ ЦЕЙ СТЕНД (і чого НЕ міряє)
Міряє СТРУКТУРНУ повноту: беремо `S` і еталон `κ₃(h)` з ТІЄЇ САМОЇ вибірки,
тож жодної помилки рекурсії тут нема — лише те, скільки важать викинуті члени.
НЕ міряє: чи існує замкнена рекурсія для `S` без вибірки. Це наступний крок,
і він має сенс ЛИШЕ якщо цей пройде.

⛔ПЛОМБИ:
 (0) шар 0: вхід `z` гаусів ⟹ `S≡0`, `D=T=0`, `κ₃(h₁)=0` — усе мусить бути ~0;
 (а) корел `(D+T)↔κ₃` ВСЕРЕДИНІ шару ≥ 0.90;
 (б) масштаб `‖D+T‖/‖κ₃‖` СТАЛИЙ по шарах, розкид <20%;
 (в) надійність еталона: половини вибірки, корел між ними.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ:
   `D` сам дасть масштаб 0.01–0.10 (більше за 0.0036 з cumrec, бо там `κ₃(a)`
   йшли з замикання, а тут — точні з вибірки; ізолюємо ЛИШЕ структуру).
   `D+T` — масштаб 0.5–0.9, корел 0.85–0.97, з погіршенням углиб.
   Головний ризик: член `(j,k,l)` може НЕ бути малим, бо кореляції активацій
   когерентні, і `ΣW_j ~ √2` множиться на велику спільну `κ₃`.
   Шанс пройти обидва гейти: **35%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/jjk.py [cuda:N] [NETS] [LOG2N]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NS = 1 << LOG2N
BATCH = 1 << 15
L_, W_ = 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж, вибірка 2^{LOG2N} = {NS}, {DEV}\n', flush=True)


def pass1_means(W, dev, ns, batch, seed):
    """Прохід 1: `E[a]` на ВХОДІ кожного шару (для центрування у проході 2)."""
    M = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    done = 0
    while done < ns:
        b = min(batch, ns - done)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            M[i] += a.double().sum(0)
            a = torch.relu(a @ W[i])
        done += b
    return [m / ns for m in M]


def pass2_stats(W, dev, ns, batch, seed, M, half=False):
    """Прохід 2: `S_jk = E[ã_j²ã_k]` на вході шару + еталон `κ₃(h)`.

    🔑 ТОЙ САМИЙ `seed`, що й у проході 1 ⟹ ті самі `z` ⟹ `S` і еталон
    рахуються на ОДНІЙ вибірці, і тотожність `D+T+R = κ₃` точна.
    `half=True` — окремо перші й другі півжеребу (пломба (в)).
    """
    nh = 2 if half else 1
    S = [[torch.zeros(W_, W_, dtype=torch.float64, device=dev)
          for _ in range(L_)] for _ in range(nh)]
    H1 = [[torch.zeros(W_, dtype=torch.float64, device=dev)
           for _ in range(L_)] for _ in range(nh)]
    H2 = [[torch.zeros(W_, dtype=torch.float64, device=dev)
           for _ in range(L_)] for _ in range(nh)]
    H3 = [[torch.zeros(W_, dtype=torch.float64, device=dev)
           for _ in range(L_)] for _ in range(nh)]
    cnt = [0] * nh
    Mf = [m.float() for m in M]
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    done = 0
    while done < ns:
        b = min(batch, ns - done)
        q = 1 if (half and done >= ns // 2) else 0
        a = torch.randn(b, W_, device=dev, generator=g)
        cnt[q] += b
        for i in range(L_):
            ad = a - Mf[i]
            S[q][i] += ((ad * ad).T @ ad).double()
            h = a @ W[i]
            hd = h.double()
            h2 = hd * hd
            H1[q][i] += hd.sum(0)
            H2[q][i] += h2.sum(0)
            H3[q][i] += (h2 * hd).sum(0)
            a = torch.relu(h)
        done += b
    OUT = []
    for q in range(nh):
        c = float(cnt[q])
        Sq, K3 = [], []
        for i in range(L_):
            Sq.append(S[q][i] / c)
            e1, e2, e3 = H1[q][i] / c, H2[q][i] / c, H3[q][i] / c
            K3.append(e3 - 3.0 * e1 * e2 + 2.0 * e1 ** 3)
        OUT.append((Sq, K3))
    return OUT


def decompose(W, S):
    """`D` = діагональ (j,j,j) · `T` = член (j,j,k). Обидва — вектори по `i`."""
    w = W.double()
    w2 = w * w
    s3 = S.diagonal()
    D = (w2 * w).T @ s3
    G = (w2 * (S @ w)).sum(0)
    T = 3.0 * (G - D)
    return D, T


def corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = float(x.norm() * y.norm())
    return float((x * y).sum()) / d if d > 0 else float('nan')


ROWS, HALF = [], []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        M = pass1_means(W, DEV, NS, BATCH, 7100 + n)
        out = pass2_stats(W, DEV, NS, BATCH, 7100 + n, M, half=True)
        (Sa, K3a), (Sb, K3b) = out
        row, hrow = [], []
        for i in range(L_):
            Sf = (Sa[i] + Sb[i]) * 0.5
            Kf = (K3a[i] + K3b[i]) * 0.5
            D, T = decompose(W[i], Sf)
            kn = float(Kf.norm())
            # 🚨 КРОС-ЖЕРЕБ: `S` з половини A проти еталона з половини B.
            #    Спільного шуму НЕМА ⟹ чесна оцінка, без фори.
            Dx, Tx = decompose(W[i], Sa[i])
            Dy, Ty = decompose(W[i], Sb[i])
            knx = max(float(K3b[i].norm()), 1e-300)
            kny = max(float(K3a[i].norm()), 1e-300)
            cx = 0.5 * (corr(Dx + Tx, K3b[i]) + corr(Dy + Ty, K3a[i]))
            sx = 0.5 * (float((Dx + Tx).norm()) / knx
                        + float((Dy + Ty).norm()) / kny)
            cd = 0.5 * (corr(Dx, K3b[i]) + corr(Dy, K3a[i]))
            row.append((corr(D, Kf), corr(D + T, Kf),
                        float(D.norm()) / max(kn, 1e-300),
                        float((D + T).norm()) / max(kn, 1e-300), kn, cx, sx,
                        cd))
            hrow.append(corr(K3a[i], K3b[i]))
        ROWS.append(row)
        HALF.append(hrow)
        z0 = row[0]
        print(f'  … {n+1}/{len(FS)} seed {sd0} · ⛔ПЛОМБА(0) шар0: '
              f'‖κ₃‖={z0[4]:.3e} ‖D‖/‖κ₃‖={z0[2]:.3e}', flush=True)
    del W
    torch.cuda.empty_cache()

A = np.array(ROWS)
H = np.array(HALF)
np.save('/mnt/win_d/llm/White-Box/arena62/logs/jjk.npy', A)

print(f'\n⛔ПЛОМБА(в) надійність еталона (корел половин): '
      f'шар 1 {H[:, 1].mean():+.4f} · 8 {H[:, 8].mean():+.4f} · '
      f'16 {H[:, 16].mean():+.4f} · 31 {H[:, 31].mean():+.4f}')
print(f'\n{"шар":>5s} {"корел D":>9s} {"корел D+T":>10s} {"⚡КРОС D+T":>11s} '
      f'{"масшт D":>9s} {"масшт D+T":>11s} {"⚡КРОС масшт":>12s}')
print('─' * 70)
for i in range(1, L_):
    if i <= 8 or i % 4 == 0 or i == L_ - 1:
        print(f'{i:5d} {A[:, i, 0].mean():+9.4f} {A[:, i, 1].mean():+10.4f} '
              f'{A[:, i, 5].mean():+11.4f} {A[:, i, 2].mean():9.5f} '
              f'{A[:, i, 3].mean():11.4f} {A[:, i, 6].mean():12.4f}')
print('─' * 70)
cx = A[:, 1:, 5].mean(); sx = A[:, 1:, 6].mean(0); cdx = A[:, 1:, 7].mean()
print(f'⚡КРОС-ЖЕРЕБ (S з половини A, еталон з B — БЕЗ спільного шуму):')
print(f'   D+T корел {cx:+.4f} · масштаб {sx.mean():.4f} ± {sx.std():.4f} '
      f'({sx.std()/max(abs(sx.mean()),1e-30)*100:.1f}%) · сама D корел {cdx:+.4f}')
cD, cT = A[:, 1:, 0].mean(), A[:, 1:, 1].mean()
sD, sT = A[:, 1:, 2].mean(0), A[:, 1:, 3].mean(0)
print(f'сер. корел: D {cD:+.4f} · D+T {cT:+.4f}          (гейт ≥0.90)')
print(f'масштаб D+T: {sT.mean():.4f} ± {sT.std():.4f} '
      f'({sT.std()/max(abs(sT.mean()),1e-30)*100:.1f}%)   (гейт розкид <20%)')
print(f'масштаб D  : {sD.mean():.5f} ± {sD.std():.5f}   '
      f'⟹ діагональ дає {sD.mean()*100:.2f}% норми')
print('\n🎯 ГЕЙТ: корел D+T ≥0.90 І розкид масштабу <20% ⟹ (j,j,k) головний,'
      '\n   і тоді має сенс шукати замкнену рекурсію для S БЕЗ вибірки.')
