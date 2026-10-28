"""ЧИ КОНЦЕНТРУЄТЬСЯ ПОХИБКА ЗАМИКАННЯ НА РІДКО-АКТИВНИХ НЕЙРОНАХ? (С87)

## ЗВІДКИ — ДВА ФАКТИ, ЯКІ ДОСІ НЕ ЗВОДИЛИ РАЗОМ
1. 🆕С87 (`stepcov87`): `Var(a)/Var(p)` = 0.48 при `|m/σ|` = 3.00 ⟹ ПОЛОВИНА
   нейронів майже завжди АКТИВНА (`t≈+3`), половина майже завжди МЕРТВА (`t≈−3`).
2. 📜С69 (`finding-bias-lives-on-rarely-switching-neurons`): 24% нейронів, що
   відкриваються рідше ніж у 0.2% семплів, несуть **76% зваженого зміщення**.

⟹ ГІПОТЕЗА: похибка замикання КОНЦЕНТРУЄТЬСЯ на нейронах із `t < 0`, бо для них
`E[ReLU(p)] = σ(φ(t) + tΦ(t))` — крихітна величина, цілком визначена формою
ЛІВОГО ХВОСТА, де гаусове наближення найгірше, а ряд Еджворта РОЗБІГАЄТЬСЯ
(`|γ₃He₃(−3)/6| = 1.5 > 1`).

## 🎯 ЩО МІРЯЄТЬСЯ
Розклад похибки замикання ПО КЛАСАХ нейронів за `t = m/σ`:
  t < −2 (майже мертві) · −2..0 · 0..2 · t > 2 (майже активні)
Для кожного класу: відносна похибка `E[ReLU]` і ВНЕСОК у сумарну похибку.
Плюс — чи точні там ХВОСТОВІ ФУНКЦІОНАЛИ (`T₁`,`D`), бо середнє по всіх
нейронах давало 0.13–1.4%, і це могло ховати катастрофу на мертвих.

## ⛔ПЛОМБИ
 P1 сума внесків по класах = повна похибка (розклад повний).
 P2 частки класів мусять дати ~50/50 навколо нуля (структурний факт вище).
 P3 шум істини окремо.

📣 ПЕРЕДБАЧЕННЯ: похибка на `t<−2` буде в РАЗИ більшою відносно, але внесок
   малим за абсолютом (бо там `E[ReLU]` крихітне). Шанс 60%.
   🎯Якщо ВНЕСОК теж великий — це нова мішень. Шанс 30%.
"""
import glob, math, sys
import numpy as np, torch
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N; BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256; PROBE = (7, 15, 23, 30)
assert DEV not in ('cuda:0', 'cuda:4')
print(__doc__.split('📣')[1], flush=True)
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} · {DEV}\n', flush=True)
BINS = [(-99, -2), (-2, 0), (0, 2), (2, 99)]
NB, NP = len(BINS), len(PROBE)
FRAC = np.zeros((len(FS), NP, NB)); RELE = np.zeros((len(FS), NP, NB))
CONTR = np.zeros((len(FS), NP, NB)); T1E = np.zeros((len(FS), NP, NB))
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV) for q in range(L_)]
    with torch.no_grad():
        SH = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        S2 = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        SA = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        SU = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(2200 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]; a = torch.relu(h)
                if q in PROBE:
                    j = PROBE.index(q); hd = h.double()
                    SH[j] += hd.sum(0); S2[j] += (hd * hd).sum(0)
                    SA[j] += a.double().sum(0)
                    SU[j] += torch.where(hd < 0, hd, torch.zeros_like(hd)).sum(0)
            d += b
        for j, q in enumerate(PROBE):
            m = SH[j] / NS; v = torch.clamp(S2[j] / NS - m * m, min=1e-30)
            s = torch.sqrt(v); t = m / s
            a_true = SA[j] / NS
            a_cl = m * Phi(t) + s * phi(t)          # гаусове замикання
            T1_true = SU[j] / NS
            T1_g = s * (t * Phi(-t) - phi(t))       # гаусів хвостовий функціонал
            tot = float(((a_cl - a_true) ** 2).sum())
            for k, (lo, hi) in enumerate(BINS):
                msk = (t >= lo) & (t < hi)
                cnt = int(msk.sum())
                FRAC[n, j, k] = cnt / W_
                if cnt == 0: continue
                dd = ((a_cl - a_true) ** 2)[msk]
                RELE[n, j, k] = float(torch.sqrt(dd.mean() / (a_true[msk] ** 2).mean()))
                CONTR[n, j, k] = float(dd.sum()) / max(tot, 1e-30)
                T1E[n, j, k] = float(torch.sqrt(
                    ((T1_g - T1_true)[msk] ** 2).mean() / (T1_true[msk] ** 2).mean()))
    print(f'  net{n}', flush=True)
print(f'\n{"=" * 70}\n⛔P1 сума внесків = {CONTR.sum(2).mean():.4f} (мусить бути 1.0000)')
print(f'⛔P2 частка t<0 = {FRAC[:, :, :2].sum(2).mean():.3f} (очік. ~0.5)')
print(f'\n{"=" * 70}\nРОЗКЛАД ПОХИБКИ ЗАМИКАННЯ ПО КЛАСАХ `t = m/σ`')
for j, q in enumerate(PROBE):
    print(f'\n--- шар {q+1} ---')
    print(f'{"клас t":>12s} {"частка":>8s} {"віднос.похиб":>13s} {"ВНЕСОК":>9s} {"похиб T₁":>10s}')
    for k, (lo, hi) in enumerate(BINS):
        nm = f'{lo}..{hi}'.replace('-99', '−∞').replace('99', '+∞')
        print(f'{nm:>12s} {FRAC[:, j, k].mean():8.3f} {RELE[:, j, k].mean():12.3%} '
              f'{CONTR[:, j, k].mean():9.3f} {T1E[:, j, k].mean():10.3%}')
w = CONTR[:, -1, :2].sum(1).mean(); fr = FRAC[:, -1, :2].sum(1).mean()
print(f'\n📣 ВИРОК на останньому шарі: нейрони з t<0 — це {fr:.1%} штук,')
print(f'   але несуть {w:.1%} похибки ⟹ концентрація ×{w / max(fr, 1e-9):.2f}')
