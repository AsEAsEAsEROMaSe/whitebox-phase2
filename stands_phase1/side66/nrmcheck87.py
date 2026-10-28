"""⛔ПЛОМБА ПРИЛАДУ: чи правильна калібровка «0.941% ⟺ MSE 1.698e-05». (С87)

## НАВІЩО — Я ЗНАЙШОВ РОЗБІЖНІСТЬ І МУШУ ЇЇ ЗАКРИТИ ДО БУДЬ-ЯКИХ ВИСНОВКІВ
`hybrid87.py` дає МК N=20000 → 0.1790%, що при перерахунку в MSE = 6.35e-07.
Але грейдерний `bud90` (util 0.90, N≈65000, ІЗ каналами) дав **4.579e-07**, а
проста МК на 65000 мала б за стендом `0.179%·√(20000/65000)` = 0.0993% ⟹ 1.9e-07.
⟹ 🚨 **СТЕНД ВИГЛЯДАЄ ОПТИМІСТИЧНІШИМ ЗА РЕАЛЬНИЙ САБМІТ У ~2.4×.**

Уся моя арифметика балу йшла через калібровку `MSE = 1.698e-5·(x/0.00941)²`,
тобто через припущення `RMS(істина)² = 0.1918`. Якщо це не так — усі числа
зміщені СИСТЕМАТИЧНО, і «×1.33 до рекорду» може бути артефактом.
🎓 Це рівно [[feedback-controls-have-three-distinct-diseases]] — «не той об'єкт».

## ЩО РОБИТЬ
Міряє ПРЯМУ MSE (грейдерну метрику `mean((pred−true)²)`) і `RMS(істина)` на
тих самих мережах, БЕЗ жодної калібровки. Плюс МК-драбину по `N`.
"""
import glob, math, sys
import numpy as np, torch
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NS = 1 << (int(sys.argv[3]) if len(sys.argv) > 3 else 20)
BATCH, L_, W_ = 1 << 12, 32, 256
NSAMP = (6000, 12000, 20000, 65000)
assert DEV not in ('cuda:0', 'cuda:4')
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · істина 2^{int(math.log2(NS))} · {DEV}\n', flush=True)
NRM2 = np.zeros(len(FS))
MSE = {q: np.zeros(len(FS)) for q in NSAMP}
REL = {q: np.zeros(len(FS)) for q in NSAMP}
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    with torch.no_grad():
        def run(nsmp, seed, anti=True):
            g = torch.Generator(device=DEV); g.manual_seed(seed)
            acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
            half = nsmp // 2 if anti else nsmp
            d = 0
            while d < half:
                b = min(BATCH, half - d)
                z = torch.randn(b, W_, device=DEV, generator=g)
                a = torch.cat([z, -z], 0) if anti else z
                for q in range(L_):
                    a = torch.relu(a @ W32[q])
                acc += a.sum(0, dtype=torch.float64)
                d += b
            return acc / float((2 if anti else 1) * half)
        tru = run(NS, 70000 + n)
        NRM2[n] = float((tru ** 2).mean())
        for q in NSAMP:
            pr = run(q, 800 + 13 * n)
            MSE[q][n] = float(((pr - tru) ** 2).mean())
            REL[q][n] = math.sqrt(MSE[q][n] / NRM2[n])
    print(f'  net{n} · RMS(істина)={math.sqrt(NRM2[n]):.4f} · '
          f'MSE(N=12000)={MSE[12000][n]:.3e}', flush=True)
print(f'\n🔑 RMS(істина)² = {NRM2.mean():.5f}  (калібровка припускала 0.1918)')
print(f'   ⟹ КОЕФІЦІЄНТ ЗМІЩЕННЯ калібровки = ×{NRM2.mean() / 0.1918:.3f}')
print('\n══ ПРЯМА MSE (грейдерна метрика), БЕЗ калібровки ══')
print('N        MSE          віднос.   бал(множ.)   vs sub97 1.873e-07')
BUD = 2.72e11
for q in NSAMP:
    C_ = q * 2.0 * L_ * W_ * W_ * 0.8 + 0.015 * BUD
    mult = max(0.1, C_ / BUD)
    print(f'{q:6d}  {MSE[q].mean():.4e}  {REL[q].mean():7.4%}  '
          f'{mult:6.3f}  {MSE[q].mean() * mult:.4e}   '
          f'×{1.873e-07 / (MSE[q].mean() * mult):6.3f}')
print('\n⛔ГЕЙТ ПЛОМБИ: чиста МК при N≈65000 мусить дати ≈6.47e-07 '
      '(MC baseline грейдера). Якщо стенд дає СИЛЬНО менше — прилад бреше.')
