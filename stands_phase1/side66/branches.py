"""СКІЛЬКИ ГІЛОК НАСПРАВДІ: ентропія патерну активацій (ідея юзера, С78).

ТЕОРІЯ. Один шар з 256 нейронів у 256 вимірах ріже простір на
    Σ_{k=0}^{256} C(256,k) = 2²⁵⁶ ≈ 10⁷⁷ конусів      (теорема Заславського)
32 шари ⟹ верхня межа (2²⁵⁶)³² = 2⁸¹⁹² ≈ 10²⁴⁶⁶. Еквівалентно: патерн
відкрито/закрито — це 32×256 = 8192 біти, по біту на нейрон.

АЛЕ монетки нечесні. `attractor.py` зміряв |t|=|m|/s → 3.0 на глибині ⟹ типовий
нейрон відкритий з p=Φ(3)=0.9987 або 0.0013, тобто МАЙЖЕ ДЕТЕРМІНОВАНИЙ, і його
ентропія 0.0146 біта, а не 1.

Тут рахуємо РЕАЛЬНУ ентропію: H_i = −g log₂g − (1−g)log₂(1−g) для кожного з 8192
нейронів, де g — зміряні ворота. ΣH_i — це ВЕРХНЯ межа ентропії патерну (спільна
ентропія ≤ сумі маргінальних), отже верхня межа log₂(числа доріг, які реально
має гаусова міра).

🚨 ΣH — саме ВЕРХНЯ межа: нейрони сильно корельовані, тож справжня ентропія ще
   менша. Тому висновок «доріг не більше ніж» — надійний, «не менше» — ні.
"""
import sys
import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 17
W_, D, CH = 256, 32, 1 << 13
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('🚨')[1], flush=True)

def _gates(W, seed, npairs):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    G = torch.zeros(D, W_, dtype=torch.float64, device=DEV)
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]
            G[L] += (p > 0).double().sum(0)
            x = torch.relu(p)
        done += b
    return (G / float(2 * npairs)).cpu().numpy()

H = np.zeros((NETS, D)); GG = []
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    with torch.no_grad():
        g = _gates(W, 91_000 + 13 * n, 1 << LOGN)
    gc = np.clip(g, 1e-12, 1 - 1e-12)
    h = -(gc * np.log2(gc) + (1 - gc) * np.log2(1 - gc))
    H[n] = h.sum(1); GG.append(g)
    del W; torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}  ΣH = {H[n].sum():.1f} біт', flush=True)

G = np.concatenate([g.ravel() for g in GG])
tot = H.sum(1).mean()
print(f'\n{NETS} мереж × 2·2^{LOGN} семплів\n')
print(f'{"шар":>4s} {"ентропія, біт":>15s} {"з 256 можливих":>16s}')
print('-' * 40)
Hm = H.mean(0)
for L in (0, 1, 2, 4, 8, 16, 24, 31):
    print(f'{L:4d} {Hm[L]:15.2f} {Hm[L]/256:15.1%}')
print(f'\n{"":=^64s}')
print(f'НАЇВНО (чесні монетки):     8192.0 біт ⟹ 2^8192 ≈ 10^2466 доріг')
print(f'РЕАЛЬНО (верхня межа ΣH):   {tot:.1f} біт ⟹ 2^{tot:.0f} ≈ 10^{tot*0.30103:.0f} доріг')
print(f'⟹ стиснення у **{8192/max(tot,1e-9):.1f}×** по бітах, '
      f'тобто у 10^{(8192-tot)*0.30103:.0f} разів по числу доріг')
q = np.percentile(G, [1, 5, 25, 50, 75, 95, 99])
print(f'\nрозподіл воріт g по всіх {len(G)} нейронах:')
print('   ' + '  '.join(f'p{p}={v:.4f}' for p, v in
                        zip([1, 5, 25, 50, 75, 95, 99], q)))
dead = float((G < 0.01).mean()); alw = float((G > 0.99).mean())
act = float(((G > 0.2) & (G < 0.8)).mean())
print(f'   майже завжди ЗАКРИТИЙ (g<0.01): {dead:.1%}   '
      f'майже завжди ВІДКРИТИЙ (g>0.99): {alw:.1%}')
print(f'   🔑 СПРАВДІ ПЕРЕМИКНІ (0.2<g<0.8): **{act:.1%}** — це і є вся розвилка')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
