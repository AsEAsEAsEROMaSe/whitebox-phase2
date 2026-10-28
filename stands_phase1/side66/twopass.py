"""ЧИ ВКАЗУЄ ДРУГИЙ ВИХІД ОДНОГО СЕМПЛА В СЕРЕДНЄ? (С80, питання юзера).

Питання: «для одного семпла перший вихід мережі і другий часом не є вектором,
що вказує в середнє?»

🔑 ЧОМУ ЦЕ НЕ ТЕ САМЕ, ЩО ВЖЕ МІРЯЛОСЬ. С79 зміряла `cos(x*, E[o]) = 0.987`
для ЗБІЖНОГО атрактора — а він настає за 7–20 проходів. Скільки дає ДРУГИЙ
прохід (`o₂ = f(f(z))`), не міряв ніхто. І ціна різна на порядки:
    збіжний `x*` для 1 вектора  — 20 проходів = 0.03% бюджету
    `o₂` для ОДНОГО семпла      — 1 зайвий прохід = 1/8192 бюджету ≈ ДАРОМ
    `o₂` для ВСІХ 8192 семплів  — 2× бюджету (уже закрито: 1.0007×)

## ЩО МІРЯЄМО
Для кожної мережі женемо ОКРЕМІ семпли `o_{k+1} = f(o_k)/‖·‖`:
 * `cos(o_k, E[o])` — чи вказує k-й ітерат у ціль (ціль = банкова істина `T1`);
 * ⛔НЕГ: `cos(o_k, T1 ЧУЖОЇ мережі)` — відділити спільну моду класу;
 * `cos` між РІЗНИМИ семплами — чи вони зливаються;
 * розкид по семплах — це властивість семпла чи мережі.

📐 ПЛАНКА, ВІДОМА НАПЕРЕД: `cos=0.987` ⟹ збоку лишається `√(1−0.987²)=16%`
   амплітуди (2.6% енергії), а ціль — **0.02%**. Тобто НАВІТЬ збіжний атрактор
   промахується на два порядки, і `o₂` може бути лише гіршим.
   ⟹ стенд НЕ шукає важіль — він відповідає «як швидко» і де саме межа.

⛔ ПЛОМБИ: `k=0` (сам вхід `z`) ⟹ cos≈0 · НЕГ сильно нижче за свою ·
   при великих `k` вихід на плато = значення збіжного атрактора.

📣 ПЕРЕДБАЧЕНО: `o₁` дасть уже високий cos (PR падає 165.6→2.61 за ОДИН прохід)
   — ставлю 0.90–0.97; `o₂` ~0.98; плато 0.987. Тобто відповідь юзеру «ТАК, і
   майже одразу», але планка лишиться недосяжною в ~130× за енергією.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 256
KS = [0, 1, 2, 3, 5, 8, 12, 20, 40]
W_, D = 256, 32

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
TZ = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
SEEDS = list(np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'))
IDX = [SEEDS.index(s) for s in TZ['seeds'].tolist()]
print(__doc__.split('📣')[1], flush=True)


def _f(W, X):
    for L in range(D):
        X = torch.relu(X @ W[L])
    return X


rows = []
for j in range(min(NETS, len(IDX))):
    W = [torch.tensor(np.asarray(WZ[IDX[j]][k], np.float32), device=DEV)
         for k in range(D)]
    mu = torch.tensor(TZ['T1'][j], dtype=torch.float32, device=DEV)
    mu = mu / mu.norm()
    mf = torch.tensor(TZ['T1'][(j + 7) % len(IDX)], dtype=torch.float32,
                      device=DEV)
    mf = mf / mf.norm()
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(31337 + 11 * j)
        X = torch.randn(NS, W_, device=DEV, generator=g)
        X = X / X.norm(dim=1, keepdim=True)
        iu = torch.triu_indices(NS, NS, offset=1, device=DEV)
        r = {}
        for k in range(max(KS) + 1):
            if k in KS:
                c, cf = X @ mu, X @ mf
                Gm = X @ X.transpose(0, 1)
                r[k] = (float(c.median()), float(c.std()), float(cf.median()),
                        float(Gm[iu[0], iu[1]].median()))
            if k < max(KS):
                Y = _f(W, X)
                X = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
        rows.append([r[k] for k in KS])
    if j % 8 == 0:
        print(f'  … мережа {j+1}/{min(NETS,len(IDX))}', flush=True)
    del W
    torch.cuda.empty_cache()

a = np.array(rows)
print(f'\n{a.shape[0]} мереж × {NS} ОКРЕМИХ семплів\n')
print(f'{"k проходів":>12s} {"cos(o_k, E[o])":>16s} {"розкид":>9s} '
      f'{"⛔НЕГ чужа":>11s} {"cos МІЖ семплами":>18s}')
print('-' * 72)
for i, k in enumerate(KS):
    c, sd, cf, gm = (np.median(a[:, i, q]) for q in range(4))
    tag = {0: '  ← сам вхід z (пломба ≈0)', 1: '  ← ПЕРШИЙ вихід o₁',
           2: '  ← ДРУГИЙ вихід o₂ ← питання'}.get(k, '')
    print(f'{k:12d} {c:16.5f} {sd:9.5f} {cf:11.5f} {gm:18.5f}{tag}')

c1 = float(np.median(a[:, KS.index(1), 0]))
c2 = float(np.median(a[:, KS.index(2), 0]))
cN = float(np.median(a[:, -1, 0]))
print(f'\n🔑 ВІДПОВІДЬ: o₁ дає cos = {c1:.4f} · o₂ = {c2:.4f} · плато = {cN:.4f}')
for nm, c in (('o₁', c1), ('o₂', c2), ('плато', cN)):
    e = float(np.sqrt(max(1.0 - c * c, 0.0)))
    print(f'   {nm:6s}: збоку лишається {100*e:6.2f}% амплітуди ⟹ '
          f'проти цілі 0.02% промах у {e/2e-4:8.0f}×')
print('\n📊 БАЛ 1.954e-07 — вимір ПОЯСНЮВАЛЬНИЙ; `o₂` для ВСІХ семплів коштує '
      '2× бюджету і вже закрито на 1.0007×')
