"""ЧИ РОЗПАДАЄТЬСЯ НАША ФУНКЦІЯ НА МАЛОВИМІРНІ ШМАТКИ? (ідея юзера, С77)

«Одиницю можна поділити на менші й зібрати назад»: замість покривати одну
256-вимірну сферу (10^134 точок) — покрити багато маленьких. Арифметика поділу
працює: 64 блоки по 4 виміри = 7 900 точок, менше за наші 8600 семплів.
🚨 АЛЕ це має сенс, ЛИШЕ якщо функція справді розпадається на блоки.

МІРЯЄМО ТРУНКАЦІЙНУ РОЗМІРНІСТЬ: яку частку дисперсії відповіді пояснює
знання ПЕРШИХ `k` координат `z`:
        частка(k) = Var( E[f | z₁..z_k] ) / Var(f)
`W₁` — щільна випадкова матриця, тож будь-які `k` координат рівноправні.

🚨 ОЦІНКА `Var(E[f|·])` ЗАВИЩЕНА, якщо внутрішнє середнє шумне: до неї
додається `Var_внутр/N`. Ліки ті самі, що й скрізь сьогодні — КРОС-ОЦІНКА:
беремо ДВІ незалежні внутрішні половини і рахуємо їхню КОВАРІАЦІЮ по зовнішніх
жеребах. Шуми половин незалежні ⟹ оцінка незміщена.

📣 ПЕРЕДБАЧЕНО: частка(k) ≈ k/256, тобто ПРЯМА. Це означало б, що дисперсія
   розмазана рівно по всіх 256 напрямках і маловимірного шматка НЕМА взагалі
   (32 шари ReLU перемішують усе, а `W₁` щільна й ізотропна).
🟢 ЖИВЕ, якщо частка(8) ≥ 0.15 (тобто 8 вимірів із 256 несуть >15% —
   у 4.8× більше за рівномірне) ⟹ тоді блокове покриття мало б сенс.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
M = int(sys.argv[3]) if len(sys.argv) > 3 else 512      # зовнішніх жеребів
N = int(sys.argv[4]) if len(sys.argv) > 4 else 64       # внутрішніх (×2 половини)
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
W_ = 256

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
print(__doc__.split('📣')[1], flush=True)


def fwd(W, z):
    a = z
    for m in W:
        a = torch.relu(a @ m)
    return a.double()


res = {k: [] for k in KS}
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV)
         for k in range(32)]
    g = torch.Generator(device=DEV).manual_seed(20_000 + 7 * n)
    with torch.no_grad():
        # повна дисперсія
        zz = torch.randn(4096, W_, device=DEV, generator=g)
        V = fwd(W, zz).var(0)                                   # (256,) по виходах
        for k in KS:
            za = torch.randn(M, k, device=DEV, generator=g)
            mu = []
            for half in range(2):                               # ✅КРОС-ОЦІНКА
                acc = torch.zeros(M, W_, dtype=torch.float64, device=DEV)
                for _ in range(N):
                    zb = torch.randn(M, W_, device=DEV, generator=g)
                    zb[:, :k] = za
                    acc += fwd(W, zb)
                mu.append(acc / N)
            a0 = mu[0] - mu[0].mean(0, keepdim=True)
            a1 = mu[1] - mu[1].mean(0, keepdim=True)
            cov = (a0 * a1).sum(0) / (M - 1)                    # незміщена Var(E[f|·])
            # 🚨 мертві вихідні координати мають V=0 ⟹ ділення покоординатно
            # дає nan. Беремо ВІДНОШЕННЯ СУМ (і це ще й правильна статистика).
            res[k].append(float(cov.clamp_min(0).sum() / V.double().sum()))
    del W
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}', flush=True)

print(f'\n{NETS} мереж · {M} зовнішніх × {N}×2 внутрішніх · крос-оцінка')
print(f'{"k вимірів":>10s} {"частка Var":>12s} {"рівномірне k/256":>18s} {"×понад":>9s}')
print('-' * 54)
for k in KS:
    p = float(np.mean(res[k]))
    print(f'{k:10d} {p:12.4f} {k/256:18.4f} {p/(k/256):8.2f}×')
p8 = float(np.mean(res[8]))
print(f'\n🔑 ВИРОК: {"🟢 ЖИВЕ" if p8 >= 0.15 else "⛔ МЕРТВО"} '
      f'(гейт: частка(8) ≥ 0.15; вийшло {p8:.4f})')
print('📊 БАЛ: 1.954e-07 — вимір структурний, сабміт не чіпався')
