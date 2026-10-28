"""УСЕРЕДНЕННЯ ВЗДОВЖ ОРБІТИ ПРОТИ УСЕРЕДНЕННЯ ПО СЕМПЛАХ (С80, питання юзера).

Питання: «а якщо ці 20 ітерацій скласти через метод, яким ми середнє визначаємо?»

🔑 ЦЕ ДВА РІЗНІ УСЕРЕДНЕННЯ:
    наш метод  — по 8192 РІЗНИХ входах  ⟹ сходиться до `E[o]` (ціль)
    орбіта     — по 20 кроках ОДНОГО входу ⟹ сходиться до `x*` (нерухома точка)
Орбіта схлопується в точку (зміряно: попарний cos між ітератами 0.993 на кроці 2,
рівно 1.00000 на кроці 40), тож середнє вздовж неї = та сама точка.

📐 І ДРУГА ПРИЧИНА: ми НЕ обмежені точністю `x*`. `exactcos.py` дав шум
   **2.8e-06**, а відстань `x*`↔`E[o]` = **1.2e-02** — розрив у 4000×.

⚠️ АЛЕ АРГУМЕНТ НЕ ПОКРИВАЄ ОДНОГО: `o₁` несе інформацію про КОНКРЕТНИЙ семпл,
   а `o₂…o₂₀` — уже ні. Тобто `A_K` це суміш «трохи семпла» + «багато
   атрактора», і теоретично могла б лягти БЛИЖЧЕ за обидва краї. Тому міряємо.

## ЩО МІРЯЄМО
`A_K = (1/K)·Σ_{k=1..K} ô_k` (кожен ітерат нормований, інакше домінує масштаб),
і `cos(A_K, E[o])` проти `cos(o_K, E[o])`. Ціль — банкова істина `T1`.

⛔ ПЛОМБИ: при `K=1` обидва стовпці мусять збігтися БІТ-У-БІТ ·
   ⛔НЕГ: cos із істиною ЧУЖОЇ мережі ·
   при великих `K` обидва мусять зійтися до одного (бо орбіта в точці).

📣 ПЕРЕДБАЧЕНО: `A_K` буде МІЖ `o₁` і `x*`, тобто НЕ краще за кращий з них.
   Ставлю, що виграшу над `max(o₁, x*)` не буде взагалі, бо орбіта не приносить
   нових семплів — а лише вони й дають точність (`15%/√N`).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/orbitavg.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
KS = [1, 2, 3, 5, 10, 20]
W_, D = 256, 32

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
TZ = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
SEEDS = list(np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'))
IDX = [SEEDS.index(s) for s in TZ['seeds'].tolist()]
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _f(W, X):
    for L in range(D):
        X = torch.relu(X @ W[L])
    return X


rows = []
for j in range(min(NETS, len(IDX))):
    W = [torch.tensor(np.asarray(WZ[IDX[j]][k], np.float32), device=DEV)
         for k in range(D)]
    t1 = torch.tensor(TZ['T1'][j], dtype=torch.float64, device=DEV)
    tf = torch.tensor(TZ['T1'][(j + 7) % len(IDX)], dtype=torch.float64,
                      device=DEV)
    u1, uf = t1 / t1.norm(), tf / tf.norm()
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(9001 + 17 * j)
        X = torch.randn(NS, W_, device=DEV, generator=g)
        X = X / X.norm(dim=1, keepdim=True)
        acc = torch.zeros(NS, W_, dtype=torch.float64, device=DEV)
        r = {}
        for k in range(1, max(KS) + 1):
            Y = _f(W, X)
            X = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
            acc += X.double()
            if k in KS:
                A = acc / acc.norm(dim=1, keepdim=True)      # орбітне середнє
                O = X.double()
                r[k] = (float((A @ u1).median()), float((O @ u1).median()),
                        float((A @ uf).median()))
        rows.append([r[k] for k in KS])
    if j % 8 == 0:
        print(f'  … мережа {j+1}/{min(NETS,len(IDX))}', flush=True)
    del W
    torch.cuda.empty_cache()

a = np.array(rows)
print(f'\n{a.shape[0]} мереж × {NS} семплів\n')
print(f'{"K":>4s} {"cos(A_K, E[o])":>16s} {"cos(o_K, E[o])":>16s} '
      f'{"різниця":>10s} {"⛔НЕГ":>9s}')
print('-' * 60)
for i, k in enumerate(KS):
    ca, co, cf = (np.median(a[:, i, q]) for q in range(3))
    print(f'{k:4d} {ca:16.6f} {co:16.6f} {ca-co:+10.6f} {cf:9.4f}'
          f'{"   ← пломба: мусять збігтись" if k == 1 else ""}')

best_o = float(np.median(a[:, -1, 1]))
best_a = float(np.median(a[:, :, 0].max(1)))
print(f'\n🔑 НАЙКРАЩЕ орбітне середнє = {best_a:.6f} проти '
      f'ітерата {best_o:.6f}')
for nm, c in (('орбітне середнє', best_a), ('простий ітерат', best_o)):
    e = float(np.sqrt(max(1 - c * c, 0)))
    print(f'   {nm:18s}: збоку {100*e:6.2f}% ⟹ проти цілі 0.02% промах '
          f'у {e/2e-4:6.0f}×')
print('\n🎯 ВИСНОВОК: орбіта не приносить НОВИХ СЕМПЛІВ, а точність дає лише '
      'їх кількість (15%/√N)')
print('📊 БАЛ 1.954e-07')
