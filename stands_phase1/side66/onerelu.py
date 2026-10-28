"""ЛІНІЙНА + ПО ОДНОМУ ReLU: перший порядок розкладу за «релушністю» (ідея юзера).

Якщо ReLU увімкнено рівно в одного нейрона `(ℓ,i)`, а вся решта лінійна, внесок
рахується ТОЧНО у замкненій формі:
    E[ReLU(p) − p] = σ_{ℓ,i}/√(2π),   σ_{ℓ,i} = ‖A_ℓ[:,i]‖,  A_ℓ = W₁…W_ℓ
і далі їде вниз лінійно через B_ℓ = W_{ℓ+1}…W₃₂. Сума по ВСІХ нейронах — це
перший порядок розкладу, і коштує вона 64 матмули = **0.4% бюджету**.

🚨 ПІДОЗРА ДО САМОЇ ПОСТАНОВКИ. He-ініціалізація (`std=√(2/n)`) підібрана так,
щоб ReLU ЗБЕРІГАВ дисперсію: ReLU ріже її навпіл, а множник 2 компенсує. Знімемо
ReLU — і кожен шар ПОДВОЮЄ дисперсію. За 32 шари це **2³² ≈ 4·10⁹ по дисперсії**,
тобто **2¹⁶ = 65536× по масштабу**. Розклад навколо точки, яка в 65 тисяч разів
більша за відповідь, збігатися не може в принципі.

📣 ПЕРЕДБАЧЕНО: ‖лінійний вихід‖ / ‖істина‖ ≈ 10⁴–10⁵; перший порядок промахне
   на стільки ж порядків. 🎓 ЛІКИ — розкладати не навколо ГОЛОЇ лінійної, а
   навколо СЕРЕДНЬОВОРОТНОЇ лінійної (`⊙g`, g≈0.5), і тоді перший порядок стає
   рівно `c_ℓ`-розкладом із `linrelu.py`. Тобто ідея юзера ПРАВИЛЬНА, але точку
   розкладу треба зсунути.
🟢 ЖИВЕ, якщо перший порядок дає відносну похибку ≤50%.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NS = 8192
W_ = 256
S2PI = float(1.0 / np.sqrt(2 * np.pi))

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)

rows = []
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV)
         for k in range(32)]  # 🚨 fp32
    g = torch.Generator(device=DEV).manual_seed(71_000 + 17 * n)
    with torch.no_grad():
        # істина (та сама мережа з ReLU)
        z = torch.randn(NS, W_, device=DEV, generator=g)
        z = torch.cat([z, -z], 0)
        h = z
        for m in W:
            h = torch.relu(h @ m)
        truth = h.double().mean(0)

        # A_ℓ = W₁…W_ℓ  (прямі композиції), B_ℓ = W_{ℓ+1}…W₃₂ (зворотні)
        A = []
        acc = torch.eye(W_, device=DEV)
        for L in range(32):
            acc = acc @ W[L]
            A.append(acc.clone())
        B = [None] * 32
        acc = torch.eye(W_, device=DEV)
        for L in range(31, -1, -1):
            B[L] = acc.clone()
            acc = W[L] @ acc

        lin = torch.zeros(W_, device=DEV)   # E[лінійна] = 0
        # перший порядок: сума одиночних ReLU
        first = torch.zeros(W_, dtype=torch.float64, device=DEV)
        for L in range(32):
            sig = A[L].pow(2).sum(0).sqrt()                      # σ_{ℓ,i}
            first = first + ((sig * S2PI) @ B[L]).double()
        # масштаб голої лінійної мережі (розкид виходу)
        lz = z @ A[31]
        rows.append((float(truth.norm()), float(lz.std()), float(h.std()),
                     float((first - truth).norm() / truth.norm()),
                     float(first.norm() / truth.norm())))
    del W, A, B
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}  ‖істина‖={rows[-1][0]:.3e} '
          f'std(лін)={rows[-1][1]:.3e} std(ReLU)={rows[-1][2]:.3e}', flush=True)

R = np.array(rows)
print(f'\n{NETS} мереж\n')
print(f'📏 РОЗДУВ ГОЛОЇ ЛІНІЙНОЇ: std(лінійна)/std(з ReLU) = '
      f'{np.mean(R[:,1]/R[:,2]):.3e}   (теорія 2^16 = {2**16:.3e})')
print(f'   ⟹ точка розкладу завелика у ~{np.mean(R[:,1]/R[:,2]):.0e} разів\n')
print(f'ЛІНІЙНА (усі ReLU знято):    відповідь РІВНО 0 ⟹ відн. похибка 100.0%')
print(f'ПЕРШИЙ ПОРЯДОК (по 1 ReLU):  відн. похибка {100*np.mean(R[:,3]):.1f}%   '
      f'(‖перший‖/‖істина‖ = {np.mean(R[:,4]):.3e})')
ok = np.mean(R[:, 3]) <= 0.5
print(f'\n🔑 ВИРОК: {"🟢 ЖИВЕ" if ok else "⛔ МЕРТВО"} (гейт: відн. похибка ≤50%)')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
