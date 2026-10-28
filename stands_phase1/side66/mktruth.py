"""ДВІ НЕЗАЛЕЖНІ ІСТИНИ ВИСОКОЇ ТОЧНОСТІ для крос-цілі.

🚨 ЧОМУ. Банкова `a_true` (400k пар) має ВЛАСНУ похибку ≈4.5e-7 по MSE — це ~20%
нашого робочого MSE 1.95e-6, і саме вона створює фальшиве «плато зміщення» у
свіпі по бюджету (обидва естиматори виположувались на однаковому рівні ~6.9e-7).
🚨🎓 МЕТАДАНІ БАНКУ БРЕШУТЬ: `avar_true/(2·pairs)` дає 6.0e-8, тобто ЗАНИЖУЄ шум
істини у ~7×. Зміряно прямим порівнянням із власною істиною на 2M пар:
розбіжність 7.6e-8 / 1.0e-6 / 4.6e-7 на трьох мережах (rel 0.065–0.086%).

✅ ЛІКИ — КРОС-ЦІЛЬ: дві НЕЗАЛЕЖНІ істини `T1`, `T2` і оцінка
      MSE = E[(p − T1)·(p − T2)]
незміщена для СПРАВЖНЬОГО MSE за будь-якого шуму істини: шуми `T1` і `T2`
незалежні, тож їхній внесок у добуток має нульове середнє. Той самий прийом
уже стоїть у `varmap.py`/`varmapeps.py` — тут він приходить у свіп по бюджету.
"""
import glob
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
P = 1 << 20
CH = 1 << 15

wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): i for i, v in enumerate(sd)}
fs = [q for q in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(q)['seed']) in have][:NETS]


def truth(W, seed):
    acc = torch.zeros(256, dtype=torch.float64, device=DEV)
    rng = np.random.default_rng(seed)
    done = 0
    while done < P:
        k = min(CH, P - done)
        g = torch.tensor(rng.standard_normal((k, 256)).astype(np.float32), device=DEV)
        a = torch.cat([g, -g], 0)
        for i in range(32):
            a = torch.relu(a @ W[i])
        acc += a.double().sum(0)
        done += k
    return (acc / (2.0 * P)).cpu().numpy()


T1, T2, seeds, dbank = [], [], [], []
for n, f in enumerate(fs):
    s0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(wz[have[s0]][k], np.float32), device=DEV)
         for k in range(32)]
    a = truth(W, 4_100_000 + 13 * n)
    b = truth(W, 8_200_000 + 17 * n)
    T1.append(a)
    T2.append(b)
    seeds.append(s0)
    tb = np.asarray(np.load(f)['a_true'])[-1].astype(np.float64)
    db = float((((a + b) / 2 - tb) ** 2).mean())
    dbank.append(db)
    print(f'  … {n+1}/{len(fs)} {s0}  шум НАШОЇ істини ≈{float(((a-b)**2).mean())/2:.3e}'
          f'  · банкова розходиться на {db:.3e}', flush=True)
    del W
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz',
         seeds=np.array(seeds), T1=np.array(T1), T2=np.array(T2))
A, Bv = np.array(T1), np.array(T2)
print(f'\n✅ шум ОДНІЄЇ нашої істини (MSE): {float(((A-Bv)**2).mean())/2:.4e}  (P={P} пар)')
print(f'🚨 шум БАНКОВОЇ істини (MSE):     {float(np.mean(dbank)):.4e}')
print(f'🚨 метадані обіцяли:              6.0e-08  ⟹ брешуть у '
      f'{float(np.mean(dbank))/6.0e-8:.1f}×')
print('📊 → bank_s53/truth2.npz')
