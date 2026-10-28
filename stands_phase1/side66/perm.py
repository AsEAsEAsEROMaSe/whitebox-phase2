"""ЧИ ВАЖИТЬ ПОРЯДОК ШАРІВ: перестановка тих самих матриць (ідея юзера, С78).

Питання: матричне множення некомутативне, а з ReLU між множеннями — тим паче.
`relu(relu(z·W₁)·W₂)` і `relu(relu(z·W₂)·W₁)` — навіть не близькі функції, бо
кожен ReLU ріже по СВОЇХ гіперплощинах, і після перестановки ріжуться інші
напрямки. Отже ФУНКЦІЯ змінюється сильно. Але чи змінюється СКЛАДНІСТЬ її
оцінювання?

Беремо ТІ САМІ 32 матриці і переставляємо:
    ORIG     — як є
    REVERSE  — задом наперед
    SHUFFLE  — кілька випадкових перестановок
    SORTED   — за зростанням спектральної норми (найгірші наприкінці)
    ANTISORT — навпаки
і міряємо відносну похибку вибіркової оцінки (крос-істина, ZCA, антитетика).

🔑 ЩО ЦЕ ВИРІШУЄ: якщо похибка НЕ залежить від порядку — немає «поганих пар
   сусідніх шарів», локальна структура не важить, і шукати треба тільки
   глобальні властивості. Це підтвердило б [[finding-error-is-additive-across-all-32-layers]]
   з іншого боку. Якщо ж SORTED помітно кращий за ANTISORT — порядок несе
   інформацію, і це нова вісь.

🚨 Кожна перестановка НОРМУЄТЬСЯ пошарово (std(p)=1), інакше порівнюємо різні
   масштаби. Позитивна однорідність робить це безпечним.

📣 ПЕРЕДБАЧЕНО: матриці iid ⟹ будь-яка перестановка дає мережу З ТОГО САМОГО
   ансамблю ⟹ похибка НЕ має залежати від порядку (розкид у межах шуму по
   мережах). Ставлю на розкид ≲1.1× між усіма варіантами.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 17
NPAIR = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
DRAWS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
W_, D, CH = 256, 32, 1 << 14

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)


def _norm(W):
    """Пошарове нормування std(p)=1 — щоб усі перестановки були порівнянні."""
    out = [w.clone() for w in W]
    zz = torch.randn(512, W_, device=DEV,
                     generator=torch.Generator(device=DEV).manual_seed(4242))
    x = torch.cat([zz, -zz], 0)
    for L in range(D):
        sd = float((x @ out[L]).double().std().clamp_min(1e-30))
        out[L] = out[L] / sd
        x = torch.relu(x @ out[L])
    return out


def _est(W, seed, npairs, whiten=False):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        if whiten:
            cov = (z.T @ z) / float(b)
            ev, U = torch.linalg.eigh(cov.double())
            z = (z.double() @ (U * ev.clamp_min(1e-8).rsqrt()) @ U.T).float()
        x = torch.cat([z, -z], 0)
        for L in range(D):
            x = torch.relu(x @ W[L])
        acc += x.double().sum(0)
        done += b
    return acc / float(2 * npairs)


rng = np.random.default_rng(20260801)
# ⛔ORIG' — ПЛОМБА: той самий порядок удруге. Розкид між порядками треба
#   порівнювати НЕ з довільним гейтом, а з цією величиною.
VARIANTS = ['ORIG', "⛔ORIG'", 'REVERSE', 'SORTED', 'ANTISORT',
            'SHUF-1', 'SHUF-2', 'SHUF-3']
res = np.zeros((len(VARIANTS), 2))
for n in range(NETS):
    W0 = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV)
          for k in range(D)]
    with torch.no_grad():
        sn = np.array([float(torch.linalg.matrix_norm(w, 2)) for w in W0])
        orders = {
            'ORIG': list(range(D)),
            'REVERSE': list(range(D))[::-1],
            'SORTED': list(np.argsort(sn)),
            'ANTISORT': list(np.argsort(-sn)),
            "⛔ORIG'": list(range(D)),
        }
        for k in range(1, 4):
            orders[f'SHUF-{k}'] = list(rng.permutation(D))
        for vi, v in enumerate(VARIANTS):
            W = _norm([W0[i] for i in orders[v]])
            tA = _est(W, 91_000 + 13 * n, 1 << LOGN)
            tB = _est(W, 40_000 + 29 * n, 1 << LOGN)
            for d in range(DRAWS):
                mh = _est(W, 55_000 + 7 * n + 1013 * d, NPAIR, whiten=True)
                res[vi, 0] += float(((mh - tA) * (mh - tB)).sum())
            res[vi, 1] += float((tA * tB).sum()) * DRAWS
            del W
    del W0
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}', flush=True)

print(f'\n{NETS} мереж × {DRAWS} жеребів (відношення СУМ)\n')
print(f'{"порядок шарів":>14s} {"відн. похибка":>15s} {"проти ORIG":>12s}')
print('-' * 44)
base = np.sqrt(res[0, 0] / res[0, 1])
rels = []
for vi, v in enumerate(VARIANTS):
    r = np.sqrt(max(res[vi, 0], 0) / res[vi, 1])
    rels.append(r)
    print(f'{v:>14s} {r:15.4e} {base/max(r,1e-30):11.3f}×')
seal = max(rels[0], rels[1]) / min(rels[0], rels[1])
spread = max(rels) / min(rels)
print(f'\n⛔ПЛОМБА (ORIG проти ORIG\'): {seal:.4f}× — це шум САМОГО стенда')
print(f'🔑 РОЗКИД між порядками: **{spread:.3f}×**')
sysd = rels[3] / rels[4]           # SORTED / ANTISORT — протилежні за побудовою
print(f'🔑 SORTED/ANTISORT (мали б розійтись найдужче): {sysd:.4f}×')
ok = spread > 1.5 * seal and abs(np.log(sysd)) > abs(np.log(seal))
print(f'ВИРОК: {"🟢 ПОРЯДОК НЕСЕ ІНФОРМАЦІЮ" if ok else "⛔ ПОРЯДОК НЕ ВАЖИТЬ — розкид у межах шуму стенда"}')
print('📊 БАЛ: 1.954e-07 — вимір діагностичний')
