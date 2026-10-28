"""ДВА МЕТОДИ НА ТИХ САМИХ СЕМПЛАХ: чи вказує їхня різниця на середнє?

Ідея юзера: `sub87` і `v2` їдуть на ОДНОМУ жеребі, тож спільна вибіркова похибка
в них скорочується, і різниця `Δ = p₈₇ − p_v2` спостережувана БЕЗ істини.
Якщо `Δ` корелює з власною похибкою `e₈₇ = p₈₇ − m`, це безкоштовний канал:
ми знаємо, у який бік хибимо, не знаючи `m`.

🚨 ЧОМУ ЦЕ НЕ ТЕ САМЕ, ЩО «ПЕРЕТИН ≠ СИГНАЛ». Там був ОДИН метод: `x̄ − m` без
`m` не спостережуване, крапка. Тут ДВА рівняння `p_A = m + e_A`, `p_B = m + e_B`
і три невідомі — усе ще недовизначено, АЛЕ якщо структура похибок відома
(коваріація), оптимальний бленд може побити обидва. Питання ЛИШЕ в тому,
наскільки `e_A` і `e_B` різні. Якщо це майже той самий вектор — бленд дає нуль.

📣 ПЕРЕДБАЧЕНО: кореляція буде ДУЖЕ висока (>0.95), бо `sub87` = `v2` плюс малі
   поправки на ТИХ САМИХ семплах ⟹ виграш бленду ≈1.00×, і думка, геометрично
   правильна, впирається в те, що обидві похибки — практично один вектор.
🟢 ЖИВЕ, якщо виграш оптимального бленду ≥1.10× АБО `R²(e₈₇ ~ Δ)` ≥ 0.3.
🚨 `w*` підбирається НА ТИХ САМИХ даних ⟹ це ОРАКУЛ, стеля. Реальний канал
   буде гіршим. Якщо навіть оракул нічого не дає — гілки нема.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
import jitanim                                                    # noqa: E402
import curvegui                                                   # noqa: E402
from base_real import cut_and_budget, closure_parts, W_           # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 24

TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
IDX = {int(v): i for i, v in enumerate(TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
HAVE = {int(v): j for j, v in enumerate(SD)}

print(__doc__.split('📣')[1], flush=True)

rows = []
for n, s0 in enumerate(list(IDX)[:NETS]):
    W = [torch.tensor(np.asarray(WZ[HAVE[s0]][k], np.float32), device=DEV)
         for k in range(32)]
    T1 = torch.tensor(TT['T1'][IDX[s0]], device=DEV)
    T2 = torch.tensor(TT['T2'][IDX[s0]], device=DEV)
    with torch.no_grad():
        wc, npair = cut_and_budget(W, DEV)
        cl = closure_parts(wc, 8, DEV, extras=True)
        P87, PV2 = [], []
        for r in range(DRAWS):
            g = torch.Generator(device=DEV).manual_seed(51_000 + 977 * r + n)
            z = torch.randn(npair, W_, device=DEV, generator=g)
            P87.append(jitanim._run_sub87(wc, npair, cl, DEV, 0, z_ext=z))
            PV2.append(curvegui._v2(W, npair, DEV, z))
        A = torch.stack(P87).double()
        Bv = torch.stack(PV2).double()
    a = A.cpu().numpy(); b = Bv.cpu().numpy()
    t1 = T1.double().cpu().numpy(); t2 = T2.double().cpu().numpy()

    # ✅КРОС-ЦІЛЬ у всіх MSE
    def mse(P):
        return float(np.mean((P - t1) * (P - t2)))
    m87, mv2 = mse(a), mse(b)
    # оптимальний бленд w·p87 + (1−w)·pv2 — ОРАКУЛ (w на тих самих даних)
    ws = np.linspace(-0.5, 2.0, 251)
    mb = [mse(w * a + (1 - w) * b) for w in ws]
    j = int(np.argmin(mb))
    # чи знає Δ про власну похибку e87? (по координатах і жеребах разом)
    tm = (t1 + t2) / 2
    e87 = (a - tm).ravel()
    dd = (a - b).ravel()
    r2 = float(np.corrcoef(e87, dd)[0, 1] ** 2)
    rows.append((s0, m87, mv2, ws[j], mb[j], r2,
                 float(np.corrcoef((a - tm).ravel(), (b - tm).ravel())[0, 1])))
    print(f'  … {n+1}/{NETS} {s0}  MSE87 {m87:.3e}  MSEv2 {mv2:.3e}  '
          f'w* {ws[j]:+.2f}  бленд {m87/mb[j]:.3f}×  R²(Δ) {r2:.3f}  '
          f'ρ(e87,ev2) {rows[-1][6]:.4f}', flush=True)
    del W
    torch.cuda.empty_cache()

S87 = sum(r[1] for r in rows)
SV2 = sum(r[2] for r in rows)
SB = sum(r[4] for r in rows)
print(f'\n{len(rows)} мереж × {DRAWS} жеребів, крос-ціль, ті САМІ семпли в обох')
print(f'MSE sub87 {S87/len(rows):.4e} · MSE v2 {SV2/len(rows):.4e} '
      f'⟹ sub87 кращий у {SV2/S87:.3f}× ПРИ РІВНИХ СЕМПЛАХ')
print(f'🔑 ρ(e₈₇, e_v2) = {np.mean([r[6] for r in rows]):.4f}  '
      f'⟹ спільної частки {100*np.mean([r[6] for r in rows])**2:.1f}%')
print(f'🔑 ОРАКУЛЬНИЙ бленд (w* на тих самих даних): {S87/SB:.4f}×  '
      f'медіана w* {np.median([r[3] for r in rows]):+.2f}')
print(f'🔑 R²(e₈₇ ~ Δ) = {np.mean([r[5] for r in rows]):.4f}  '
      f'(чи знає різниця про власну похибку)')
g = S87 / SB
print(f'\n🔑 ВИРОК: {"🟢 ЖИВЕ" if (g >= 1.10 or np.mean([r[5] for r in rows]) >= 0.3) else "⛔ МЕРТВО"}'
      f'  (гейт: бленд ≥1.10× або R² ≥0.3; і це ще ОРАКУЛ, реальність гірша)')
print(f'📊 БАЛ: 1.954e-07 → {1.954e-07/max(g,1e-9):.3e} у найкращому оракульному разі')
