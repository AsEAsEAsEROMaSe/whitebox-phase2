"""ЕДЖВОРТ-ЗАМИКАННЯ END-TO-END: чи виживе виграш після λ? (С82)

## НАВІЩО ОКРЕМИЙ СТЕНД
`side66/clsedge.py` міряє ТОЧНІСТЬ замикання (`m`, `s`, `q` проти істини).
Але С76 винесла вирок: **λ всмоктують хибу механіки**
([[finding-lambda-absorbs-any-mechanism-you-fix]]) — точніше замикання не
зобов'язане дати менший MSE, бо коефіцієнти вже компенсують стару хибу.
Тому вирок дає ЛИШЕ це: той самий пайплайн, підмінене замикання, λ
перефітовані LOO ПО МЕРЕЖАХ, крос-ціль.

## 🔑 ЧОМУ ПІДМІНА ДЕШЕВА
`run_side1(wc, npair, s_cl, m_cl, …)` бере замикання ПАРАМЕТРОМ. Тож
«інше замикання» = інші два аргументи, решта пайплайну недоторкана —
жодного форку коду, жодного ризику розійтися з сабмітом.

## ВАРІАНТИ (24 мережі × 48 жеребів, крос-ціль, LOO по мережах)
  0 ⛔ гаусове (нинішній сабміт)     — база, мусить збігтись із `qedge.py`
  1 🟢 Еджворт-3, рекурсивний γ₁     — БЕЗКОШТОВНО (0.1% флопів)
  2 🎯 Еджворт-3, ІСТИННИЙ γ₁        — СТЕЛЯ гілки
  3 ⛔ Еджворт-3, γ₁ ЗІ ЗСУНУТОЇ мережі — контроль «чи не просто ворушіння»

🚨 Контроль 3 обов'язковий: С81 показала, що будь-яка ДОДАТКОВА ознака в
   регресії сама по собі дає ×1.0165 ні за що. Виграш нижче цього рівня —
   не виграш. Тут ознак не додається (міняється сам базис), але зсунутий
   `γ₁` перевіряє, чи не працює поправка як звичайне підмішування шуму.

📣 ПЕРЕДБАЧЕНО: якщо `clsedge.py` покаже стелю точності `m` нижче 1.3×,
   тут не буде нічого (<1%). Якщо стеля 2×+, чекаю 1.02…1.08× для істинного
   і майже нічого для рекурсивного — бо λ вже вибрані під стару хибу.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/clsend.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, run_side1, LAM_S         # noqa: E402
from clsedge import closure_edge, moments                      # noqa: E402
from tilt_wide import truth_last                               # noqa: E402
from price31 import loo, three_slices                          # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
L_, KS, NDRAW = 32, 31, 48
NPAIR = 262_144
KAPPA = 1.0                       # ⟵ перезаписується найкращим із `clsedge`
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

VAR = ['⛔ гаусове (нинішній сабміт)', '🟢 Еджворт-3, рекурсивний γ₁',
       '🎯 Еджворт-3, ІСТИННИЙ γ₁', '⛔ Еджворт-3, γ₁ ЗІ ЗСУНУТОЇ']
NV = len(VAR)
G = [[] for _ in VAR]
B = [[] for _ in VAR]
C, base_mse = [], []

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]

torch.manual_seed(108)
prev_g = None
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair = cut_and_budget(W, DEV)
        MT = moments(wc, DEV, 830_000 + 7919 * n, NPAIR)
        g_t = [MT[i][3] for i in range(KS)]
        # 🚨 зсунутий `γ₁` мусить пасувати ПО ФОРМІ (обрізання різне у мереж)
        g_x = ([g_t[i] * 0 + prev_g[i].mean() for i in range(KS)]
               if prev_g is not None else None)
        prev_g = g_t
        CL = [closure_edge(wc, KS, DEV, kappa=0.0),
              closure_edge(wc, KS, DEV, kappa=KAPPA),
              closure_edge(wc, KS, DEV, gam_ext=g_t),
              closure_edge(wc, KS, DEV, gam_ext=g_x) if g_x is not None
              else closure_edge(wc, KS, DEV, kappa=0.0)]
        T1 = truth_last(wc, DEV, 510_000 + 7919 * n)
        T2 = truth_last(wc, DEV, 610_000 + 7919 * n)

        nreg = 2 * KS
        gg = [torch.zeros(nreg, nreg, dtype=torch.float64, device=DEV)
              for _ in range(NV)]
        bb = [torch.zeros(nreg, dtype=torch.float64, device=DEV)
              for _ in range(NV)]
        cc, bm = 0.0, 0.0
        for d in range(NDRAW):
            sd = 77_000_000 + 733 * n + d
            s0, m0 = CL[0][0], CL[0][1]
            prb, _ = run_side1(wc, npair, s0[:8], m0[:8], DEV, sd,
                               lam_s=LAM_S)
            bm += float(((prb - T1) * (prb - T2)).sum())
            for v in range(NV):
                s_c, m_c = CL[v][0], CL[v][1]
                p0, S_cl = run_side1(wc, npair, s_c, m_c, DEV, sd,
                                     collect_sigma=True)
                _, A_cl = run_side1(wc, npair, s_c, m_c, DEV, sd,
                                    lam_s=None, collect_anchors=True)
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                if v == 0:
                    cc += float((e1 * e2).sum())
                R = torch.cat([S_cl, A_cl], 0)
                gg[v] += R @ R.T
                bb[v] += R @ e
        for v in range(NV):
            G[v].append(gg[v]); B[v].append(bb[v])
        C.append(cc); base_mse.append(bm)
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

base = np.array(base_mse)
print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ, LOO ПО МЕРЕЖАХ.\n')
print(f'{"варіант":34s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s} '
      f'{"понад базу":>11s}')
print('-' * 78)
out = []
for v, nm in enumerate(VAR):
    val = loo(G[v], B[v], C, 2 * KS)
    s, md, w3 = three_slices(base, val)
    out.append(s)
    g = f'×{s / out[0]:.4f}' if v else '     —'
    print(f'{nm:34s} {s:8.4f}× {md:8.4f}× {w3:9.4f}× {g:>11s}')
ctl = out[3] / out[0]
real = out[1] / out[0]
print(f'\n🎯 РЕАЛІСТИЧНИЙ ×{real:.4f} проти ⛔контролю ×{ctl:.4f}: '
      f'{"🟢 ВИЩЕ ⟹ гілка жива" if real > ctl + 0.01 else "⛔ НЕ ВИЩЕ ⟹ мертво"}')
print(f'📊 БАЛ 1.954e-07 ⟹ з реалістичним {1.954e-07/max(real,1e-9):.4e}, '
      f'зі стелею {1.954e-07/max(out[2]/out[0],1e-9):.4e}')
