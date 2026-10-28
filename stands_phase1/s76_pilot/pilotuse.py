"""№1: ПІЛОТНІ СЕМПЛИ РАХУЮТЬСЯ І ВИКИДАЮТЬСЯ — ЗАБРАТИ ЇХ БЕЗКОШТОВНО. (С76)

════════ ФАКТ ІЗ КОДУ САБМІТУ (`estimator_side1.py`, рядки 155-160) ════════
    a = xp
    for i in range(depth):
        h = fnp.matmul(a, mlp.weights[i])
        if i < depth - 1: alive.append(fnp.max(h, axis=0) >= 0.0)
        a = fnp.maximum(h, 0.0)
Пілот проганяє 256 семплів (128 АНТИТЕТИЧНИХ пар) через **ПОВНУ** мережу до
самого виходу. Після циклу `a` = relu(вихід) — і воно НЕ ВИКОРИСТОВУЄТЬСЯ.
Флопи вже оплачені (3.9% бюджету), результат викидається.

🔑 І це оцінка РІВНО тієї величини, яку міряє грейдер: `E[f]` ПОВНОЇ мережі.
   Наша основна гілка йде через ОБРІЗАНУ і має зміщення обрізання (36% ‖b‖²).
✅ НЕЗМІЩЕНІСТЬ ЧИСТА: вихід повної мережі на пілотному семплі не залежить від
   того, кого ми потім обрізали ⟹ обумовленості немає.

════════ ЩО МІРЯЄМО ════════
`pred(w) = (1−w)·pred_осн + w·pred_пілот`, свіп `w`, вибір LOO ПО МЕРЕЖАХ.
Крос-ціль (дві незалежні істини ПОВНОЇ мережі) ⟹ шум істини не входить.
MSE(w) — квадратична за `w`, тож уся крива береться з ТРЬОХ скалярів на
(мережа, жереб): ⟨d1,d2⟩, ⟨d1,e2⟩+⟨e1,d2⟩, ⟨e1,e2⟩, де d = pred_осн − T,
e = pred_пілот − T.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 пілотна оцінка у **50–60×** гірша за основну (128 антитетичних пар проти
   ~2930, помножених на 2.45× від вибілювання й КЗ);
📣 оптимальне `w` ≈ **0.017**, виграш **1.015–1.022×**;
📣 `w_опт` мусить збігтися з арифметикою `MSE_осн/(MSE_осн+MSE_пілот)` — якщо
   ні, десь витік або зміщення.
🟢 ЖИВЕ при виграші ≥1.010× (ціна НУЛЬ флопів ⟹ гейт нижчий за звичайний 1.05).

🚨 ПЛОМБИ
  ✅ w=0 мусить дати РІВНО базу;
  ✅ пілотна оцінка НЕЗМІЩЕНА: ⟨e1,e2⟩ по мережах ≈ чиста дисперсія, а
     середнє `e` не мусить систематично збігатися з `d` (інакше спільне
     джерело); друкуємо corr(d, e) — має бути ≈0;
  ✅ пілот того самого сіду, що в сабміті (SEED=12345) ⟹ детермінований на
     мережу, як і в бою.
"""
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (closure_parts, run_side1 as run72, W_,               # noqa: E402
                    PILOT_PAIRS, TARGET_UTIL, B_BUDGET, MIN_PAIRS, SEED)
from biassplit import truth_full                                         # noqa: E402
from varsub import S72, A72, Q72, G72                                    # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:1'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 16
KSIG = 8
WGRID = np.concatenate([[0.0], np.geomspace(2e-3, 0.25, 24)])


def cut_and_pilot(W, dev):
    """`cut_and_budget` сабміту + ПІЛОТНА ОЦІНКА ВИХОДУ, яку сабміт викидає.

    Рядок у рядок як у `base72.cut_and_budget`; єдина відмінність — не
    втрачаємо `a` після циклу."""
    depth = len(W)
    rng = np.random.default_rng(SEED)
    zp = torch.tensor(rng.standard_normal((PILOT_PAIRS, W_)).astype(np.float32),
                      device=dev)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(depth):
        h = a @ W[i]
        if i < depth - 1:
            alive.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    pilot = a.double().mean(0)          # ← ОСЬ ВОНО. Уже оплачене.
    wc = []
    for i in range(depth):
        wi = W[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < depth - 1:
            wi = wi[:, alive[i]]
        wc.append(wi.contiguous())
    per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * depth * W_ * W_)
    left = TARGET_UTIL * float(B_BUDGET) - used
    return wc, max(MIN_PAIRS, int(left / (2.0 * max(per_cut, 1.0)))), pilot


def main(nets=32, dev=DEV, ndraw=NDRAW):
    t0 = time.time()
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'№1 ПІЛОТ ЯК СЕМПЛИ · {nets} мереж × {ndraw} жеребів, крос-ціль, '
          f'{dev}')
    print('📣 ПЕРЕДБАЧЕНО: пілот у 50–60× гірший · w_опт≈0.017 · '
          'виграш 1.015–1.022×')
    print('🟢 ЖИВЕ при ≥1.010× (ціна НУЛЬ флопів)\n', flush=True)

    A = np.zeros((nets, 3))          # ⟨d1,d2⟩, ⟨d1,e2⟩+⟨e1,d2⟩, ⟨e1,e2⟩
    P_ = np.zeros(nets)              # ⟨e1,e2⟩ окремо (MSE пілота)
    CORR, NP_ = [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair, pilot = cut_and_pilot(W, dev)
            NP_.append(npair)
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, KSIG, dev, extras=True)
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)
            e1, e2 = pilot - T1, pilot - T2
            acc = np.zeros(3)
            cds = []
            for k in range(ndraw):
                pred, _ = run72(wc, npair, s_cl, m_cl, dev,
                                71_000_000 + 977 * n + k,
                                lam_s=S72, lam_a=A72, lam_q=Q72, lam_g=G72,
                                q_ext=q_cl, p_ext=f_cl)
                d1, d2 = pred - T1, pred - T2
                acc[0] += float((d1 * d2).sum())
                acc[1] += float((d1 * e2).sum() + (e1 * d2).sum())
                acc[2] += float((e1 * e2).sum())
                cds.append(float(torch.dot(d1, e1)
                                 / (d1.norm() * e1.norm()).clamp_min(1e-30)))
            A[n] = acc / ndraw
            P_[n] = float((e1 * e2).sum())
            CORR.append(np.mean(cds))
        del W, wc
        torch.cuda.empty_cache()
        if (n + 1) % 4 == 0:
            print(f'  … {n+1}/{nets}  ({time.time()-t0:.0f}s)', flush=True)

    def mse(w, idx):
        return np.mean((1 - w) ** 2 * A[idx, 0] + w * (1 - w) * A[idx, 1]
                       + w ** 2 * A[idx, 2])

    base = mse(0.0, np.arange(nets))
    pil = float(np.mean(P_))
    print(f'\nnpair сер {np.mean(NP_):.0f} · пілот 2×{PILOT_PAIRS}=256 семплів')
    print(f'MSE основна  {base:.5e}')
    print(f'MSE пілотна  {pil:.5e}   ({pil/base:.1f}× гірша)')
    w_ar = base / (base + pil)
    print(f'\nАРИФМЕТИКА: w_опт = MSE_осн/(MSE_осн+MSE_піл) = {w_ar:.5f} '
          f'⟹ очікуваний виграш {1/(1-w_ar):.4f}×')
    cc = float(np.mean(CORR))
    print(f'✅ПЛОМБА НЕЗАЛЕЖНОСТІ: corr(похибка_осн, похибка_пілот) = {cc:+.4f} '
          f'{"✅" if abs(cc) < 0.12 else "🛑 СПІЛЬНЕ ДЖЕРЕЛО"}')

    # LOO ПО МЕРЕЖАХ: `w` обирається на решті мереж
    tot = 0.0
    ws = []
    for n in range(nets):
        tr = np.array([q for q in range(nets) if q != n])
        wb = WGRID[int(np.argmin([mse(w, tr) for w in WGRID]))]
        ws.append(wb)
        tot += mse(wb, np.array([n]))
    loo = tot / nets
    print(f'\n{"w":>8} {"MSE":>13} {"×BASE":>8}')
    for w in (0.0, 0.005, 0.01, 0.0175, 0.025, 0.05, 0.1):
        m = mse(w, np.arange(nets))
        print(f'{w:>8.4f} {m:>13.5e} {base/m:>8.4f}')
    print(f'\nLOO ПО МЕРЕЖАХ (w обране на решті): {loo:.5e} '
          f'⟹ **{base/loo:.4f}×**  (медіана w = {np.median(ws):.4f})')
    ok = base / loo >= 1.010
    print(f'\n🔑 ВИРОК: {"🟢 ЖИВЕ" if ok else "⛔ МЕРТВО"} при гейті 1.010×')
    print(f'📊 БАЛ: 1.954e-07 → {1.954e-07/(base/loo):.4e} '
          f'(якщо перенесеться на грейдер)')
    print(f'   час {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 32)
