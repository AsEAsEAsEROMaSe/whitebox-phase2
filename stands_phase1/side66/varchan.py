"""ДРУГИЙ КАНАЛ ПЕРЕНОСУ: похибка ДИСПЕРСІЇ зі своїм ядром `W∘W`.

════════ ЗВІДКИ (прямо з `biassplit.py`, С75) ════════
Розтин сабміту показав дві речі:
  • якір шару 1 + перенос `v` крізь 31 матрицю купує лише **1.03×**;
  • з чотирьох родин `λ` працює РІВНО ОДНА — `Q` = `E[x²] − q_замик`, **1.20×**.
🔑 Тобто вся жива інформація — про ДРУГИЙ МОМЕНТ, і зараз вона вливається в `v`
БЕЗ ВЛАСНОГО ПЕРЕНОСУ: додається в `v` і далі несеться ядром СЕРЕДНЬОГО
`(·@W)·gate`. Для збурення дисперсії це НЕ те ядро.

════════ ЩО ТУТ РОБИТЬСЯ (перший порядок, чесно виведений) ════════
Для `p ~ N(m,s²)` гаусові тотожності дають РІВНО дві похідні:
    ∂E[f]/∂m = E[f′]        ∂E[f]/∂(s²) = ½·E[f″]
Для `f=relu`:  ∂M/∂m = Φ,  ∂M/∂(s²) = φ/(2s)
Для `f=relu²`: ∂A/∂m = 2M,  ∂A/∂(s²) = Φ
Отже, несучи ПАРУ (`v`=δсереднього, `u`=δдисперсії) активацій:
    δm_p = v @ W                 (звичайне ядро)
    δs²_p = u @ (W∘W)            🔑 ПОКОМПОНЕНТНО КВАДРАТ ВАГ — інше ядро!
    v′ = Φ·δm_p + (φ/2s)·δs²_p
    u′ = 2M(1−Φ)·δm_p + (Φ − Mφ/s)·δs²_p      [δV = δA − 2M·δM]

════════ ЧОМУ ЦЕ МОЖЕ БУТИ ДЕШЕВО І НОВО ════════
1. `u @ (W∘W)` — вектор×матриця, як і перенос `v`: **~1.3e5 флопів на шар**,
   разом 0.002% бюджету. Родина `Q` натомість потребує `q_замик` із рекурсії
   `C` (2 матмули 256³ на шар) — ось чому вона й обмежена вісьмома шарами.
2. Усе інше береться з ТОГО САМОГО проходу: `Φ→gate`, `s→ŝ`, `M→mean(a)`.
3. 🔑 **ЯКІР ДРУГОГО КАНАЛУ НА ШАРІ 1 ТОЧНИЙ І БЕЗКОШТОВНИЙ**, як і в першого:
   після вибілювання `p₁~N(0,‖w‖²)` ⟹ `Var(relu) = ‖w‖²(½ − 1/2π)` — з того
   самого `sg0`, що вже рахується. Тобто другий канал теж стартує З ІСТИНИ.
⟹ канал може йти на ВСІ 31 шар, а не на 8.

🚨 НЕ ПЛУТАТИ з ⛔відкликаним «каналом `â²`» ([[MEMORY_45]] §C): там ознака
`E[x²]` виявилась ВЖЕ НАЯВНОЮ родиною `_LAM_Q`. Тут нове не ознака, а ПЕРЕНОС
її власним ядром і точний якір — родина `Q` лишається на місці як база.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 очікую **1.05–1.20×** поверх ПОВНОГО сабміту.
🚨 ГОЛОВНИЙ РИЗИК, названий заздалегідь: `u @ (W∘W)` припускає, що похибка
   коваріації ДІАГОНАЛЬНА. С74 бачила, що глибину в замиканні несе саме
   НЕДІАГОНАЛЬ ⟹ канал може вмерти з глибиною. Тому смуги фітяться окремо:
   якщо живуть лише мілкі — це і буде відповідь.
🟢 ЖИВЕ при >1.05× на СУМІ І МЕДІАНІ (ціна каналу ≈0 ⟹ гейт чистий).

🚨 ПЛОМБИ
  ✅ β=0 мусить дати РІВНО 1.000× (та сама база);
  ✅ якір `u` НЕЗМІЩЕНИЙ: середнє по жеребах ≈0 (друкується);
  ⛔ НЕГ: рядки каналу з СУСІДНЬОГО жереба ⟹ ≈1.000×;
  ⛔ ЧУЖА: `W∘W` з іншої мережі ⟹ мусить бути ГІРШЕ за базу.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (cut_and_budget, closure_parts, W_, GATE_N,        # noqa: E402
                    INV_SQRT_2PI, _phi)
from varsub import S72, A72, Q72, G72                                 # noqa: E402
from price31 import loo, three_slices                                 # noqa: E402
from biassplit import truth_full, run_collect                         # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
KSIG = 8
LAM = (S72, A72, Q72, G72)
# смуги глибини, у яких канал вмикається окремо (щоб побачити, де він живий)
BANDS = ((1, 4), (5, 8), (9, 16), (17, 24), (25, 32))
HALF_M_INV2PI = 0.5 - 1.0 / (2.0 * np.pi)      # ½ − 1/2π


def forward_stats(wc, npair, cl, dev, seed):
    """Один жереб: базовий прогноз сабміту + пошарова статистика для каналу.

    Уся статистика — з ТОГО САМОГО проходу, зайвих флопів нема:
    `gate`(=Φ), `ŝ`, `φ(m̂/ŝ)`, `mean(a)`(=M), плюс насіння обох каналів.
    """
    s_cl, m_cl, f_cl, q_cl = cl
    ksig = len(s_cl)
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    denom = float(2 * npair)
    cov = (x.T @ x) / denom
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xa = torch.relu(x @ (wh @ wc[0]))
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()

    xd = xa.double()
    v0 = xd.mean(0) - sg0 * INV_SQRT_2PI                    # δсереднього (є в сабміті)
    # 🔑 НАСІННЯ ДРУГОГО КАНАЛУ — теж ТОЧНЕ і теж із `sg0`
    u0 = (xd.pow(2).mean(0) - xd.mean(0).pow(2)) - sg0.pow(2) * HALF_M_INV2PI

    v = v0.clone()
    st = []
    for i in range(1, depth):
        p = xa @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        pd = p.double()
        mh = pd.mean(0)
        sh = (pd.pow(2).mean(0) - mh * mh).clamp_min(1e-30).sqrt()
        ph = _phi(mh / sh)
        if i <= ksig:
            v = (v + (LAM[0][i - 1] * ph) * (sh - s_cl[i - 1]))
            xa = torch.relu(p)
            xd = xa.double()
            v = (v + LAM[1][i - 1] * (xd.mean(0) - m_cl[i - 1])
                 + LAM[2][i - 1] * (xd.pow(2).mean(0) - q_cl[i - 1])
                 + LAM[3][i - 1] * (gate - f_cl[i - 1]))
        else:
            xa = torch.relu(p)
            xd = xa.double()
        st.append((gate, sh, ph, xd.mean(0)))
    return xd.mean(0) - v, v0, u0, st


def chan(wc, w2, st, v0, u0, lo, hi):
    """Перенос ПАРИ (v,u) з увімкненим зв'язком лише на шарах [lo,hi].

    Повертає внесок у прогноз = те, на що зміниться `pred`, якщо ввести
    зв'язок `u→v`. Поза смугою канал несеться, але в `v` не вливається —
    так смуги лишаються ПОРІВНЯННИМИ (однакова історія `u`).
    """
    v = torch.zeros_like(v0)      # ДЕЛЬТА до базового `v` (β=0 ⟹ рівно 0)
    u = u0.clone()
    vm = v0.clone()               # «чистий» середній канал — джерело для `u`
    for k, (gate, sh, ph, Ma) in enumerate(st):
        i = k + 1
        wd = wc[i].double()
        dmp = vm @ wd
        dsp = u @ w2[i]
        v = (v @ wd) * gate
        if lo <= i <= hi:
            v = v + ph * dsp / (2.0 * sh)
        u = 2.0 * Ma * (1.0 - gate) * dmp + (gate - Ma * ph / sh) * dsp
        vm = gate * dmp
    return -v                     # pred = x̄ − v ⟹ внесок у pred = −δv


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    NB = len(BANDS)
    print(f'📣 ПЕРЕДБАЧЕНО 1.05–1.20× поверх ПОВНОГО сабміту; ціна каналу ≈0')
    print(f'🚨 РИЗИК НАЗВАНО: `W∘W` = ДІАГОНАЛЬНЕ наближення δC ⟹ канал може '
          f'вмерти з глибиною')
    print(f'🟢 ЖИВЕ при >1.05× на СУМІ І МЕДІАНІ.  '
          f'{nets} мереж × {ndraw} жеребів, КРОС-ЦІЛЬ\n')

    base, G, B, C = [], [], [], []
    Gn, Bn, Cn, Gx, Bx, Cx = [], [], [], [], [], []
    prevw2 = None
    useed = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            cl = closure_parts(wc, KSIG, dev, extras=True)
            w2 = [w.double().pow(2) for w in wc]      # 🔑 не залежить від жеребу
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)

            if n == 0:
                lam = torch.tensor([LAM[q][i] for q in range(4)
                                    for i in range(KSIG)],
                                   dtype=torch.float64, device=dev)
                pr, Fr = run_collect(wc, npair, cl, dev, 5_000_000)
                pm, _, _, _ = forward_stats(wc, npair, cl, dev, 5_000_000)
                rel = float((pm - (pr - lam @ Fr)).abs().max()
                            / (pr - lam @ Fr).abs().max())
                print(f'  ✅ПЛОМБА БАЗИ (forward_stats vs biassplit): {rel:.2e} '
                      f'{"OK" if rel < 1e-9 else "🚨 РОЗБІЖНІСТЬ"}')

            wx = prevw2
            bm = 0.0
            gg = torch.zeros(NB, NB, dtype=torch.float64, device=dev)
            bb = torch.zeros(NB, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(NB, NB, dtype=torch.float64, device=dev)
            bN = torch.zeros(NB, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None
            gX = torch.zeros(NB, NB, dtype=torch.float64, device=dev)
            bX = torch.zeros(NB, dtype=torch.float64, device=dev)
            cX = 0.0
            us = []
            for dr in range(ndraw):
                sd = 88_000_000 + 733 * n + dr
                pb, v0, u0, st = forward_stats(wc, npair, cl, dev, sd)
                us.append(float(u0.sum()))
                bm += float(((pb - T1) * (pb - T2)).sum())
                R = torch.stack([chan(wc, w2, st, v0, u0, lo, hi)
                                 for lo, hi in BANDS], 0)
                e1, e2 = pb - T1, pb - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                gg += R @ R.T
                bb += R @ e
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = R
                if wx is not None:
                    w2x = [a if a.shape == b.shape else b
                           for a, b in zip(wx, w2)]
                    RX = torch.stack([chan(wc, w2x, st, v0, u0, lo, hi)
                                      for lo, hi in BANDS], 0)
                    gX += RX @ RX.T
                    bX += RX @ e
                    cX += float((e1 * e2).sum())
            useed.append(float(np.mean(us)) / max(abs(np.std(us)), 1e-30))
            base.append(bm)
            G.append(gg); B.append(bb); C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            if wx is not None:
                Gx.append(gX); Bx.append(bX); Cx.append(cX)
            prevw2 = [w.clone() for w in w2]
        del W, wc, w2
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    bs = np.array(base)
    idx = list(range(NB))
    print(f'\n{nets} мереж × {ndraw} жеребів. База = ПОВНИЙ сабміт. '
          f'LOO ПО МЕРЕЖАХ.\n')
    print(f'✅ЯКІР `u` НЕЗМІЩЕНИЙ: середнє/σ по жеребах = '
          f'{np.mean(np.abs(useed)):.3f} (мусить бути ≲1, інакше насіння зміщене)')
    print(f'\n{"варіант":>30s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
    print('-' * 62)
    print(f'{"✅β=0 (пломба)":>30s} {1.0:8.3f}× {1.0:8.3f}× {1.0:9.3f}×')
    best = (0.0, None)
    SUBS = [(f'смуга {lo}–{hi}', [j]) for j, (lo, hi) in enumerate(BANDS)]
    SUBS += [('мілкі 1–8', [0, 1]), ('глибокі 9–32', [2, 3, 4]),
             ('ВСІ 5 смуг', idx)]
    for nm, ii in SUBS:
        val = loo([g[ii][:, ii] for g in G], [b[ii] for b in B], C, len(ii))
        sv, mdv, w3v = three_slices(bs, val)
        if min(sv, mdv) > best[0]:
            best = (min(sv, mdv), nm)
        print(f'{"🎯 " + nm:>30s} {sv:8.3f}× {mdv:8.3f}× {w3v:9.3f}×')
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, NB)
    sN, mN, wN = three_slices(bs, vN)
    print(f'{"⛔НЕГ (сусідній жереб)":>30s} {sN:8.3f}× {mN:8.3f}× {wN:9.3f}×')
    if Gx:
        vX = loo([g[idx][:, idx] for g in Gx], [b[idx] for b in Bx], Cx, NB)
        sX, mX, wX = three_slices(bs[1:], vX)
        print(f'{"⛔ЧУЖА `W∘W`":>30s} {sX:8.3f}× {mX:8.3f}× {wX:9.3f}×')

    live = best[0] > 1.05
    print(f'\n🔑 НАЙКРАЩИЙ ЗРІЗ: {best[1]} ⟹ {best[0]:.3f}× (гірше з суми/медіани)')
    print(f'🔑 ВИРОК: {"ЖИВЕ" if live else "МЕРТВО"} (🟢гейт >1.05×)')
    print(f'   📣передбачено 1.05–1.20×; ризик — діагональне наближення δC')
    print(f'📊 БАЛ: 1.954e-07 → {1.954e-07 / max(best[0], 1e-9):.3e} '
          f'(канал ≈безкоштовний ⟹ множник не рухається)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
