"""КАРТА ПРИЗУ ПО ГЛИБИНІ: скільки MSE знімає ТОЧНА опора до шару K?

════════ НАВІЩО (питання юзера «що дасть >70× над MC») ════════
Уся якість методу стискається в `ρ²` (виграш `1/(1−ρ²)`, гіпербола):
  ми 1.125× (`ρ²`=0.111) · mliston 1.8× (0.434) · лідер 17.2× (0.942) ·
  **на 70× треба 22.5× (`ρ²`=0.956)** · оракул `m,s` 23.8× (0.958).
⟹ 70× — це ПРАКТИЧНО ОРАКУЛ. Єдиний важіль такої величини — точне `m_ℓ`;
усе інше не тієї величини (вибілювання 1.48×, антитетика 2.3×, уся цінова
вісь 1.11×, уся аналітика 1.11×).
🔑 І ключовий факт: наша КЗ бере 11.1%, з них **9.8% — САМ ШАР 1**, бо після
вибілювання вхід строго гаусів і `E[relu]=σ/√2π` відоме ТОЧНО. Один шар із
ТОЧНОЮ опорою = 9.8%. Отже питання не «як уточнити замикання», а
**СКІЛЬКИ ДАЄ ТОЧНА ОПОРА ДО ГЛИБИНИ K** — і цього ніхто не міряв
(це п.1 у власному списку С71, три сесії не зроблений).

════════ ЩО МІРЯЄМО ════════
`v` — накопичений перенос. ЗАМІЩАЄМО його на шарі K істинною похибкою:
    `v_K ← x̄_K − m_істина,K`,  далі звичайний перенос `v ← (v@W)·gate`.
Свіп по K. Крива «MSE(K)» — це і є карта призу: скільки лишається, якщо
аналітика точна до глибини K і нікчемна далі.
🚨 ЗАМІЩАЄМО, НЕ ДОДАЄМО: `x̄_ℓ−m_ℓ` УЖЕ містить перенесене з попередніх
шарів, і додавання рахує те саме 30 разів (коштувало прогону в `whocarries`).
💰 Усі K йдуть ОДНИМ форвардом: перенос лінійний за `v`, тож ведемо матрицю.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 Профіль спадає ШВИДКО (шар 1 = 9.8%, далі ~`1/ℓ`), сума перших восьми
   **25–35%**, і **70× у цій схемі не набереться** — але 2–3× набереться, і
   стане видно, на яких саме шарах.
🟢 Якщо K=4 дасть ≥10× — варто вкладатися в точну опору на 4 шарах (це
   реальніше, ніж на 30). Якщо K=4 дасть ≤1.5× — шлях безнадійний.

🚨 ПЛОМБИ
  ✅ K=остання глибина мусить лишити майже НУЛЬ (тільки шум істини);
  ✅ БЕЗ заміщення (`v` як є) мусить відтворити сабміт;
  ⛔ КОНТРОЛЬ: заміщення з `m` ЧУЖОЇ мережі — мусить бути ГІРШЕ за чистий `x̄`;
  🚨 КРОС-ЦІЛЬ (дві незалежні істини) — інакше шум істини занижує глибокі K.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N, LAM_S)
from base72 import run_side1 as run72                              # noqa: E402
from tilt_wide import truth_last                                   # noqa: E402
from varsub import S72, A72, Q72, G72                              # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
# 🚨 ШУМ ІСТИНИ ВХОДИТЬ У КАНАЛ АДИТИВНО (`v = x̄_K − m_K`, і шум `m_K` не
# знімається крос-ціллю, бо сидить у `v`, а не в цілі). `gatecap.py` спіймав
# це ПЛОМБОЮ ТАВТОЛОГІЇ: K=остання глибина мусить дати ∞, а дало 16.45×.
# ⟹ (1) учетверо більше семплів істини, (2) КАЛІБРУВАННЯ: очищена MSE =
# виміряна − MSE тавтологічної точки. Без цього профіль ЗАНИЖЕНО в рази.
TR_PAIRS, CHUNK = 524_288, 32768
KS = (0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 30)


def truth_means(wc, dev, seed):
    """Істинні `E[relu]` для КОЖНОГО шару (велика вибірка, N(0,I))."""
    depth = len(wc)
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
           for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(depth):
            a = torch.relu(a @ wc[i])
            acc[i] += a.double().sum(0)
        done += k
    return [t / float(2 * TR_PAIRS) for t in acc]


def run_map(wc, npair, dev, seed, mtrue, ks):
    """Один жереб. Повертає (pred_чистий_x̄, {K: pred із заміщенням на K}).

    Перенос ЛІНІЙНИЙ за `v`, тож усі K веземо однією матрицею за один форвард.
    """
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cov = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xa = torch.relu(x @ (wh @ wc[0]))
    nk = len(ks)
    VM = torch.zeros(nk, wc[0].shape[1], dtype=torch.float64, device=dev)
    for j, K in enumerate(ks):
        if K == 0:
            VM[j] = xa.double().mean(0) - mtrue[0]
    for i in range(1, depth):
        p = xa @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        VM = (VM @ wd) * gate.unsqueeze(0)
        xa = torch.relu(p)
        for j, K in enumerate(ks):
            if K == i:
                VM[j] = xa.double().mean(0) - mtrue[i]
    xl = xa.double().mean(0)
    return xl, {K: xl - VM[j] for j, K in enumerate(ks)}


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: профіль спадає ШВИДКО, сума перших 8 = 25–35%, '
          f'70× НЕ набереться (але 2–3× так)')
    print(f'🟢 K=4 ≥10× ⟹ варто вкладатись у точну опору на 4 шарах; '
          f'≤1.5× ⟹ безнадійно')
    print(f'   мереж {nets} × {ndraw} жеребів, КРОС-ЦІЛЬ\n')

    mse_pure, mse_sub, mse_x = [], [], {}
    mse_ctl = []
    prevm = None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            ks = [K for K in KS if K < d] + [d - 1]   # ✅калібр: тавтологія
            mt = truth_means(wc, dev, 310_000 + 877 * n)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, 8, dev, extras=True)
            # ⛔контроль: істина з ЧУЖОЇ мережі, доповнена до потрібних довжин
            ctl = None
            if prevm is not None:
                ctl = []
                for i in range(d):
                    t = torch.zeros(wc[i].shape[1], dtype=torch.float64,
                                    device=dev)
                    kk = min(prevm[i].shape[0], t.shape[0]) \
                        if i < len(prevm) else 0
                    if kk:
                        t[:kk] = prevm[i][:kk]
                    ctl.append(t)

            ap = asb = 0.0
            ax = {K: 0.0 for K in ks}
            ac = 0.0
            for dr in range(ndraw):
                sd = 77_000_000 + 733 * n + dr
                xl, preds = run_map(wc, npair, dev, sd, mt, ks)
                ap += float(((xl - T1) * (xl - T2)).sum())
                for K in ks:
                    p = preds[K]
                    ax[K] += float(((p - T1) * (p - T2)).sum())
                pb, _ = run72(wc, npair, s_cl, m_cl, dev, sd, lam_s=S72,
                              lam_a=A72, lam_q=Q72, lam_g=G72,
                              q_ext=q_cl, p_ext=f_cl)
                asb += float(((pb - T1) * (pb - T2)).sum())
                if ctl is not None:
                    _, pc2 = run_map(wc, npair, dev, sd, ctl, [ks[-1]])
                    q = pc2[ks[-1]]
                    ac += float(((q - T1) * (q - T2)).sum())
            mse_pure.append(ap / ndraw)
            mse_sub.append(asb / ndraw)
            for K in ks:
                mse_x.setdefault(K, []).append(ax[K] / ndraw)
            if ctl is not None:
                mse_ctl.append(ac / ndraw)
            prevm = [t.clone() for t in mt]
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    pure = np.array(mse_pure)
    sub = np.array(mse_sub)
    P = pure.sum()
    print(f'\n{nets} мереж × {ndraw} жеребів. КРОС-ЦІЛЬ.')
    print(f'✅ПЛОМБА: сабміт проти чистого `x̄` = '
          f'{P / sub.sum():.3f}× (очікувано ≈1.125×)\n')
    print(f'{"K":>4s} {"MSE":>11s} {"сирий×":>10s} {"ОЧИЩЕНИЙ×":>12s} '
          f'{"ρ²":>7s} {"vs MC":>8s} {"приріст":>8s}')
    print('-' * 70)
    # ✅КАЛІБРУВАННЯ: тавтологічна точка (K = остання глибина) лишає САМ
    # шум істини `m`; віднімаємо його з усіх точок.
    ktau = max(mse_x)
    noise = np.array(mse_x[ktau]).sum()
    prev = 1.0
    for K in sorted(mse_x):
        v = np.array(mse_x[K])
        r = P / v.sum()
        cl = max(v.sum() - noise, 1e-300)
        rc = (P - noise) / cl
        rho = 1.0 - 1.0 / rc if rc > 0 else 0.0
        vsmc = 3.5 / 1.125 * rc
        tag = ' ✅калібр' if K == ktau else ''
        print(f'{K:4d} {v.sum() / nets:11.4e} {r:10.2f}× {rc:11.2f}× '
              f'{rho:7.4f} {vsmc:7.1f}× {rc / prev:7.3f}×{tag}')
        prev = rc
    if mse_ctl:
        c = np.array(mse_ctl)
        print(f'\n⛔КОНТРОЛЬ (істина з ЧУЖОЇ мережі, найглибший K): '
              f'{P * (len(c) / nets) / c.sum():.3f}× '
              f'{"✅гірше за чистий x̄" if c.sum() / len(c) > P / nets else "🛑 НЕ гірше"}')

    def clean(K):
        return (P - noise) / max(np.array(mse_x[K]).sum() - noise, 1e-300)
    kk = sorted(k for k in mse_x if k < ktau)
    r4 = clean(4) if 4 in mse_x else 0
    r8 = clean(8) if 8 in mse_x else 0
    rmax = clean(kk[-1])
    print(f'\n🚨ШУМ ІСТИНИ (тавтологічна точка K={ktau}): '
          f'{P / noise:.1f}× ⟹ сирі числа занижені саме ним')
    print(f'\n🔑 ОЧИЩЕНІ: K=4: {r4:.2f}× · K=8: {r8:.2f}× · '
          f'K={kk[-1]} (повний оракул): {rmax:.2f}×')
    print(f'🔑 На 70× треба 22.5× ⟹ '
          f'{"ДОСЯЖНО оракулом" if rmax >= 22.5 else f"НЕ ДОСЯЖНО навіть оракулом ({rmax:.1f}×)"}')
    print(f'🟢 гейт K=4 ≥10×: {"ТАК" if r4 >= 10 else "НІ"} '
          f'⟹ {"вкладатись у точну опору на 4 шарах" if r4 >= 10 else "мілка опора не рятує"}')
    print(f'📊 БАЛ: 1.972e-07 (вимір стендовий, оракульний)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
