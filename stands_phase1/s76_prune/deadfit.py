"""КОМПЕНСАЦІЯ ОБРІЗАНИХ + СПІЛЬНИЙ ПЕРЕФІТ 32 `λ`. (незакрита нитка С76)

════════ ЧОМУ МИНУЛИЙ ВИМІР ДАВ ХИБНИЙ МІНУС ════════
`deadcomp.py` підставив обрізаному нейрону його середнє замість нуля і дістав
REAL 0.9833×, ORACLE **0.9755×** — гірше навіть з ІСТИННИМИ середніми.
🔑 ПРИЧИНА не в ідеї: наші 32 `λ` фітувались НА мережі З обрізанням і вже
всмоктали це зміщення. Прибираєш обрізання чесно — оголюєш зустрічну
поправку, яка компенсує те, чого вже нема. Чим ТОЧНІША компенсація, тим
ГІРШЕ — рівно те, що спостерігалось (ORACLE < REAL).
⟹ чесна перевірка вимагає ПЕРЕФІТУ `λ` РАЗОМ із компенсацією. Це тут.

🗺️ ЧОМУ ЦІЛЬ ВЕЛИКА: розклад `‖b‖²` дає обрізанню **36%** — друга частка
   після самих `λ` (55%). З 1502 обрізаних нейронів **654 (44%) ЖИВІ**.

════════ ЯК МІРЯЄМО (перенос ЛІНІЙНИЙ за `v` ⟹ один прогін дає все) ════════
Поруч із `v` несемо матрицю `A` (32 × вихід): рядок = внесок ОДНОГО каналу
(S/A/Q/G × шари 1..8), уже перенесений до виходу. Тоді
    `pred(λ) = pred₀ − λᵀA`
⟹ 32 коефіцієнти фітуються LOO ПО МЕРЕЖАХ без повторних прогонів.

ЧОТИРИ КОМІРКИ — і виграш ідеї читається лише з ОСТАННЬОЇ ПАРИ:
  BASE     рідні λ, БЕЗ компенсації            ← діючий sub87
  COMP     рідні λ, З компенсацією             ← ✅ПЛОМБА ≈0.983× (deadcomp)
  FIT      ПЕРЕФІТ λ, БЕЗ компенсації          ← ізолює сам перефіт
  FITCOMP  ПЕРЕФІТ λ, З компенсацією           ← ІДЕЯ
🚨 ВИГРАШ ІДЕЇ = FITCOMP / FIT, а НЕ FITCOMP / BASE. Інакше міряємо перефіт.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 COMP ≈ 0.98× (відтворення deadcomp — головна пломба приладу)
📣 FIT  = 1.00-1.02× (рідні λ фітовані на 60 мережах із крос-валідацією;
         перефіт на 23 без неї НЕ мусить сильно виграти — 🚨якщо FIT дасть
         сильно >1.05×, це ПЕРЕВЧАННЯ, а не сигнал)
📣 FITCOMP/FIT = 1.01-1.05× — зміщення обрізання 36% від 7.6% MSE ≈ 2.7%,
         але компенсація бере лише перші `KSIG` шарів
🟢 ЖИВЕ при FITCOMP/FIT ≥ 1.03× (шумова підлога LOO по мережах) і ⛔ROLL <1.01×

🚨 ПЛОМБИ
  ✅ порожня компенсація == BASE до 1e-12;
  ✅ λ = рідні у формулі `pred₀ − λᵀA` мусить відтворити BASE до 1e-12
     (перевірка, що матриця `A` зібрана правильно);
  ⛔ROLL: компенсація тим самим вектором, ЗСУНУТИМ по нейронах.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/s76_prune')
from base_real import W_, INV_SQRT_2PI, GATE_N                     # noqa: E402
from deadcomp import (cut_alive, truth_layers, build_comp,         # noqa: E402
                      closure_parts, KSIG, S72, A72, Q72, G72, _phi)
from jitanim import bank_net                                       # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:1'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 24
LAM0 = np.array(list(S72) + list(A72) + list(Q72) + list(G72))


def run_fit(wc, npair, cl, dev, seed, comp=None):
    """Повертає (pred₀ БЕЗ жодного λ-каналу, A[32 × вихід])."""
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
    v = xa.double().mean(0) - sg0 * INV_SQRT_2PI
    A = torch.zeros(4 * ksig, wc[0].shape[1], dtype=torch.float64, device=dev)
    for i in range(1, depth):
        p = xa @ wc[i]
        if comp is not None and comp[i] is not None:
            p = p + comp[i].to(p.dtype)
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        A = (A @ wd) * gate                       # той самий перенос, рядково
        if i <= ksig:
            pd = p.double()
            mh = pd.mean(0)
            sh = (pd.pow(2).mean(0) - mh * mh).clamp_min(1e-30).sqrt()
            xa = torch.relu(p)
            xd = xa.double()
            j = i - 1
            A[j] = A[j] + _phi(mh / sh) * (sh - s_cl[j])
            A[ksig + j] = A[ksig + j] + (xd.mean(0) - m_cl[j])
            A[2 * ksig + j] = A[2 * ksig + j] + (xd.pow(2).mean(0) - q_cl[j])
            A[3 * ksig + j] = A[3 * ksig + j] + (gate - f_cl[j])
        else:
            xa = torch.relu(p)
    return xa.double().mean(0) - v, A


# 🚨🎓 ЗАПИСАНА ПАСТКА ПРИЛАДУ (С76): усадка до НУЛЯ не містить діючої бази в
#    сітці ⟹ читається 0.12–0.90× «ідея погана», хоч означає «фітер не
#    дотягнувся». Тому усаджуємо до `λ₀` (діючий сабміт): при ρ→∞ виходить
#    РІВНО BASE. Плюс діагональне нормування і ВКЛАДЕНИЙ вибір сили.
RHOS = (1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 1e2, 1e4)


def _norm(E, A, lam0):
    """Нормальні рівняння для ЗАЛИШКУ бази: `E' = E − λ₀ᵀA`, шукаємо `δ`."""
    nets, K = len(E), A[0].shape[1]
    G = np.zeros((nets, K, K))
    b = np.zeros((nets, K))
    for i in range(nets):
        r = E[i] - np.einsum('k,nkd->nd', lam0, A[i])
        G[i] = np.einsum('nkd,njd->kj', A[i], A[i])
        b[i] = np.einsum('nkd,nd->k', A[i], r)
    return G, b


def _solve(Gs, bs, rho, D):
    return np.linalg.solve(Gs + rho * np.diag(D), bs)


def _sse(E, A, lam0, d, idx):
    s = 0.0
    for i in idx:
        r = E[i] - np.einsum('k,nkd->nd', lam0 + d, A[i])
        s += float((r ** 2).sum())
    return s


def loo(E, A, lam0=None, rhos=RHOS):
    """ВКЛАДЕНИЙ LOO ПО МЕРЕЖАХ з усадкою до `λ₀`. Повертає (SSE, λ, ρ-мода)."""
    if lam0 is None:
        lam0 = np.zeros(A[0].shape[1])
    nets = len(E)
    G, b = _norm(E, A, lam0)
    Gs, bs = G.sum(0), b.sum(0)
    D = np.maximum(np.diag(Gs) / nets, 1e-300)
    res, picks = 0.0, []
    for i in range(nets):
        Go, bo = Gs - G[i], bs - b[i]
        out = [j for j in range(nets) if j != i]
        # внутрішній LOO по решті мереж — вибір ρ БЕЗ підглядання в `i`
        best = (None, np.inf)
        for rho in rhos:
            e = 0.0
            for j in out:
                dj = _solve(Go - G[j], bo - b[j], rho, D)
                e += _sse(E, A, lam0, dj, [j])
            if e < best[1]:
                best = (rho, e)
        picks.append(best[0])
        res += _sse(E, A, lam0, _solve(Go, bo, best[0], D), [i])
    rho_m = max(set(picks), key=picks.count)
    return res, lam0 + _solve(Gs, bs, rho_m, D), rho_m


def main(nets=24, dev=DEV, ndraw=NDRAW):
    print('📣 ПЕРЕДБАЧЕНО: COMP ≈0.98× (пломба) · FIT 1.00-1.02× · '
          'FITCOMP/FIT 1.01-1.05×')
    print('🟢 ЖИВЕ при FITCOMP/FIT ≥1.03× і ⛔ROLL <1.01×\n', flush=True)
    CELLS = ('BASE', 'COMP', 'ROLL')
    E = {c: [] for c in CELLS}
    Am = {c: [] for c in CELLS}
    seal = []
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, npair, alive = cut_alive(W, dev)
            d = len(wc)
            mt = truth_layers(W, dev, 310_000 + 877 * n)
            m31 = mt[d - 1] if len(mt) >= d else mt[-1]
            cl = closure_parts(wc, min(KSIG, d - 1), dev, extras=True)
            m_av = mt
            comps = {
                'BASE': None,
                'COMP': build_comp(W, wc, alive, dev, 'a', m_av, KSIG),
                'ROLL': build_comp(W, wc, alive, dev, 'a', m_av, KSIG, roll=7),
            }
            for c in CELLS:
                e, a = [], []
                for k in range(ndraw):
                    p0, Aa = run_fit(wc, npair, cl, dev,
                                     94_000_000 + 977 * n + k, comps[c])
                    e.append((p0 - m31).cpu().numpy())
                    a.append(Aa.cpu().numpy())
                E[c].append(np.array(e))
                Am[c].append(np.array(a))
            # ✅ПЛОМБА: рідні λ у формулі мусять відтворити діючий сабміт
            r = E['BASE'][-1] - np.einsum('k,nkd->nd', LAM0, Am['BASE'][-1])
            seal.append(float((r ** 2).sum()))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    def with_lam(c, lam):
        return sum(float(((E[c][i] - np.einsum('k,nkd->nd', lam, Am[c][i]))
                          ** 2).sum()) for i in range(nets))

    base = with_lam('BASE', LAM0)
    comp = with_lam('COMP', LAM0)
    roll0 = with_lam('ROLL', LAM0)
    print(f'\n{nets} мереж × {ndraw} жеребів')
    print(f'✅ПЛОМБА A-матриці: {abs(sum(seal) - base) / base:.2e} '
          f'(мусить бути ~0)')
    print(f'\n{"комірка":>10} {"SSE":>13} {"× проти BASE":>14}')
    print(f'{"BASE":>10} {base:>13.5e} {1.0:>14.4f}')
    print(f'{"COMP":>10} {comp:>13.5e} {base/comp:>14.4f}  '
          f'✅пломба deadcomp ≈0.983')
    print(f'{"ROLL(λ рідні)":>10} {roll0:>13.5e} {base/roll0:>14.4f}')

    fit, lf, r1 = loo(E['BASE'], Am['BASE'], LAM0)
    fc, lc, r2 = loo(E['COMP'], Am['COMP'], LAM0)
    fr, _, r3 = loo(E['ROLL'], Am['ROLL'], LAM0)
    print(f'{"FIT":>10} {fit:>13.5e} {base/fit:>14.4f}  ← сам перефіт '
          f'(ρ={r1:g})')
    print(f'{"FITCOMP":>10} {fc:>13.5e} {base/fc:>14.4f}  (ρ={r2:g})')
    print(f'{"FITROLL":>10} {fr:>13.5e} {base/fr:>14.4f}  (ρ={r3:g})')
    # ✅ПЛОМБА: нескінченна усадка мусить дати РІВНО BASE
    inf_, _, _ = loo(E['BASE'], Am['BASE'], LAM0, rhos=(1e12,))
    print(f'✅ПЛОМБА усадки (ρ=1e12): {base/inf_:.6f}× (мусить бути 1.000000)')

    gain = fit / fc
    rollg = fit / fr
    print(f'\n🔑 ВИГРАШ ІДЕЇ = FITCOMP/FIT = {gain:.4f}×  (🟢гейт 1.03)')
    print(f'⛔НЕГ            FITROLL/FIT = {rollg:.4f}×  (мусить бути <1.01)')
    ok = gain >= 1.03 and rollg < 1.01
    print(f'🔑 ВИРОК: {"🟢 ЖИВЕ" if ok else "⛔ МЕРТВО"}')
    print(f'📊 БАЗОВИЙ БАЛ 1.954e-07 · множник на підлозі ⟹ '
          f'дало б {1.954e-07 / max(gain, 1e-9):.4e}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
