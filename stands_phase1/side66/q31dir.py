"""БАГАТОНАПРЯМКОВИЙ КАНАЛ `q̄₃₁−q_замик`: чи несе векторна структура більше?

ЗВІДКИ (`q31use.py`). Форма правила вичерпана: довільна функція від скаляра
каналу ГІРША за пряму (BIN 1.0412 проти SCAL 1.0804), бо `E[похибка|c]` точно
лінійна за ЦГТ. Але там же видно, куди йти:
    LIN (ПОВНИЙ вектор, 1 параметр) 1.0970  >  SCAL (скаляр вздовж û) 1.0804
⟹ **+0.017 несе ВЕКТОРНА структура, а не форма правила.**
Питання цього стенду: якщо дати каналу не одну `λ`, а по одній НА НАПРЯМОК —
скільки ще лежить?

════════ ЯКІ НАПРЯМКИ І ЧОМУ ВОНИ ДОСТУПНІ ════════
`e₀` = напрямок `m_замик,₃₁` (детермінований, з ваг);
`e₁..e_r` = головні власні вектори `C_замик,₃₁`, ортогоналізовані до `e₀`.
🔑 усе це рахує САМЕ замикання, яке канал і так вимагає ⟹ ДОДАТКОВА ЦІНА НУЛЬ.
🔑 `λ_k` індексуються НОМЕРОМ власного числа, а не самим вектором ⟹ переносяться
між мережами (вектори різні, порядок — ні).

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 r=0 (лише `e₀`) відтворить ≈SCAL, тобто ~1.08×;
📣 приріст із `r` НАСИЧУЄТЬСЯ близько r=4..8 — PR-ранг шуму на виході 5.9,
   отже незалежних напрямків там ~6;
📣 стеля всієї конструкції **1.10–1.15×** над поглибленою базою.
🟢 ЖИВЕ, якщо найкраще `r` дає ≥1.15× НАД LIN (бо LIN уже 1.097×, а нетто
   після ціни поглиблення має перевалити 1.05×).

🚨 ПЛОМБИ
  ✅ ВЛАСНЕ ЗАМИКАННЯ: `q` з `closure_full` мусить збігтися з `closure_parts`
     до машинної точності (інакше моя копія рекурсії брехлива);
  ✅ r=0 ≈ SCAL із попереднього стенду;
  ⛔ НЕГ-1 «сусідній жереб» — СЛАБКИЙ контроль (не руйнує мережеву складову);
  ⛔ НЕГ-2 ВИПАДКОВИЙ вектор тієї ж норми — СИЛЬНИЙ контроль;
  📏 обидва дають шумову підлогу; виграш рахується НАД НЕЮ, а не над 1.000.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI)
from jitprof import truth_means                                    # noqa: E402
from jitanim import bank_net                                       # noqa: E402
from q31ab import run, budget, KSIG                                # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
RS = (0, 1, 2, 4, 8, 16, 32)
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_P = 0.3275911


def _phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQRT_2PI


def _Phi(x):
    ax = x.abs() * 0.7071067811865476
    t = 1.0 / (1.0 + _P * ax)
    poly = ((((_A[4] * t + _A[3]) * t + _A[2]) * t + _A[1]) * t + _A[0]) * t
    return 0.5 * (1.0 + torch.sign(x) * (1.0 - poly * torch.exp(-ax * ax)))


def closure_full(wc, dev):
    """Та сама рекурсія, що `closure_parts`, але ще й ПОВЕРТАЄ `C` на виході."""
    d = len(wc)
    m = torch.zeros(wc[0].shape[0], dtype=torch.float32, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float32, device=dev)
    q_last = None
    for i in range(d):
        w = wc[i]
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(
            ph / s, ph / s)
        dd = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(dd - C.diagonal())
        q_last = dd + m * m
    return m.double(), C.double(), q_last.double()


def loo_dirs(E, P, D):
    """LOO ПО МЕРЕЖАХ: поправка `Σ_k λ_k · p_k · d_k`. P — проєкції каналу
    (жереб × r), D — напрямки (r × вихід)."""
    nets, r = len(E), P[0].shape[1]
    A = np.zeros((nets, r, r))
    b = np.zeros((nets, r))
    for i in range(nets):
        G = P[i][:, :, None] * D[i][None, :, :]        # (жереб, r, вихід)
        Gf = G.reshape(G.shape[0], r, -1)
        A[i] = np.einsum('nkd,njd->kj', Gf, Gf)
        b[i] = np.einsum('nkd,nd->k', Gf, E[i])
    res = 0.0
    for i in range(nets):
        Ai, bi = A.sum(0) - A[i], b.sum(0) - b[i]
        lam = np.linalg.solve(Ai + 1e-10 * np.trace(Ai) / r * np.eye(r), bi)
        corr = np.einsum('k,nk,kd->nd', lam, P[i], D[i])
        res += float(((E[i] - corr) ** 2).sum())
    return res


def loo_lin(E, C):
    nets = len(E)
    num = np.array([float((E[i] * C[i]).sum()) for i in range(nets)])
    den = np.array([float((C[i] * C[i]).sum()) for i in range(nets)])
    res = 0.0
    for i in range(nets):
        lam = (num.sum() - num[i]) / max(den.sum() - den[i], 1e-300)
        res += float(((E[i] - lam * C[i]) ** 2).sum())
    return res


def main(nets=24, dev=DEV, ndraw=NDRAW):
    print('📣 ПЕРЕДБАЧЕНО: насичення на r=4..8 (PR-ранг шуму 5.9), '
          'стеля 1.10-1.15× над поглибленою базою')
    print('🟢 ЖИВЕ при ≥1.15× над LIN\n', flush=True)
    E, CH, DIR, seal = [], [], [], []
    rmax = max(RS)
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            d = len(wc)
            nd, _ = budget(wc, d - 1)
            mt, _ = truth_means(wc, dev, 310_000 + 877 * n)
            m31 = mt[d - 1]
            s_cl, m_cl, P_cl, q_cl = closure_parts(wc, d - 1, dev, extras=True)
            cl8 = (s_cl[:KSIG], m_cl[:KSIG], P_cl[:KSIG], q_cl[:KSIG])
            mfull, Cfull, qfull = closure_full(wc, dev)
            q31 = q_cl[d - 2].double()
            seal.append(float((qfull - q31).abs().max()
                              / q31.abs().max()))
            # напрямки: m_замик + головні власні вектори C_замик, ортогон.
            e0 = mfull / mfull.norm()
            ev, V = torch.linalg.eigh(Cfull)
            B = [e0]
            for j in range(V.shape[1] - 1, -1, -1):
                v = V[:, j].clone()
                for u in B:
                    v = v - (v @ u) * u
                nv = v.norm()
                if float(nv) > 1e-6:
                    B.append(v / nv)
                if len(B) >= rmax:
                    break
            Dm = torch.stack(B[:rmax], 0).cpu().numpy()
            e, c = [], []
            for k in range(ndraw):
                sd = 83_000_000 + 977 * n + k
                pr, ch = run(wc, nd, cl8, dev, sd, q31_cl=q31)
                e.append((pr - m31).cpu().numpy())
                c.append(ch[2].cpu().numpy())
            E.append(np.array(e))
            CH.append(np.array(c))
            DIR.append(Dm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж × {ndraw} жеребів')
    print(f'✅ПЛОМБА ВЛАСНОЇ РЕКУРСІЇ (q мій проти базового): '
          f'{max(seal):.2e} {"✅" if max(seal) < 1e-5 else "🛑"}')
    base = sum(float((E[i] ** 2).sum()) for i in range(nets))
    lin = loo_lin(E, CH)
    print(f'\n{"спосіб":>34} {"×":>8}')
    print(f'{"без поправки":>34} {1.0:>8.4f}')
    print(f'{"LIN λ·C повним вектором (1 пар.)":>34} {base/lin:>8.4f}')
    rng = np.random.default_rng(7)
    for r in RS:
        rr = max(r, 1)
        P = [np.einsum('nd,kd->nk', CH[i], DIR[i][:rr]) for i in range(nets)]
        D = [DIR[i][:rr] for i in range(nets)]
        v = loo_dirs(E, P, D)
        Pn = [np.roll(P[i], 1, axis=0) for i in range(nets)]
        n1 = loo_dirs(E, Pn, D)
        Pr = [rng.standard_normal(P[i].shape) * P[i].std(0, keepdims=True)
              for i in range(nets)]
        n2 = loo_dirs(E, Pr, D)
        tag = ' ← лише напрямок m' if r == 0 else ''
        print(f'{f"НАПР r={rr} ({rr} пар.)":>34} {base/v:>8.4f}   '
              f'⛔НЕГ-жереб {base/n1:.4f}  ⛔НЕГ-випадк {base/n2:.4f}{tag}')
    print('\n📏 ВИГРАШ ЧИТАТИ НАД ПІДЛОГОЮ КОНТРОЛІВ, а не над 1.0000')
    print('📊 БАЗОВИЙ БАЛ 1.954e-07 · ціна поглиблення замикання 0.9519× '
          'уже врахована в `budget()`')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
