"""ЧИ Б'Є ЗАМИКАННЯ ВИБІРКОВИЙ ШУМ НА РАННІХ ШАРАХ. (мета: 1e-8)

ЗВІДКИ. `price_rule.py` виміряв: нев'язка дисперсії доставляється на шар 1
ТОЧНО (лін.R²=1.0000, бо `p₁` лінійне по `x₀`, а `C(x₀)` = арккосинусне ядро
в замкненому вигляді). Ламається інше — на шарах 1..4 нев'язка НАРОДЖУЄТЬСЯ
(перенос пояснює лише 22% на шарі 1), і до шару 6 від стартової лишається
чверть. Тому доставка δs дає 1.06× замість оракульних 3.0×.

ІДЕЯ. Істинні моменти на ранніх шарах не обов'язково НЕСТИ — їх можна
ПОРАХУВАТИ замиканням. Замиканню тут НЕ треба бути точним на 32 шари (там
воно дає 6.29e-05 = 315× гірше за лідерів). Йому треба лише побити масштаб
вибіркового шуму `1/√6200 ≈ 1.3%` на перших кількох шарах. Цього не міряли.

ЩО РАХУЄМО. Рекурсія в замкненому вигляді, без жодної вибірки:
    m₀ = σ_j/√(2π),  C₀ = арккосинусне ядро          ← ТОЧНО
    m_p = m·W,       C_p = Wᵀ C W                    ← ТОЧНО (лінійність)
    m,C наступного шару = моменти relu(гаусa) з (m_p, C_p)   ← ЄДИНЕ припущення
Другий момент пари береться одновимірною квадратурою (внутрішній інтеграл
узято аналітично):
    E[relu(X)relu(Y)] = E_u[ relu(m₁+s₁u)·( μ(u)Φ(μ(u)/σ) + σφ(μ(u)/σ) ) ]

ЩО ДРУКУЄМО пошарово, у ЄДИНИХ одиницях — часток істинного значення:
    похибка ЗАМИКАННЯ   |s_зам − s_іст| / s_іст
    шум ВИБІРКИ         |ŝ − s_іст| / s_іст
    ВІДНОШЕННЯ          <1 ⟹ замикання точніше за вибірку ⟹ гілка ЖИВА
і головне: ρ(δs_замикання, δs_істинна) — чи справді замикання дає ту саму
поправку, що й оракул.

КОНТРОЛІ
  * істина — власна, 4М свіжих семплів (не мітка банку);
  * шар 0 має вийти ТОЧНО (арккосинус) ⟹ похибка замикання ~1e-7: позитивний
    контроль формул і квадратури;
  * друкуємо і абсолютні числа, і відношення — щоб не переплутати рівні.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
from sub66_torch import (INV_SQRT_2PI, MIN_PAIRS, PILOT_PAIRS,  # noqa: E402
                         TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                              # noqa: E402
from born_source import truth_moments                         # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:2'
TRUTH_N = 4_194_304
NQ = 48
SQRT2 = math.sqrt(2.0)


def _Phi(t):
    return 0.5 * (1.0 + torch.erf(t / SQRT2))


def _phi(t):
    return INV_SQRT_2PI * torch.exp(-0.5 * t * t)


def relu_m(m, s):
    """E[relu(X)] для X ~ N(m, s²)."""
    t = m / s
    return m * _Phi(t) + s * _phi(t)


def relu_cov0(w0):
    """ТОЧНА центрована коваріація relu(z W₀) при z ~ N(0,I) — арккосинус."""
    S = (w0.T @ w0).double()
    s = S.diagonal().clamp_min(1e-30).sqrt()
    rho = (S / torch.outer(s, s)).clamp(-1.0, 1.0)
    e2 = torch.outer(s, s) / (2.0 * math.pi) * (
        (1.0 - rho * rho).clamp_min(0.0).sqrt() + rho * (math.pi / 2 + torch.asin(rho)))
    return e2 - torch.outer(s, s) / (2.0 * math.pi)


def relu_cov_gauss(m, C, nodes, wts, chunk=64):
    """Коваріація relu(p) для ГАУСОВОГО p ~ N(m, C). Квадратура по одному виміру."""
    s = C.diagonal().clamp_min(1e-30).sqrt()
    n = m.shape[0]
    out = torch.empty(n, n, dtype=torch.float64, device=m.device)
    rho_all = (C / torch.outer(s, s)).clamp(-0.999999, 0.999999)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        rho = rho_all[a:b]                                   # (k, n)
        s1 = s[a:b].unsqueeze(1)                             # (k, 1)
        m1 = m[a:b].unsqueeze(1)
        u = nodes.view(1, 1, -1)
        x1 = torch.relu(m1.unsqueeze(2) + s1.unsqueeze(2) * u)          # (k,1,Q)
        mu = m.view(1, n, 1) + (s.view(1, n, 1) * rho.unsqueeze(2)) * u  # (k,n,Q)
        sg = (s.view(1, n, 1) * (1.0 - rho * rho).sqrt().unsqueeze(2)).clamp_min(1e-30)
        t = mu / sg
        y = mu * _Phi(t) + sg * _phi(t)
        out[a:b] = (x1 * y * wts.view(1, 1, -1)).sum(2)
    mo = relu_m(m, s)
    return out - torch.outer(mo, mo), mo


def sample_moments(wc, seed, dev):
    """Вибіркові (m, s) доактиваційних `p` кожного шару — рівно як у sub66."""
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    per_cut = sum(2.0 * w.shape[0] * w.shape[1] for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * L_ * W_ * W_)
    npair = max(MIN_PAIRS, int((TARGET_UTIL * B_BUDGET - used) / (2.0 * per_cut)))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32), device=dev)
    x = torch.cat([z, -z], 0)
    n2 = float(x.shape[0])
    cov = (x.T @ x) / n2
    ev, evec = torch.linalg.eigh(cov.double())
    wh = (evec * (1.0 / ev.clamp_min(1e-8).sqrt()) @ evec.T).float()
    x = torch.relu(x @ (wh @ wc[0]))
    ms, ss = [], []
    for i in range(1, L_):
        p = (x @ wc[i]).double()
        m = p.sum(0) / n2
        ms.append(m)
        ss.append(((p * p).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt())
        x = torch.relu(p.float())
    return ms, ss


def main(nets=4, upto=9, dev=DEV):
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    tab = {}
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            _, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            ms, ss = sample_moments(wc, 12345, dev)

            w0 = wc[0].double()
            sg0 = w0.pow(2).sum(0).clamp_min(1e-30).sqrt()
            m_est = sg0 * INV_SQRT_2PI
            C_est = relu_cov0(wc[0])
            for i in range(1, min(upto, L_)):
                Wd = wc[i].double()
                m_p = m_est @ Wd
                C_p = Wd.T @ (C_est @ Wd)
                s_p = C_p.diagonal().clamp_min(1e-30).sqrt()

                st = sg_t[i - 1].double()
                sh = ss[i - 1]
                e_clo = float(((s_p - st).abs() / st).mean())
                e_smp = float(((sh - st).abs() / st).mean())
                d_clo, d_tru = sh - s_p, sh - st
                rho = float((d_clo * d_tru).sum()
                            / (d_clo.norm() * d_tru.norm()).clamp_min(1e-300))
                mt = mu_t[i - 1].double()
                e_clo_m = float(((m_p - mt).abs() / st).mean())
                tab.setdefault(i, []).append((e_clo, e_smp, e_clo / e_smp, rho,
                                              e_clo_m))
                C_est, m_est = relu_cov_gauss(m_p, C_p, nodes, wts)
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    print(f'\n{len(fs)} мереж, істина {TRUTH_N}, квадратура {NQ} вузлів')
    print(f'{"шар":>4s} {"замикання":>11s} {"вибірка":>10s} {"зам/виб":>9s}'
          f' {"ρ(δs)":>8s} {"зам.по m":>10s}')
    print('-' * 58)
    for i in sorted(tab):
        a = np.mean(np.array(tab[i]), 0)
        print(f'{i:4d} {a[0]:11.3e} {a[1]:10.3e} {a[2]:9.3f} {a[3]:+8.4f}'
              f' {a[4]:10.3e}')
    print('\nЧИТАННЯ: «зам/виб» < 1 ⟹ замикання точніше за вибірку на цьому шарі,')
    print('тобто δs можна ПОРАХУВАТИ, а не нести. ρ(δs) — чи та сама це поправка.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
         upto=int(sys.argv[2]) if len(sys.argv) > 2 else 9)
