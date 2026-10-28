"""ЧИ ЗМЕНШУЄ ТРЕТІЙ ЧЛЕН ПРАЙСА ЛОКАЛЬНУ ПОХИБКУ КРОКУ ЗАМИКАННЯ? (С73)

════════ ЗВІДКИ ════════
`arena62/deepsrc.py` з'ясував походження `B²` (похибки замикання, яка вбиває
глибокий якір): це **НАКОПИЧЕННЯ ~30 майже однакових локальних внесків**
(0.218% на кроці 0→1, 0.131% на 24→25), а не одна погана ділянка ⟹
**зменшити локальну похибку кроку у X разів = опустити ВСЮ криву у X разів.**
🎯 мішень: локальна 0.147% → 0.031% (у 4.8×) ⟹ стеля шару 30 стає 2×.

════════ ЩО САМЕ ДОДАЄМО ════════
Наше замикання — це розклад `Cov(relu(p_i), relu(p_j))` за степенями
КОВАРІАЦІЇ `cp`, за теоремою Прайса `∂ᵏCov/∂cpᵏ = E[f⁽ᵏ⁾(p_i)]·E[f⁽ᵏ⁾(p_j)]`.
Для relu похідні — Гевісайд, δ, δ′:
    k=1:  E[relu′] = Φ(t)            ⟹  cp·Φ_iΦ_j                    ✔ є в коді
    k=2:  E[relu″] = φ(t)/s          ⟹  cp²/2·(φ_i/s_i)(φ_j/s_j)     ✔ є в коді
    k=3:  E[relu‴] = −t·φ(t)/s²      ⟹  cp³/6·(t_iφ_i/s_i²)(t_jφ_j/s_j²)  ← НЕМА
(перевірка: `ρ_p(x)=φ((x−m)/s)/s`, `E[δ′(p)] = −ρ_p′(0) = −tφ(t)/s²`.)

📐 **ВІДНОШЕННЯ ЧЛЕНІВ РАХУЄТЬСЯ НАПЕРЕД**:  `k3/k2 = (ρ·t_i·t_j)/3`,
де `ρ = cp/(s_i s_j)`. ⟹ 🚨 **НА ШАРІ 1 ЦЕ ТОТОЖНО НУЛЬ** (старт `m=0` ⟹ t=0),
а саме крок 0→1 має НАЙБІЛЬШУ локальну похибку. Тому стенд друкує ПРОГНОЗ
(з реальних ρ і t) ПЕРЕД виміром — і вимір мусить його підтвердити або вбити.

════════ ГЕЙТ ════════
🟢 живий, якщо локальна похибка `m` після кроку падає ≥2× хоча б на кроках ≥8;
🔴 мертвий, якщо падіння <20% — тоді залишок K=2 не в третьому члені, і
   лагодити треба не ряд (найімовірніше — НЕГАУСОВІСТЬ `p`, якої ряд не бачить).

🚨 ПЛОМБИ
  ✅ крок стартує з ІСТИННИХ `(m,C)` ⟹ похибка кроку не змішана з накопиченою;
  ✅ `t=0` (шар 1): K=3 мусить дати РІВНО той самий результат, що K=2;
  ⛔ КОНТРОЛЬ ЗНАКУ: член з ПЕРЕВЕРНУТИМ знаком мусить зробити ГІРШЕ.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_, _phi, _Phi          # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
TR_PAIRS, CHUNK = 131_072, 16384
STEPS = (0, 1, 2, 4, 8, 12, 16, 20, 24, 28)


def truth_moments(wc, dev, seed):
    """Істинні `m_ℓ`, `C_ℓ` для кожного шару (велика вибірка, антитетика)."""
    d = len(wc)
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
    gram = [torch.zeros(w.shape[1], w.shape[1], dtype=torch.float64,
                        device=dev) for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(d):
            a = torch.relu(a @ wc[i])
            ad = a.double()
            acc[i] += ad.sum(0)
            gram[i] += ad.T @ ad
        done += k
    n = float(2 * TR_PAIRS)
    ms = [t / n for t in acc]
    cs = [gram[i] / n - torch.outer(ms[i], ms[i]) for i in range(d)]
    return ms, cs


def one_step(w, m, C, k3=0.0):
    """ОДИН крок замикання. `k3` — вага третього члена Прайса (0 = як у сабміті).

    🔑 Діагональ вписується ТОЧНОЮ формулою в обох випадках, тож третій член
    впливає лише на НЕДІАГОНАЛЬ `C` — і далі на `s` наступного шару."""
    cp = w.T @ (C @ w)
    s = cp.diagonal().clamp_min(1e-20).sqrt()
    mp = m @ w
    t = mp / s
    ph, Ph = _phi(t), _Phi(t)
    m_new = mp * Ph + s * ph
    C_new = cp * torch.outer(Ph, Ph) \
        + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
    if k3 != 0.0:
        u3 = t * ph / (s * s)                       # t·φ(t)/s²
        C_new = C_new + (k3 / 6.0) * (cp * cp * cp) * torch.outer(u3, u3)
    dg = (mp * mp + s * s) * Ph + mp * s * ph - m_new * m_new
    C_new = C_new + torch.diag(dg - C_new.diagonal())
    return m_new, C_new, t, s


def rel(a, b):
    """RMS-похибка `a` проти `b`, у відсотках від RMS(b)."""
    return 100.0 * float(((a - b) ** 2).mean().sqrt() / (b ** 2).mean().sqrt())


def main(nets=4, dev=DEV):
    torch.manual_seed(77)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    rows = {k: {'m2': [], 'm3': [], 'c2': [], 'c3': [], 'neg': [],
                'pred': [], 'rho': [], 't': []} for k in STEPS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            ms, cs = truth_moments(wc, dev, 900_000 + 7919 * n)
            for k in STEPS:
                if k + 1 >= len(wc):
                    continue
                # ✅ старт із ІСТИННИХ моментів шару k ⟹ похибка ЧИСТО локальна
                if k == 0:
                    m0 = torch.zeros(wc[0].shape[0], dtype=torch.float64,
                                     device=dev)
                    C0 = torch.eye(wc[0].shape[0], dtype=torch.float64,
                                   device=dev)
                    w = wc[0]
                else:
                    m0, C0, w = ms[k - 1], cs[k - 1], wc[k]
                mt, ct = ms[k], cs[k]

                m2, c2, t, s = one_step(w.double(), m0, C0, k3=0.0)
                m3, c3, _, _ = one_step(w.double(), m0, C0, k3=1.0)
                mn, _, _, _ = one_step(w.double(), m0, C0, k3=-1.0)

                # 📐 ПРОГНОЗ: середнє |ρ·t_i·t_j|/3 по недіагоналі
                cp = w.double().T @ (C0 @ w.double())
                sd = cp.diagonal().clamp_min(1e-20).sqrt()
                rho = cp / torch.outer(sd, sd)
                off = ~torch.eye(len(sd), dtype=torch.bool, device=dev)
                pr = (rho * torch.outer(t, t) / 3.0).abs()[off].mean()

                rows[k]['m2'].append(rel(m2, mt))
                rows[k]['m3'].append(rel(m3, mt))
                rows[k]['neg'].append(rel(mn, mt))
                rows[k]['c2'].append(rel(c2, ct))
                rows[k]['c3'].append(rel(c3, ct))
                rows[k]['pred'].append(100.0 * float(pr))
                rows[k]['rho'].append(float(rho[off].abs().mean()))
                rows[k]['t'].append(float(t.abs().mean()))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж, істина {2*TR_PAIRS} семплів, старт із ІСТИННИХ '
          f'моментів кожного кроку\n')
    print(f'{"крок":>6s} {"|ρ|":>6s} {"|t|":>6s} {"ПРОГНОЗ k3/k2":>13s} │ '
          f'{"похибка m: K=2":>14s} {"K=3":>8s} {"падіння":>8s} │ '
          f'{"похибка C: K=2":>14s} {"K=3":>8s} │ {"⛔знак −":>9s}')
    print('-' * 118)
    for k in STEPS:
        r = rows[k]
        if not r['m2']:
            continue
        f = np.median
        m2, m3 = f(r['m2']), f(r['m3'])
        tag = '  ← t=0 ⟹ мусить бути РІВНО як K=2' if k == 0 else ''
        print(f'{k:6d} {f(r["rho"]):6.3f} {f(r["t"]):6.3f} '
              f'{f(r["pred"]):12.3f}% │ {m2:13.4f}% {m3:7.4f}% '
              f'{m2/max(m3,1e-12):7.2f}× │ {f(r["c2"]):13.4f}% '
              f'{f(r["c3"]):7.4f}% │ {f(r["neg"]):8.4f}%{tag}')
    print('-' * 118)
    print('🟢 ГЕЙТ: падіння ≥2× на кроках ≥8 ⟹ важіль живий.')
    print('🔴 падіння <1.2× ⟹ залишок K=2 НЕ в третьому члені; найімовірніша '
          'причина — НЕГАУСОВІСТЬ `p`, якої ряд Прайса не бачить у принципі.')
    print('⛔ колонка «знак −» мусить бути ГІРШОЮ за K=2, інакше стенд не '
          'розсуджує (член не впливає).')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
