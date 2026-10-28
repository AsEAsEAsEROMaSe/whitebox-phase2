"""ЧИ ДІСТАЄМО ШАР 30: ПОВНА `C` ПРОТИ ДІАГОНАЛЬНОЇ (С80, запит юзера).

С72 (`mdepth.py`) звузила ТЗ до ОДНОГО вектора: `m₃₀` (256 чисел). Її драбина —
готове ТЗ, бо обидва канали ЛІНІЙНІ за переданим значенням:
    `m_k = m_іст + k·(m_замик − m_іст)`   ⟹   k=1.00 → 1.007×  (нинішній стан)
                                              k=0.50 → 1.176×
                                              k=0.25 → **1.694×** (1.21e-07)
                                              k=0.00 → 13.435×  (оракул)
🎯 ПИТАННЯ ЦЬОГО СТЕНДА: який `k` дає ПОВНА коваріаційна рекурсія?
С56 міряла окремо: середнє на глибині 3.47% (діагональ) проти 0.998% (повна `C`).
Якщо це відношення тримається на `‖m₃₀‖`, то `k ≈ 0.29` ⟹ близько 1.6×.
🚨 АЛЕ ці числа НІХТО НЕ ЗВОДИВ: 3.47%/0.998% міряні в С56 як відносні похибки
   ІНШОЇ величини й на ІНШОМУ стенді, а драбина — С72. Зводжу ОДНИМ приладом.

⚠️ ЗАСТЕРЕЖЕННЯ, ЯКЕ ТРЕБА ТРИМАТИ: драбина масштабує ОДИН І ТОЙ САМИЙ вектор
   похибки. Повна `C` дає похибку ІНШОГО НАПРЯМКУ, а не вкорочену. Тому
   `k = ‖b_повна‖/‖b_діаг‖` — це ОЦІНКА, і разом із нею міряю
   `cos(b_повна, b_діаг)`: якщо косинус високий, драбина застосовна; якщо
   низький — число `k` НЕ МОЖНА підставляти в неї.

## РЕКУРСІЯ (обидві версії, той самий код, різниця в ОДНОМУ рядку)
Стан — `(m, S)`, де `S = E[h hᵀ]` (другий момент, не центрований).
    доактивація: `m_p = m @ W`,  `S_p = WᵀSW`,  `σ²_p = diag(S_p) − m_p²`
    ⛔ДІАГОНАЛЬНА версія бере `σ²_p,j = Σ_i W_ij²·Var(h_i)` (позадіагональ `S` НЕ бачить)
    `t = m_p/σ_p`
    `E[relu]  = m_p·Φ(t) + σ_p·φ(t)`
    `E[relu²] = (m_p²+σ_p²)·Φ(t) + m_p·σ_p·φ(t)`
    позадіагональ: `Cov(relu_i,relu_j) ≈ Φ(t_i)Φ(t_j)·Cov(p_i,p_j)`   (член `c₁`
    ряду Мелера; С56 зміряла, що `k=2,3,4` дають 1.006%→0.998%, тобто НІЧОГО)

⛔ ПЛОМБИ:
 * шар 0: вхід ТОЧНО гаусів і `Cov=I` ⟹ ОБИДВІ версії мусять дати `m₀` з
   похибкою на рівні шуму МК (для `σ²_p,j = Σ_i W_ij²` діагональ і повна
   збігаються тотожно);
 * дві НЕЗАЛЕЖНІ істини МК ⟹ їхня розбіжність = шум приладу, і він мусить бути
   МАЛИМ проти вимірюваних 1–3.5%;
 * ⛔НЕГ.: `m₃₀` з ЧУЖОЇ мережі ⟹ похибка мусить бути ~100%.

📣 ПЕРЕДБАЧЕНО ДО ПРОГОНУ: діагональ ~3.5%, повна ~1.0%, отже `k≈0.29`. Але
   ставлю на те, що `cos(b_повна, b_діаг)` вийде ВИСОКИЙ (>0.9), бо обидві
   помиляються в один бік — недооцінюють `σ` через ігнорування додатних
   кореляцій. Якщо так, драбина застосовна і 1.6× реальні; тоді все впирається
   в ЦІНУ (0.77% бюджету флопами + час).

🚀 fp32 у матмулах МК, fp64 у рекурсії та накопиченні.
Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/deep30.py [cuda:N] [NETS] [NMC]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NMC = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000
NMC2 = NMC // 4
LTGT = 30                      # цільовий шар: `m₃₀` (вхід останнього матмулу)
W_, D = 256, 32
F64 = torch.float64
SQ2PI = float(np.sqrt(2.0 * np.pi))

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1].split('🚀')[0], flush=True)


def _phi(x):
    return torch.exp(-0.5 * x * x) / SQ2PI


def _Phi(x):
    return 0.5 * torch.erfc(-x / np.sqrt(2.0))


def _mc(W32, gen, n, layer):
    """Середнє активацій шару `layer` по `n` семплах (fp32 матмули, fp64 сума)."""
    acc = torch.zeros(W_, dtype=F64, device=DEV)
    done = 0
    while done < n:
        b = min(16384, n - done)
        x = torch.randn(b, W_, device=DEV, generator=gen)
        for L in range(layer + 1):
            x = torch.relu(x @ W32[L])
        acc += x.double().sum(0)
        done += b
    return acc / n


def _closure(W, layer, full):
    """Рекурсія (m, S) до `layer`. full=False ⟹ позадіагональ `S` ігнорується."""
    m = torch.zeros(W_, dtype=F64, device=DEV)
    S = torch.eye(W_, dtype=F64, device=DEV)          # вхід: E[zzᵀ] = I
    for L in range(layer + 1):
        Wl = W[L]
        m_p = m @ Wl
        if full:
            S_p = Wl.transpose(0, 1) @ S @ Wl
            var_p = torch.diagonal(S_p) - m_p * m_p
        else:
            # ⛔ЄДИНА ВІДМІННІСТЬ: бачимо лише діагональ `S`
            var_h = torch.diagonal(S) - m * m
            var_p = (Wl * Wl).transpose(0, 1) @ var_h
            S_p = None
        sd = var_p.clamp_min(1e-300).sqrt()
        t = m_p / sd
        Ph, ph = _Phi(t), _phi(t)
        m_new = m_p * Ph + sd * ph
        e2 = (m_p * m_p + var_p) * Ph + m_p * sd * ph
        if full:
            cov_p = S_p - torch.outer(m_p, m_p)
            g = Ph
            S = torch.outer(g, g) * cov_p + torch.outer(m_new, m_new)
            S.diagonal().copy_(e2)
        else:
            S = torch.diag(e2)
        m = m_new
    return m


rows = []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), dtype=F64, device=DEV)
         for k in range(D)]
    W32 = [w.float() for w in W]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(4242 + 97 * net)
        tru = _mc(W32, g, NMC, LTGT)
        tru2 = _mc(W32, g, NMC2, LTGT)
        m_d = _closure(W, LTGT, False)
        m_f = _closure(W, LTGT, True)
        # ⛔ПЛОМБА ШАРУ 0
        t0 = _mc(W32, g, NMC2, 0)
        d0 = _closure(W, 0, False)
        f0 = _closure(W, 0, True)
        s0d = float((d0 - t0).norm() / t0.norm())
        s0f = float((f0 - t0).norm() / t0.norm())
        # ⛔НЕГ.: істина з ЧУЖОЇ мережі
        Wf = [torch.tensor(np.asarray(WZ[(net + 5) % 100][k], np.float32),
                           device=DEV) for k in range(D)]
        tn = _mc(Wf, g, NMC2, LTGT)
        del Wf
        nrm = tru.norm()
        e_d, e_f = (m_d - tru), (m_f - tru)
        rows.append((float(e_d.norm() / nrm), float(e_f.norm() / nrm),
                     float((tru2 - tru).norm() / nrm),
                     float(e_f.norm() / e_d.norm().clamp_min(1e-300)),
                     float((e_d * e_f).sum() / e_d.norm() / e_f.norm()),
                     s0d, s0f, float((tn - tru).norm() / nrm)))
    print(f'  … мережа {net+1}/{NETS}: діаг {100*rows[-1][0]:.3f}%  '
          f'повна {100*rows[-1][1]:.3f}%  k={rows[-1][3]:.3f}  '
          f'cos={rows[-1][4]:+.3f}  шум {100*rows[-1][2]:.4f}%', flush=True)
    del W, W32
    torch.cuda.empty_cache()

a = np.array(rows)
print(f'\n{NETS} мереж, МК {NMC:,} (крос {NMC2:,}), ціль `m_{LTGT}`\n')
print('⛔ ПЛОМБИ')
print(f'  шар 0, діагональна = {100*np.median(a[:,5]):.4f}%   (треба ≈ шум МК)')
print(f'  шар 0, повна       = {100*np.median(a[:,6]):.4f}%   (треба те саме)')
print(f'  шум МК (дві істини)= {100*np.median(a[:,2]):.4f}%')
print(f'  ⛔НЕГ. чужа мережа  = {100*np.median(a[:,7]):.1f}%   (треба ~100%)')
print(f'\n{"величина":40s} {"медіана":>11s} {"мін":>11s} {"макс":>11s}')
print('-' * 78)
for k, nm in ((0, 'відн. похибка `m₃₀`, ДІАГОНАЛЬНА'),
              (1, 'відн. похибка `m₃₀`, ПОВНА `C`'),
              (3, '🎯 k = ‖b_повна‖/‖b_діаг‖'),
              (4, '⚠️ cos(b_повна, b_діаг)')):
    c = a[:, k]
    print(f'{nm:40s} {np.median(c):11.5f} {c.min():11.5f} {c.max():11.5f}')

k = float(np.median(a[:, 3])); cs = float(np.median(a[:, 4]))
LAD = [(1.00, 1.007), (0.50, 1.176), (0.25, 1.694), (0.10, 4.109), (0.00, 13.435)]
lo = max([x for x in LAD if x[0] >= k], key=lambda x: -x[0])
hi = min([x for x in LAD if x[0] <= k], key=lambda x: x[0])
w = (lo[0] - k) / max(lo[0] - hi[0], 1e-9)
gain = lo[1] + w * (hi[1] - lo[1])
print(f'\n🎯 k = {k:.3f} ⟹ по драбині С72 приблизно **{gain:.3f}×**')
print(f'⚠️ cos(b_повна,b_діаг) = {cs:+.3f} — '
      f'{"драбина ЗАСТОСОВНА" if cs > 0.9 else "🚨 драбина НЕ застосовна, напрямок ІНШИЙ"}')
print(f'\n💰 ЦІНА повної `C` до шару {LTGT}: 4·256³·{LTGT+1} = '
      f'{4*256**3*(LTGT+1):.3g} флоп = {100*4*256**3*(LTGT+1)/2.72e11:.2f}% бюджету B')
print(f'   ⟹ множник 0.0997 → {0.0997+4*256**3*(LTGT+1)/2.72e11:.4f} '
      f'(×{(0.0997+4*256**3*(LTGT+1)/2.72e11)/0.0997:.3f}), БЕЗ урахування ЧАСУ')
print(f'📊 БАЛ 1.954e-07 ⟹ якщо драбина застосовна: {1.954e-07/gain*((0.0997+4*256**3*(LTGT+1)/2.72e11)/0.0997):.4e}')
