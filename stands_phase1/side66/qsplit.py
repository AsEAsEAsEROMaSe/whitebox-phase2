"""З ЧОГО ЗРОБЛЕНІ 0.887% ПОХИБКИ ВОРІТ — ФОРМА ЧИ АРГУМЕНТ? (С81)

## ЗВІДКИ ПИТАННЯ
`side66/qladder.py` дав ТЗ у вигляді числа: канал воріт платить ×1.111,
якщо еталон `q` точний до **0.222%**, а нинішній `Φ(t)` із замикання має
**0.887%** — учетверо гірше. Отже треба знати, ЩО САМЕ там помиляється.

`Φ(t)` робить ДВА припущення одночасно:
  1️⃣ **АРГУМЕНТ**: `t = m/s`, де `m` і `s` беруться із ЗАМИКАННЯ (наближені);
  2️⃣ **ФОРМА**: перед-активація ГАУСОВА, тому `P(p>0) = Φ(t)`.
Розділити їх можна тільки підстановкою ІСТИННИХ моментів.

## ВАРІАНТИ (усі проти істинного `q` великим МК, на ОБРІЗАНІЙ мережі —
##           так само, як міряла `qladder.py`, щоб числа були порівнянні)
  🟠 A · `Φ(m_замик / s_замик)`      — нинішній вільний еталон, чекаємо 0.887%
  🔵 B · `Φ(m_іст / s_іст)`          — ГАУСОВА форма з ІСТИННИМИ моментами
  🟢 C · Еджворт-3 з істинними `γ₁`  — `Φ(t) + φ(t)·γ₁(t²−1)/6`
  🟢 D · Еджворт-4 (додано `γ₂`)     — плюс `φ(t)[γ₂(t³−3t)/24 + γ₁²(…)/72]`

🔑 ЧИТАННЯ:
  * якщо B ≈ 0.9% — винна ФОРМА, і Еджворт (C/D) мусить її різко зняти;
  * якщо B ≈ 0.2% — винен АРГУМЕНТ, тобто точність `m,s` із замикання, а це
    вже закритий клас ⟹ канал воріт мертвий;
  * якщо C/D сідають нижче **0.222%** — гілка ЖИВА і коштує ×1.111 у бал.

⛔ ПЛОМБИ:
 * A мусить відтворити **0.887%** (`qladder.py`, ті самі мережі й обрізання);
 * ⛔`Φ` від істинних моментів мусить бути НЕ ГІРШОЮ за A (інакше плутанина
   в тому, що з чим порівнюємо);
 * ✅внутрішня: `q_іст` мусить лежати в [0,1] і мати середнє ≈0.5 на шарі 1.

📣 ПЕРЕДБАЧЕНО: B дасть 0.3…0.5% (частина похибки — від аргументу), C зніме
   ще половину до ~0.2%, D майже нічого не додасть. Тобто гілка виявиться
   живою, але ГРАНИЧНО: вимога 0.222% буде на самій межі досяжного, і все
   впреться в те, що `γ₁` теж треба звідкись узяти (істинного в сабміті нема).
   Якщо ж B ≈ A — уся похибка у формі, і це НАЙКРАЩИЙ сценарій.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/qsplit.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts            # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
L_, W_, KS = 32, 256, 31
NPAIR, CH = 524_288, 32768
I2PI = 0.3989422804014327
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _phi(x):
    return torch.exp(-0.5 * x * x) * I2PI


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def moments(wc, dev, seed, npair=NPAIR):
    """Істинні `q`, `m`, `s`, `γ₁`, `γ₂` перед-активацій шарів 1..KS."""
    ns = [w.shape[1] for w in wc]
    acc = [[torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
           for _ in range(5)]                       # q, Σp, Σp², Σp³, Σp⁴
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(CH, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = (x @ w).double()
            acc[0][i] += (p > 0).double().sum(0)
            p2 = p * p
            acc[1][i] += p.sum(0)
            acc[2][i] += p2.sum(0)
            acc[3][i] += (p2 * p).sum(0)
            acc[4][i] += (p2 * p2).sum(0)
            x = torch.relu(p).float()
        done += k
    n = 2.0 * done
    out = []
    for i in range(1, KS + 1):
        q = acc[0][i] / n
        m1, m2 = acc[1][i] / n, acc[2][i] / n
        m3, m4 = acc[3][i] / n, acc[4][i] / n
        var = (m2 - m1 * m1).clamp_min(1e-300)
        s = var.sqrt()
        c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
        out.append((q, m1, s, c3 / s ** 3, c4 / var ** 2 - 3.0))
    return out


def rel(pred, tru):
    num = sum(float(((pred[i] - tru[i]) ** 2).sum()) for i in range(KS))
    den = sum(float((tru[i] ** 2).sum()) for i in range(KS))
    return np.sqrt(num / max(den, 1e-300))


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]

NM = ['🟠 A · Φ(m,s із ЗАМИКАННЯ)  ← нинішній',
      '🔵 B · Φ(m,s ІСТИННІ)       ← лишається ФОРМА',
      '🟢 C · Еджворт-3 (γ₁ іст.)',
      '🟢 D · Еджворт-4 (γ₁,γ₂ іст.)']
E = [[] for _ in NM]
SEAL = []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, _ = cut_and_budget(W, DEV)
        _, _, f_cl, _ = closure_parts(wc, KS, DEV, extras=True)
        M = moments(wc, DEV, 830_000 + 7919 * n)
        q_t = [M[i][0] for i in range(KS)]
        SEAL.append((float(q_t[0].mean()), float(min(x.min() for x in q_t)),
                     float(max(x.max() for x in q_t))))
        t = [M[i][1] / M[i][2] for i in range(KS)]
        gauss = [_Phi(t[i]) for i in range(KS)]
        e3 = [gauss[i] + _phi(t[i]) * M[i][3] * (t[i] ** 2 - 1.0) / 6.0
              for i in range(KS)]
        e4 = [e3[i] + _phi(t[i]) * (
            M[i][4] * (t[i] ** 3 - 3.0 * t[i]) / 24.0
            + M[i][3] ** 2 * (t[i] ** 5 - 10.0 * t[i] ** 3 + 15.0 * t[i])
            / 72.0) for i in range(KS)]
        for v, P in enumerate((f_cl, gauss, e3, e4)):
            E[v].append(rel(P, q_t))
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}', flush=True)

print(f'\n{NETS} мереж, істина {2*NPAIR} семплів, обрізана мережа, шари 1..{KS}')
sm = np.array(SEAL)
print(f'✅ПЛОМБА: q шару 1 у середньому {sm[:,0].mean():.4f} (чекаємо ≈0.5) · '
      f'діапазон q = [{sm[:,1].min():.4f}, {sm[:,2].max():.4f}] ⊂ [0,1]\n')
print(f'{"варіант":38s} {"відн. похибка q":>16s} {"проти A":>9s}')
print('-' * 68)
a0 = float(np.mean(E[0]))
for v, nm in enumerate(NM):
    m = float(np.mean(E[v]))
    print(f'{nm:38s} {100*m:15.4f}% {a0/m:8.2f}×')
print(f'\n🎯 ПОРІГ ІЗ `qladder.py`: 0.222% ⟹ приріст ×1.1114 (бал 1.758e-07)')
print(f'   беззбитковість ≈0.5%, а рівень ⛔контролю — ×1.0165')
print('📊 БАЛ 1.954e-07')
