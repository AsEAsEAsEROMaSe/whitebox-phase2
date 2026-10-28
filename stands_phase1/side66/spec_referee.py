"""СУД МІЖ ДВОМА ПРОЦЕДУРАМИ СПЕКТРА — на синтетиці з ВІДОМОЮ відповіддю.

Стара (`degree_spectrum.py`): одна спільна NNLS по ρ>0, KMAX=24.
Нова (`degree_spectrum2.py`): розділення ПАРНОСТІ, дві задачі вдвічі меншої
вимірності.

Вони дали РІЗНІ відповіді на тих самих мережах (стара: a₄=0.039 < a₅=0.125,
немонотонно; нова: a₄=0.058 > a₅=0.032, монотонно). Розсудити може лише
синтетика з відомою істиною. Беру п'ять спектрів, серед них обидва кандидати
на правду, і дивлюсь, яка процедура відтворює ЯКИЙ.

Одновимірні функції ⟹ шум мізерний, обчислення секунди.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from degree_spectrum2 import he, spectrum_parity, spectrum_old, RHOS   # noqa: E402

DEV = 'cuda:0'
NC = 1 << 21

CASES = {
    'A монотонний (нова каже так)': {1: .338, 2: .294, 3: .108, 4: .058,
                                     5: .032, 6: .010, 8: .150},
    'B немонотонний (стара)': {1: .341, 2: .277, 3: .100, 4: .039,
                               5: .125, 6: .060, 8: .045},
    'C геометричний 0.6': {1: .400, 2: .240, 3: .144, 4: .086,
                           5: .052, 6: .031, 8: .047},
    'D пік на 5': {1: .200, 2: .150, 3: .100, 4: .050, 5: .400, 6: .050,
                   8: .050},
    'E рівний хвіст': {1: .250, 2: .150, 3: .100, 4: .100, 5: .100,
                       6: .100, 8: .200},
}


def run(spec, dev, seed):
    cf = {k: math.sqrt(v) for k, v in spec.items()}
    g = torch.Generator(device=dev).manual_seed(seed)
    u = torch.randn(NC, device=dev, dtype=torch.float64, generator=g)
    up = torch.randn(NC, device=dev, dtype=torch.float64, generator=g)

    def fn(t):
        s = torch.zeros_like(t)
        for k, c in cf.items():
            s = s + c * he(t, k)
        return s
    f0 = fn(u)
    m0, v0 = float(f0.mean()), float(f0.var())
    cs = []
    for r in RHOS:
        s = math.sqrt(1.0 - r * r)
        gg = fn(r * u + s * up)
        cs.append(float(((f0 - m0) * (gg - gg.mean())).mean()))
    ap, _ = spectrum_parity(cs, v0)
    ao = spectrum_old(cs, v0)
    return ap / v0, ao / v0


def fmt(a):
    return ' '.join(f'{a[k]:.3f}' for k in range(6))


def main():
    print(f'ns=2**21, одновимірні функції з ТОЧНО відомим спектром\n')
    err_p, err_o = [], []
    for nm, spec in CASES.items():
        t = np.zeros(14)
        for k, v in spec.items():
            t[k - 1] = v
        ap, ao = run(spec, DEV, 4242 + len(nm))
        c_t = np.cumsum(t)
        c_p, c_o = np.cumsum(ap), np.cumsum(ao)
        print(f'{nm}')
        print(f'   істина  a₁..a₆ = {fmt(t)}   ≤3={c_t[2]:.3f} ≤4={c_t[3]:.3f} '
              f'≤5={c_t[4]:.3f}')
        print(f'   ПАРНІСТЬ a₁..a₆ = {fmt(ap)}   ≤3={c_p[2]:.3f} '
              f'≤4={c_p[3]:.3f} ≤5={c_p[4]:.3f}')
        print(f'   СТАРА    a₁..a₆ = {fmt(ao)}   ≤3={c_o[2]:.3f} '
              f'≤4={c_o[3]:.3f} ≤5={c_o[4]:.3f}')
        ep = abs(c_p[:6] - c_t[:6]).max()
        eo = abs(c_o[:6] - c_t[:6]).max()
        err_p.append(ep)
        err_o.append(eo)
        print(f'   макс. похибка НАКОПИЧЕНОЇ суми: парність {ep:.3f}  '
              f'стара {eo:.3f}   {"→ ПАРНІСТЬ" if ep < eo else "→ СТАРА"}\n')
    print(f'ПІДСУМОК: середня макс.похибка — парність {np.mean(err_p):.3f}, '
          f'стара {np.mean(err_o):.3f}')
    print('🔑 хто менше — тому й вірити на реальних мережах')


if __name__ == '__main__':
    main()
