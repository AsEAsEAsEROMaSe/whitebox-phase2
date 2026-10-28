"""СПЕКТР СТЕПЕНІВ-2: чи РЕАЛЬНА немонотонність a₄=0.039 / a₅=0.125?

ЗВІДКИ. `degree_spectrum.py` дав накопичені суми ≤3→0.718, ≤5→0.882, ≥6→0.105
і стелі 3.59× / 8.73× / 9.49×. Юзер прочитав таблицю далі за мене: перехід
d=3→4 дає лише +15%, а d=4→5 ПОДВОЮЄ стелю. Тобто вся ставка на «Мелер K=5»
спирається на різницю a₅−a₄ = 0.086 — а власний пойнт визнає похибку ±0.08–0.09
саме на цих коефіцієнтах. Отже досі НЕ БУЛО контролю, здатного це розсудити.

ЩО ДОДАНО (чого не було в №1)
  1. ⚖️ ПАРНІСТЬ. ρ пробігає ± значення. C(ρ)+C(−ρ) = 2Σ_парні, C(ρ)−C(−ρ) =
     2Σ_непарні. Дві окремі задачі вдвічі меншої вимірності — саме там, де
     плуталися a₄ і a₅ (сусіди РІЗНОЇ парності; у №1 вони конкурували напряму).
  2. ✅ КОНТРОЛІ СТЕПЕНІВ 4 і 5 поодинці — чи взагалі прилад їх розрізняє.
  3. ✅ КОНТРОЛЬ-ДВІЙНИК: синтезую функцію з ТОЧНО нашим виміряним спектром
     (a₁..a₅, a₆, a₈) і дивлюсь, чи процедура поверне немонотонність. Це
     головний контроль: якщо на двійнику a₄/a₅ зливаються — число не наше.
  4. 📈 ДВА ОБСЯГИ ВИБІРКИ (65536 і 262144) на ТИХ САМИХ мережах: питання
     юзера «стоїть чи повзе».
  5. 🔬 SOFTPLUS. Гіпотеза юзера: хвіст ≥6 — це ЗЛАМИ ReLU (явище Гіббса в
     ермітових координатах), а не «складна структура». Якщо згладити активацію
     і хвіст обвалиться — діагноз доведено, і тоді 9.49× це стеля ГЛАДКОСТІ,
     а не стеля нашого методу.

Ермітова нормалізація: He_k, E[He_j He_k]=k!δ ⟹ He_k(u)/√(k!) має Var=1,
тож коефіцієнт c_k дає Var_k = c_k² рівно.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:0'
CH = 8192
KMAX = 14
RP = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
RHOS = np.concatenate([-RP[::-1], RP])          # симетрична сітка, 22 точки

# наш виміряний спектр із degree_spectrum.py — для контролю-двійника
TWIN = {1: 0.341, 2: 0.277, 3: 0.100, 4: 0.039, 5: 0.125, 6: 0.060, 8: 0.045}


def he(u, k):
    """Ймовірнісний поліном Ерміта He_k, нормований на одиничну дисперсію."""
    p = {1: u,
         2: u ** 2 - 1.0,
         3: u ** 3 - 3.0 * u,
         4: u ** 4 - 6.0 * u ** 2 + 3.0,
         5: u ** 5 - 10.0 * u ** 3 + 15.0 * u,
         6: u ** 6 - 15.0 * u ** 4 + 45.0 * u ** 2 - 15.0,
         8: (u ** 8 - 28.0 * u ** 6 + 210.0 * u ** 4 - 420.0 * u ** 2 + 105.0)}[k]
    return p / math.sqrt(math.factorial(k))


def fwd(wc, z):
    x = z
    for i in range(L_):
        x = torch.relu(x @ wc[i])
    return x


def fwd_soft(wc, z, beta):
    """Та сама мережа зі згладженою активацією softplus_β → relu при β→∞."""
    x = z
    for i in range(L_):
        h = x @ wc[i]
        x = torch.nn.functional.softplus(h, beta=beta)
    return x


def spectrum_parity(cs, var):
    """Розв'язок з РОЗДІЛЕНОЮ парністю.

    C(ρ) = Σ ρᵏ a_k.  Для симетричної сітки:
        E(ρ) = ½(C(ρ)+C(−ρ)) = Σ_{j≥1} (ρ²)^j a_{2j}
        O(ρ) = ½(C(ρ)−C(−ρ)) = ρ · Σ_{j≥0} (ρ²)^j a_{2j+1}
    Дві незалежні невід'ємні задачі — сусіди різної парності більше НЕ
    конкурують між собою, а це й був механізм плутанини a₄↔a₅.
    """
    from scipy.optimize import nnls
    c = np.asarray(cs)
    n = len(RP)
    cm, cp = c[:n][::-1], c[n:]            # C(−ρ) у порядку зростання ρ, C(+ρ)
    ev, od = 0.5 * (cp + cm), 0.5 * (cp - cm)
    u = RP ** 2
    ke = [k for k in range(2, KMAX + 1, 2)]
    ko = [k for k in range(1, KMAX + 1, 2)]
    Ae = np.array([[uu ** (k // 2) for k in ke] for uu in u])
    Ao = np.array([[uu ** (k // 2) for k in ko] for uu in u])
    ae, _ = nnls(Ae, ev)
    ao, _ = nnls(Ao, od / RP)
    a = np.zeros(KMAX)
    for j, k in enumerate(ke):
        a[k - 1] = ae[j]
    for j, k in enumerate(ko):
        a[k - 1] = ao[j]
    res = (np.linalg.norm(Ae @ ae - ev) + np.linalg.norm(Ao @ ao - od / RP))
    return a, res / max(abs(var), 1e-30)


def spectrum_old(cs, var):
    """Стара процедура №1 (лише ρ>0, спільна задача) — для прямого зіставлення."""
    from scipy.optimize import nnls
    n = len(RP)
    A = np.array([[r ** k for k in range(1, KMAX + 1)] for r in RP])
    b = np.asarray(cs)[n:]
    A = np.vstack([A, np.ones(KMAX)])
    b = np.concatenate([b, [var]])
    a, _ = nnls(A, b)
    return a


def run_scalar(fn, ns, dev, seed):
    """Спектр одновимірної функції від u=w·z (для контролів). Точний і швидкий."""
    g = torch.Generator(device=dev).manual_seed(seed)
    u = torch.randn(ns, device=dev, dtype=torch.float64, generator=g)
    up = torch.randn(ns, device=dev, dtype=torch.float64, generator=g)
    f0 = fn(u)
    m0, v0 = float(f0.mean()), float(f0.var())
    cs = []
    for r in RHOS:
        s = math.sqrt(1.0 - r * r)
        gg = fn(r * u + s * up)
        cs.append(float(((f0 - m0) * (gg - gg.mean())).mean()))
    a, res = spectrum_parity(cs, v0)
    return a / v0, res


def run_net(wc, u_out, ns, dev, seed, act='relu', beta=8.0):
    g = torch.Generator(device=dev).manual_seed(seed)
    F0, Z, ZP = [], [], []
    for _ in range(ns // CH):
        z = torch.randn(CH, W_, device=dev, generator=g)
        zp = torch.randn(CH, W_, device=dev, generator=g)
        Z.append(z)
        ZP.append(zp)
        o = fwd(wc, z) if act == 'relu' else fwd_soft(wc, z, beta)
        F0.append((o @ u_out).double())
    F0 = torch.cat(F0)
    m0, v0 = float(F0.mean()), float(F0.var())
    cs = []
    for r in RHOS:
        s = math.sqrt(1.0 - r * r)
        tot, nn = 0.0, 0
        gm = []
        for z, zp in zip(Z, ZP):
            zz = r * z + s * zp
            o = fwd(wc, zz) if act == 'relu' else fwd_soft(wc, zz, beta)
            gm.append((o @ u_out).double())
        gm = torch.cat(gm)
        cs.append(float(((F0 - m0) * (gm - gm.mean())).mean()))
    a, res = spectrum_parity(cs, v0)
    ao = spectrum_old(cs, v0)
    return a / v0, ao / v0, res, v0


def show(tag, a, res=None):
    cum = np.cumsum(a)
    body = ' '.join(f'{a[k]:.3f}' for k in range(6))
    tail = a[6:].sum()
    extra = f'  нев\'язка={res:.2e}' if res is not None else ''
    print(f'  {tag:26s} a₁..a₆ = {body}  ≥7={tail:.3f}  '
          f'|  ≤3={cum[2]:.3f} ≤4={cum[3]:.3f} ≤5={cum[4]:.3f}{extra}', flush=True)


def main(nets=6, dev=DEV):
    torch.manual_seed(4321)
    print(f'== КОНТРОЛІ (одновимірні, точний спектр відомий), ns=2**20 ==')
    NC = 1 << 20
    for k in (1, 2, 3, 4, 5):
        a, res = run_scalar(lambda t, k=k: he(t, k), NC, dev, 100 + k)
        show(f'✅ чистий He_{k}', a, res)

    # двійник: рівно наш виміряний спектр
    cf = {k: math.sqrt(v) for k, v in TWIN.items()}

    def twin(t):
        s = torch.zeros_like(t)
        for k, c in cf.items():
            s = s + c * he(t, k)
        return s
    a, res = run_scalar(twin, NC, dev, 777)
    show('✅ ДВІЙНИК нашого спектра', a, res)
    print('     істина двійника          a₁..a₆ = ' +
          ' '.join(f'{TWIN.get(k, 0.0):.3f}' for k in range(1, 7)) +
          f'  ≥7={TWIN[8]:.3f}  |  ≤3={sum(TWIN[k] for k in (1,2,3)):.3f} '
          f'≤4={sum(TWIN[k] for k in (1,2,3,4)):.3f} '
          f'≤5={sum(TWIN[k] for k in (1,2,3,4,5)):.3f}')
    # чи розрізняє прилад сусідів 4 і 5 у СУМІШІ
    for pair in ((4, 5), (5, 6)):
        def mix(t, pair=pair):
            return (he(t, pair[0]) + he(t, pair[1])) / math.sqrt(2.0)
        a, res = run_scalar(mix, NC, dev, 900 + pair[0])
        show(f'✅ суміш He_{pair[0]}+He_{pair[1]} (0.5/0.5)', a, res)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    for NS in (65536, 262144):
        print(f'\n== МЕРЕЖІ, ns={NS} ==')
        A, AO = [], []
        for n, f in enumerate(fs):
            sd0 = int(np.load(f)['seed'])
            W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
                 for q in range(L_)]
            with torch.no_grad():
                wc, _, _ = cut_net(W, dev)
                nw = wc[-1].shape[1]
                uo = torch.ones(nw, device=dev) / math.sqrt(nw)
                a, ao, res, v0 = run_net(wc, uo, NS, dev, 5000 + n)
                A.append(a)
                AO.append(ao)
            show(f'мережа {n+1}  парність', a, res)
            show(f'мережа {n+1}  стара №1', ao)
            del W, wc
            torch.cuda.empty_cache()
        A, AO = np.array(A), np.array(AO)
        m, s = A.mean(0), A.std(0)
        print(f'\n  СЕРЕДНЄ по {len(fs)} мережах (парність), ns={NS}')
        print(f'  {"k":>3s} {"a_k":>8s} {"±":>8s} {"накопич":>9s} {"стеля d=k":>10s}')
        c = 0.0
        for k in range(min(10, KMAX)):
            c += m[k]
            tail = max(A[:, k + 1:].sum(1).mean(), 1e-9)
            print(f'  {k+1:3d} {m[k]:8.4f} {s[k]:8.4f} {c:9.4f} {1.0/tail:9.2f}×')
        print(f'  стара процедура №1: ≤3={AO.mean(0)[:3].sum():.3f} '
              f'≤4={AO.mean(0)[:4].sum():.3f} ≤5={AO.mean(0)[:5].sum():.3f}')

    print('\n== 🔬 SOFTPLUS: чи хвіст ≥6 — це ЗЛАМИ? ==')
    for n, f in enumerate(fs[:3]):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            nw = wc[-1].shape[1]
            uo = torch.ones(nw, device=dev) / math.sqrt(nw)
            for bt in (16.0, 4.0, 1.0):
                a, _, res, v0 = run_net(wc, uo, 65536, dev, 5000 + n,
                                        act='soft', beta=bt)
                show(f'мережа {n+1} softplus β={bt:g}', a, res)
        del W, wc
        torch.cuda.empty_cache()
    print('\n🔑 читати так: якщо ≥6 обвалюється зі спаданням β — хвіст це ЗЛАМИ, '
          'і 9.49× є стелею ГЛАДКОСТІ, а не стелею методу')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
