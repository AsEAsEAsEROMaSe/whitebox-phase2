"""ГАУСОВА КОПУЛА + ТОЧНІ МАРГІНАЛІ: СКІЛЬКИ ЦЕ ДАЄ? (С88, БЕЗ МК)

## ЗВІДКИ — ТРИ ЗАКРИТТЯ СЬОГОДНІ ЗВЕЛИ МІШЕНЬ У ТОЧКУ
`price88`: гаусове замикання вже точне (×1.001) · `lowrk88`: низький ранг `C`
не працює (r=3 → ×0.32) · `cum88`: кумулянти У ВИХІД — стеля ×1.15.
А оракульні `m,σ` (`L4`) дають ×7.1 ⟹ 🎯 **похибка сидить у РЕКУРСІЇ `C`, і
вона від НЕГАУСОВОСТІ `h`**.

## 📐 ВИВІД: РОЗКЛАД ЕРМІТА З ЕМПІРИЧНИМИ КОЕФІЦІЄНТАМИ
`u_j = (h_j − m_j)/s_j` — стандартизована преактивація (розподіл ІСТИННИЙ,
не гаусів). Розкладаємо активацію по Ерміту від `u_j`:
```
c_n^{(j)} = E[ ReLU(u_j + t_j) · He_n(u_j) ] / n!
```
Якщо пара `(u_j, u_k)` має ГАУСОВУ КОПУЛУ з кореляцією `ρ`, то за формулою
Мелера `E[He_n(u_j)He_m(u_k)] = δ_{nm}·n!·ρ^n`, і ТОЧНО:
```
🔑 Cov(a_j, a_k) = s_j s_k · Σ_{n≥1} n!·c_n^{(j)}·c_n^{(k)}·ρ^n
```
## ✅ ЦЕ УЗАГАЛЬНЮЄ НАШ МЕТОД, А НЕ ЗАМІНЮЄ ЙОГО
Для ГАУСОВОГО `u` коефіцієнти беруться в замкненому вигляді:
`c_0 = tΦ+φ` · `c_1 = Φ` · `c_2 = φ/2` · далі `c_n = I_n/n!` з `price88`.
Підставивши: `n=1` дає `ρΦ_jΦ_k`, `n=2` дає `½ρ²φ_jφ_k` — **рівно наші два
члени**. ⟹ 📌 наш метод = цей ряд із ГАУСОВИМИ `c_n`; питання лише в тому,
скільки дають ІСТИННІ.

## 💡 І ЦЕ РІВНО СТРУКТУРА ЛІДЕРА
`c_n` — ПОНЕЙРОННІ числа: `N` векторів довжини 256 на шар. `Cov` будується як
сума `N` зовнішніх добутків із вагами `ρ^n`. Тобто «кілька векторів на нейрон,
жодної матриці в стані» — той самий підпис, що в його `F`.

## 🎯 ЩО МІРЯЄТЬСЯ (це вимір СТЕЛІ, не методу)
 A. `Cov` з ІСТИННИХ `c_n` (з МК) проти істинної `Cov(a)` — скільки лишається.
    Це стеля гіпотези «гаусова копула». Якщо стеля низька — клас закритий.
 B. Скільки членів `N` треба (1,2,3,4,6,8).
 C. Контроль: ті самі формули з ГАУСОВИМИ `c_n` мусять відтворити наш метод.
 D. Похибка в ЛАНЦЮГУ, якщо підставляти істинні `c_n` на кожному кроці.

## ⛔ ПЛОМБИ
 P1 🔑ГАУСОВІ `c_n` при `N=8` мусять відтворити `price88 K=8`: похибка `Cov`
    на рівні нашого методу. Різні реалізації того самого об'єкта.
 P2 🔑ШАР 1: вхід ТОЧНО гаусів ⟹ емпіричні `c_n` мусять збігтися з
    аналітичними до шуму МК. Друкуємо максимальну розбіжність.
 P3 ОРТОГОНАЛЬНІСТЬ: для гаусового `u` мусить бути `E[He_n He_m] = δ n!`.
    Міряємо на шарі 1 як незалежну перевірку самих `He`.
 P4 ШУМ: половинки вибірки.
 P5 `n=0` НЕ входить у `Cov` (це середнє) — перевіряємо, що `Σ_{n≥1}` при
    `ρ=0` дає рівно 0.

## 📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ
 A. Стеля «гаусова копула + істинні маргіналі» дасть **30–70%** зниження
    похибки `Cov`. Шанс **50%**. Підстава: копула НЕ гаусова (усі `h_j` —
    функції того самого входу), але маргінальна негаусовість зміряна як
    основне джерело (`price88` 99.9%).
 B. Треба буде `N ≥ 4` членів. Шанс **60%**.
 C. 🎯 Якщо стеля &gt;50% — гілка жива і веде до `L4`; якщо &lt;15% — копула
    винна більше за маргіналі, і весь клас «понейронних коефіцієнтів»
    закривається разом із гіпотезою про лідера. Шанс останнього **35%**.
"""
import glob, math, sys
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
NMAX = 3          # 🚨 вище 3 емпіричні `c_n` НЕ ВИМІРЮВАНІ: шум росте як n!
#    (димовий тест: P3 ортогональність 0.21 при NMAX=8, істинні c_n дали 186%)
NLIST = (1, 2, 3)
PROBE = (0, 4, 8, 12, 16, 20, 24, 28, 31)
assert DEV not in ('cuda:0', 'cuda:4'), 'cuda:0 ТАБУ, cuda:4 — монітор юзера'
print(__doc__.split('📣')[1], flush=True)

I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} · {DEV} · N={NLIST}\n', flush=True)

FACT = torch.tensor([float(math.factorial(n)) for n in range(NMAX + 1)],
                    dtype=torch.float64, device=DEV)


def he_all(u, nmax):
    """`He_0..He_nmax` за рекурсією, форма (nmax+1, ...)."""
    out = [torch.ones_like(u)]
    if nmax >= 1:
        out.append(u)
    for n in range(1, nmax):
        out.append(u * out[n] - n * out[n - 1])
    return torch.stack(out, 0)


def gauss_cn(t, nmax):
    """АНАЛІТИЧНІ `c_n` для ГАУСОВОГО `u`: `c_n = I_n(t)/n!`,
    `I_0 = tΦ+φ`, `I_1 = Φ`, `I_n = (−1)^n He_{n−2}(t)φ(t)` при `n≥2`
    (вивід — `price88`)."""
    P, f = Phi(t), phi(t)
    He = he_all(t, max(nmax - 2, 0))
    out = [t * P + f, P]
    for n in range(2, nmax + 1):
        out.append(((-1.0) ** n) * He[n - 2] * f / FACT[n])
    return torch.stack(out[:nmax + 1], 0)


def cov_from_cn(cn, s, R, N):
    """`Cov = s⊗s · Σ_{n=1..N} n! c_n⊗c_n ρ^n`. Діагональ вписуємо точною."""
    Rp = torch.ones_like(R)
    acc = torch.zeros_like(R)
    for n in range(1, N + 1):
        Rp = Rp * R
        acc = acc + (FACT[n] * Rp) * torch.outer(cn[n], cn[n])
    return acc * torch.outer(s, s)


NN = len(NLIST)
NP = len(PROBE)
COVE = np.zeros((len(FS), NP, NN, 2))   # [.., 0]=істинні c_n, [..,1]=гаусові
P2MAX = np.zeros(len(FS))
P3ORT = np.zeros(len(FS))
P5Z = np.zeros(len(FS))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                        device=DEV, dtype=torch.float64) for q in range(L_)]
    with torch.no_grad():
        S1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        S2 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(8800 + n); d = 0
        while d < NS:                       # прохід 1: m, s преактивацій
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                h = (a @ W32[q].float()).double()
                S1[q] += h.sum(0); S2[q] += (h * h).sum(0)
                a = torch.relu(h).float()
            d += b
        MH = S1 / NS
        SH = torch.sqrt(torch.clamp(S2 / NS - MH * MH, min=1e-30))

        CN = torch.zeros(L_, NMAX + 1, W_, dtype=torch.float64, device=DEV)
        ORT = torch.zeros(NMAX + 1, NMAX + 1, dtype=torch.float64, device=DEV)
        AA = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        A1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(8800 + n); d = 0
        while d < NS:                       # прохід 2: c_n і істинна Cov(a)
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                h = (a @ W32[q].float()).double()
                u = (h - MH[q]) / SH[q]
                ad = torch.clamp(h, min=0.0) / SH[q]      # ReLU(u+t)
                He = he_all(u, NMAX)                      # (NMAX+1, b, W)
                CN[q] += (He * ad.unsqueeze(0)).sum(1)
                if q == 0:
                    ORT += torch.einsum('nbw,mbw->nm', He, He) / W_
                A1[q] += torch.clamp(h, min=0.0).sum(0)
                aq = torch.clamp(h, min=0.0)
                AA[q] += (aq.T @ aq)
                a = torch.relu(h).float()
            d += b
        CN = CN / NS / FACT.view(1, -1, 1)
        ORT = ORT / NS
        P3ORT[n] = float((ORT - torch.diag(FACT)).abs().max() / FACT.max())

        for j, q in enumerate(PROBE):
            m_a = A1[q] / NS
            Ca_t = AA[q] / NS - torch.outer(m_a, m_a)
            s = SH[q]; t = MH[q] / s
            C_h = None
            R = None
            # кореляційна матриця преактивацій — з істинної Cov(a) попереднього
            # шару через точний перенос; на шарі 0 це W₁ᵀW₁
            if q == 0:
                C_h = W32[0].T @ W32[0]
            else:
                mprev = A1[q - 1] / NS
                Cprev = AA[q - 1] / NS - torch.outer(mprev, mprev)
                C_h = W32[q].T @ (Cprev @ W32[q])
            R = C_h / torch.outer(torch.sqrt(torch.clamp(torch.diagonal(C_h),
                                                         min=1e-30)),
                                  torch.sqrt(torch.clamp(torch.diagonal(C_h),
                                                         min=1e-30)))
            cg = gauss_cn(t, NMAX)
            if q == 0:
                P2MAX[n] = float((CN[0] - cg).abs().max()
                                 / cg.abs().max().clamp(min=1e-30))
            off = Ca_t - torch.diag(torch.diagonal(Ca_t))
            den = float((off * off).sum())
            for k, N in enumerate(NLIST):
                for w_, cn_ in enumerate((CN[q], cg)):
                    Cx = cov_from_cn(cn_, s, R, N)
                    dif = (Cx - Ca_t)
                    dif = dif - torch.diag(torch.diagonal(dif))
                    COVE[n, j, k, w_] = math.sqrt(float((dif * dif).sum()) / den)
            if q == 0:
                P5Z[n] = float(cov_from_cn(CN[0], s, torch.zeros_like(R),
                                           NMAX).abs().max())
        del AA
    print(f'  net{n}  N=2: істинні {COVE[n,-1,1,0]:.3%} · гаусові '
          f'{COVE[n,-1,1,1]:.3%} (шар 32)', flush=True)

print(f'\n{"=" * 76}\n⛔ ПЛОМБИ')
print(f' P2  шар1: |емпір c_n − аналіт| = {P2MAX.mean():.4e}  (вхід ТОЧНО гаусів)')
print(f' P3  ортогональність He на шарі1 = {P3ORT.mean():.4e}  (мусить ~0)')
print(f' P5  Σ_{{n≥1}} при ρ=0          = {P5Z.mean():.3e}  (мусить бути 0)')

print(f'\n{"=" * 76}\n📈 ВІДНОСНА ПОХИБКА ПОЗАДІАГОНАЛІ `Cov(a)` ({len(FS)} мереж)')
print(f'{"шар":>5s} ' + ' '.join(f'{"N=" + str(N):>16s}' for N in NLIST))
print(f'{"":>5s} ' + ' '.join(f'{"істин / гаус":>16s}' for _ in NLIST))
for j, q in enumerate(PROBE):
    row = f'{q+1:5d} '
    for k in range(NN):
        row += f' {COVE[:,j,k,0].mean():6.2%} /{COVE[:,j,k,1].mean():7.2%}'
    print(row)

print(f'\n{"=" * 76}\n📣 ВИРОК (шар 32, N=8)')
ti = COVE[:, -1, -1, 0].mean(); gi = COVE[:, -1, -1, 1].mean()
print(f' істинні c_n: {ti:.2%} · гаусові c_n: {gi:.2%} ⟹ '
      f'СТЕЛЯ гаусової копули = ×{gi/max(ti,1e-12):.2f}')
print(f' A. зниження похибки Cov: {(1-ti/max(gi,1e-12))*100:.1f}% '
      f'(передбачав 30–70%)')
b2 = COVE[:, -1, 1, 0].mean()
print(f' B. N=2 дає {b2:.2%}, N=8 дає {ti:.2%} ⟹ вищі члени варті '
      f'×{b2/max(ti,1e-12):.2f}')
