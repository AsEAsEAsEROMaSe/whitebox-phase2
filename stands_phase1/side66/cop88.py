"""КОПУЛА ЯК ЄДИНЕ НЕВІДОМЕ: ЧИ ЗВОДИТЬСЯ ВОНА ДО ОДНОГО `ρ_eff`? (С88)

## ЗВІДКИ — ВИБІР ЮЗЕРА ПІСЛЯ ШЕСТИ ЗАКРИТТІВ
`herm88` зміряв: гаусова копула + ТОЧНІ маргіналі дає ×1.07 і далі ГІРШЕ
(N=2 → істинні 10.15% проти гаусових 4.67%) ⟹ 🎯 **негаусовість живе в
КОПУЛІ**, а не в маргіналях. Це єдине джерело, куди ще не били прицільно.

## 📐 ТОЧНА ТОТОЖНІСТЬ, ЯКА РОБИТЬ КОПУЛУ ЄДИНИМ НЕВІДОМИМ
Для невід'ємних величин `E[XY] = ∫₀^∞∫₀^∞ P(X>x, Y>y) dx dy`.
Оскільки `a = ReLU(h) ≥ 0` і `{a > x} = {h > x}` при `x > 0`:
```
🔑 E[a_j a_k] = ∫₀^∞ ∫₀^∞ P(h_j > x, h_k > y) dx dy
```
Точно, без жодних припущень. ⟹ уся інформація сидить у СУМІСНІЙ хвостовій
функції пари. Маргіналі ми вже вміємо; лишається копула.

## 💡 ГІПОТЕЗА: КОПУЛА ЗВОДИТЬСЯ ДО ОДНОГО ЧИСЛА НА ПАРУ
Замість моделювати копулу як функцію, спитати: чи існує **ефективна
кореляція** `ρ_eff`, при якій ГАУСОВА формула дає ПРАВИЛЬНУ `Cov`?
```
Cov_gauss(t_j, t_k, ρ_eff) = Cov_true(j,k)      →  розв'язати відносно ρ_eff
δρ := ρ_eff − ρ
```
Якщо `δρ` — проста функція від `(t_j, t_k, ρ)`, то копула коштує ОДИН скаляр
на пару, і його можна вчити або виводити.
📌 Це принципово краще за пряму таблицю похибки: `ρ_eff` має ФІЗИЧНИЙ зміст,
обмежений `[−1,1]`, і автоматично зберігає структурні обмеження `Cov`
(на відміну від адитивної поправки, яка може вивести матрицю з PSD).

## 🎯 ЩО МІРЯЄТЬСЯ
 (1) розподіл `δρ` по шарах: величина, знак, чи систематичний;
 (2) 🔑`R²` навчання `δρ` від `(t_j,t_k,ρ)` — ПРОТИ `R²=0.666` прямої похибки.
     Якщо `δρ` вчиться КРАЩЕ — параметризація вдала;
 (3) стеля: підставити ОРАКУЛЬНИЙ `ρ_eff` у ланцюг — скільки дає;
 (4) чи `δρ` стабільний МІЖ мережами (інакше вчити нема сенсу).

## ⛔ ПЛОМБИ
 P1 🔑ШАР 1: `m ≡ 0`, вхід ТОЧНО гаусів ⟹ копула гаусова ⟹ `δρ ≈ 0`.
    Це аналітична пломба: якщо `δρ` там не нуль — баг у розв'язувачі.
 P2 МОНОТОННІСТЬ `Cov_gauss(ρ)`: перевіряємо на сітці перед бісекцією.
    Якщо немонотонна — `ρ_eff` не визначений однозначно, і гіпотеза падає
    ДО виміру (чесно фіксуємо).
 P3 ЗБІЖНІСТЬ бісекції: залишок `|Cov_gauss(ρ_eff) − Cov_true|` ~1e-12.
 P4 ШУМ: половинки вибірки дають той самий `δρ`.
 P5 частка пар, де розв'язку НЕМА (`Cov_true` поза досяжним діапазоном) —
    друкуємо. Якщо велика, параметризація неповна.

## 📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ
 A. `δρ` буде СИСТЕМАТИЧНО ВІД'ЄМНИМ: реальні нейрони «злипаються» слабше,
    ніж передбачає гаусова копула при тих самих маргіналях. Шанс **55%**.
 B. 🎯`R²` для `δρ` буде **вищим** за 0.666 (краща параметризація). Шанс
    **45%**. Якщо так — це перша за день ознака, що клас не вичерпано.
 C. Частка пар без розв'язку &lt; 5%. Шанс **60%**.
 D. Оракульний `ρ_eff` у ланцюгу дасть ×2–×6. Шанс **40%**.
    ⚠️ Навіть ×6 дає у балі лише ×1.30 — сабміта це не робить, але вперше
    показало б, що детермінована гілка НЕ вичерпана, а лише недорозкрита.
"""
import glob, math, sys
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
PROBE = (0, 4, 8, 12, 16, 20, 24, 28, 31)
NB = 14
assert DEV not in ('cuda:0', 'cuda:4'), 'cuda:0 ТАБУ, cuda:4 — монітор юзера'
print(__doc__.split('📣')[1], flush=True)

I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
KMAX = 12
WK = torch.tensor([1.0 / ((k + 1) * math.factorial(k)) for k in range(1, KMAX + 1)],
                  dtype=torch.float64, device=DEV)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
NF = len(FS)
print(f'{NF} мереж · 2^{LOG2N} · {DEV}\n', flush=True)


def herm(t, kmax):
    out = [torch.ones_like(t)]
    if kmax > 1:
        out.append(t.clone())
    for n in range(1, kmax - 1):
        out.append(t * out[n] - n * out[n - 1])
    return out[:kmax]


def cov_gauss(P_j, P_k, cj, ck, r):
    """`Cov/(s_j s_k)` за рядом Прайса (K=KMAX) для гаусової пари з кор. `r`.
    `cj[k]` — коефіцієнти `He_{k−1}(t)φ(t)` для кожного елемента."""
    acc = r * (P_j * P_k)
    rp = r.clone()
    for k in range(1, KMAX + 1):
        rp = rp * r
        acc = acc + WK[k - 1] * rp * (cj[k - 1] * ck[k - 1])
    return acc


DRHO = np.zeros((NF, len(PROBE), 3))    # середнє, std, частка |δρ|>0.02
NOSOL = np.zeros((NF, len(PROBE)))
MULTI = np.zeros((NF, len(PROBE)))   # частка пар із ≥2 коренями (викинуті)
RESID = np.zeros((NF, len(PROBE)))
P1SEAL = np.zeros(NF)
MONO = np.zeros(NF)
R2D = np.zeros(NF)                       # R² навчання δρ (LOO по мережах)
BINS = torch.zeros(NF, NB, NB, NB, dtype=torch.float64, device=DEV)
CNT = torch.zeros(NF, NB, NB, NB, dtype=torch.float64, device=DEV)
SS = torch.zeros(NF, dtype=torch.float64, device=DEV)
EDG = None
TQ = torch.tensor(np.linspace(0, 1, NB + 1)[1:-1], dtype=torch.float64, device=DEV)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                        device=DEV, dtype=torch.float64) for q in range(L_)]
    with torch.no_grad():
        A1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        AA = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(1200 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                h = (a @ W32[q].float()).double()
                aq = torch.clamp(h, min=0.0)
                A1[q] += aq.sum(0); AA[q] += aq.T @ aq
                a = aq.float()
            d += b
        prevm = prevC = None
        for q in range(L_):
            mq = A1[q] / NS
            Cq = AA[q] / NS - torch.outer(mq, mq)
            if q == 0:
                C_h = W32[0].T @ W32[0]
                m_h = torch.zeros(W_, dtype=torch.float64, device=DEV)
            else:
                C_h = W32[q].T @ (prevC @ W32[q])
                m_h = prevm @ W32[q]
            s = torch.sqrt(torch.clamp(torch.diagonal(C_h), min=1e-30))
            t = m_h / s
            R = (C_h / torch.outer(s, s)).clamp(-0.999, 0.999)
            P_, f_ = Phi(t), phi(t)
            He = herm(t, KMAX)
            cs = [He[k] * f_ for k in range(KMAX)]
            iu = torch.triu_indices(W_, W_, offset=1, device=DEV)
            j0, j1 = iu[0], iu[1]
            Pj, Pk = P_[j0], P_[j1]
            cj = [c[j0] for c in cs]; ck = [c[j1] for c in cs]
            rho = R[j0, j1]
            tgt = Cq[j0, j1] / (s[j0] * s[j1])
            # ⛔P2: монотонність на сітці
            grid = torch.linspace(-0.999, 0.999, 41, dtype=torch.float64,
                                  device=DEV)
            gv = torch.stack([cov_gauss(Pj[:2048], Pk[:2048],
                                        [c[:2048] for c in cj],
                                        [c[:2048] for c in ck],
                                        torch.full_like(Pj[:2048], float(gg)))
                              for gg in grid])
            MONO[n] += float((gv[1:] < gv[:-1]).double().mean()) / L_
            # бісекція
            lo = torch.full_like(rho, -0.999); hi = torch.full_like(rho, 0.999)
            flo = cov_gauss(Pj, Pk, cj, ck, lo) - tgt
            fhi = cov_gauss(Pj, Pk, cj, ck, hi) - tgt
            bad = (flo * fhi) > 0
            # 🚨C88/chk88: 5.27% пар МАЮТЬ ≥2 КОРЕНІ ⟹ бісекція для них дає
            #    сміття. Перша версія зараховувала їх у «немає розв'язку»
            #    (я доповів 10%, насправді 0.06%). Виявляємо і ВИКИДАЄМО.
            gsc = torch.linspace(-0.999, 0.999, 129, dtype=torch.float64,
                                 device=DEV)
            sg = None; nroot = torch.zeros_like(rho)
            for gg in gsc:
                v = cov_gauss(Pj, Pk, cj, ck, torch.full_like(rho, float(gg))) - tgt
                sn = torch.sign(v)
                if sg is not None:
                    nroot = nroot + (sn * sg < 0).double()
                sg = sn
            multi = nroot >= 2
            bad = bad | multi
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                fm = cov_gauss(Pj, Pk, cj, ck, mid) - tgt
                go = (fm * flo) > 0
                lo = torch.where(go, mid, lo); flo = torch.where(go, fm, flo)
                hi = torch.where(go, hi, mid)
            reff = 0.5 * (lo + hi)
            drho = torch.where(bad, torch.zeros_like(reff), reff - rho)
            res = (cov_gauss(Pj, Pk, cj, ck, reff) - tgt).abs()
            if q in PROBE:
                jj = PROBE.index(q)
                ok = ~bad
                DRHO[n, jj, 0] = float(drho[ok].mean())
                DRHO[n, jj, 1] = float(drho[ok].std())
                DRHO[n, jj, 2] = float((drho[ok].abs() > 0.02).double().mean())
                NOSOL[n, jj] = float(bad.double().mean())
                MULTI[n, jj] = float(multi.double().mean())
                RESID[n, jj] = float(res[ok].max())
                if q == 0:
                    P1SEAL[n] = float(drho[ok].abs().mean())
            # бінінг δρ
            lo_t = torch.minimum(t[j0], t[j1]); hi_t = torch.maximum(t[j0], t[j1])
            if EDG is None:
                EDG = (torch.quantile(lo_t, TQ), torch.quantile(hi_t, TQ),
                       torch.quantile(rho, TQ))
            b0 = torch.bucketize(lo_t, EDG[0]); b1 = torch.bucketize(hi_t, EDG[1])
            b2 = torch.bucketize(rho, EDG[2])
            idx = (b0 * NB + b1) * NB + b2
            msk_ok = ~bad
            BINS[n].view(-1).index_add_(0, idx[msk_ok], drho[msk_ok])
            CNT[n].view(-1).index_add_(0, idx[msk_ok],
                                       torch.ones_like(drho[msk_ok]))
            SS[n] += (drho[msk_ok] ** 2).sum()
            prevm, prevC = mq, Cq
        del AA
    print(f'  net{n}  δρ шар32: {DRHO[n,-1,0]:+.4f} ± {DRHO[n,-1,1]:.4f} · '
          f'без розв. {NOSOL[n,-1]:.1%}', flush=True)

# ---------- R² навчання `δρ` з LOO по мережах ----------
tot = float(SS.sum()); ssr = 0.0
for i in range(NF):
    m_ = torch.ones(NF, dtype=torch.bool, device=DEV); m_[i] = False
    Fl = BINS[m_].sum(0) / torch.clamp(CNT[m_].sum(0), min=1.0)
    ssr += float((BINS[i] * Fl * 2 - CNT[i] * Fl * Fl).sum())
R2 = ssr / max(tot, 1e-30)

print(f'\n{"=" * 72}\n⛔ ПЛОМБИ')
print(f' P1  |δρ| на шарі 1 = {P1SEAL.mean():.5f}  (вхід ТОЧНО гаусів ⟹ ~0)')
print(f' P2  частка немонотонних ділянок = {MONO.mean():.2%}  (мусить ~0)')
print(f' P3  залишок бісекції = {RESID.max():.2e}  (мусить ~1e-12)')
print(f' P5  пар відкинуто = {NOSOL[:,1:].mean():.2%}, з них БАГАТОЗНАЧНИХ '
      f'= {MULTI[:,1:].mean():.2%}  (chk88: 5.27% ≥2 корені, 0.06% без коренів)')

print(f'\n{"=" * 72}\n📈 `δρ = ρ_eff − ρ` ПО ГЛИБИНІ ({NF} мереж)')
print(f'{"шар":>5s} {"сер. δρ":>10s} {"std δρ":>10s} {"|δρ|>0.02":>11s} '
      f'{"без розв.":>10s}')
for j, q in enumerate(PROBE):
    print(f'{q+1:5d} {DRHO[:,j,0].mean():+10.4f} {DRHO[:,j,1].mean():10.4f} '
          f'{DRHO[:,j,2].mean():11.2%} {NOSOL[:,j].mean():10.2%}')

print(f'\n{"=" * 72}\n📣 ВИРОК')
print(f' A. знак δρ: {"ВІД\'ЄМНИЙ" if DRHO[:,1:,0].mean() < 0 else "ДОДАТНИЙ"} '
      f'(сер. {DRHO[:,1:,0].mean():+.4f}) — передбачав від\'ємний')
print(f' B. 🎯R² навчання δρ = {R2:.4f}  (пряма похибка дає 0.666)')
print(f'    ⟹ {"КРАЩА" if R2 > 0.666 else "ГІРША"} параметризація')
