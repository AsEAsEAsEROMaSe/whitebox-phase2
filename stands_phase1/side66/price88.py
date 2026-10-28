"""СКІЛЬКИ З ПОХИБКИ — ЦЕ НАШЕ ОБРІЗАННЯ РЯДУ, А НЕ НЕГАУСОВІСТЬ? (С88, БЕЗ МК)

## 📐 ВИВІД (не вгадування): ТОЧНА ФОРМУЛА ЧЕРЕЗ ТЕОРЕМУ ПРАЙСА
`X = m_j + s_j U`, `Y = m_k + s_k V`, `(U,V)` стандартні з кореляцією `ρ`.
Теорема Прайса: `∂E[f(X)g(Y)]/∂Cov(X,Y) = E[f'(X)g'(Y)]`.
Для ReLU похідна — індикатор ⟹
```
∂/∂(s_j s_k ρ) E[ReLU(X)ReLU(Y)] = P(X>0, Y>0) = Φ₂(t_j, t_k; ρ)
```
При `ρ=0` величини незалежні ⟹ `E[ReLU·ReLU] = m_a_j · m_a_k`. Отже ТОЧНО:
```
🔑 Cov(a_j, a_k) = s_j·s_k · ∫₀^ρ Φ₂(t_j, t_k; r) dr
```
Розкладаємо `Φ₂` рядом Мелера і інтегруємо почленно:
```
Φ₂(a,b;r) = Φ(a)Φ(b) + Σ_{k≥1} (r^k/k!)·He_{k−1}(a)φ(a)·He_{k−1}(b)φ(b)
⟹ Cov/(s_j s_k) = ρΦ_jΦ_k + Σ_{k≥1} w_k ρ^{k+1} c_k(t_j) c_k(t_k)
   w_k = 1/((k+1)·k!) ,  c_k(t) = He_{k−1}(t)·φ(t)
```
Перші члени: `w₁=1/2, c₁=φ` · `w₂=1/6, c₂=tφ` · `w₃=1/24, c₃=(t²−1)φ` ·
`w₄=1/120, c₄=(t³−3t)φ` · `w₅=1/720` · `w₆=1/5040`.

🚨 **ЧЛЕН `k=1` — ЦЕ РІВНО НАШ**: `½·C²/(s⊗s)·φφᵀ` в `_closure_parts`.
⟹ 📌 НАШЕ ЗАМИКАННЯ = ЦЕЙ РЯД, ОБРІЗАНИЙ НА ПЕРШОМУ ЧЛЕНІ. Тобто питання
«обрізання чи негаусовість» має ПРЯМУ відповідь: додати члени і подивитись.

## 💰 ЧОМУ ЦЕ СТАЛО МОЖЛИВИМ ЩОЙНО
`lowrk88` зміряв: наш ПОВНИЙ детермінований ланцюг коштує **C/B = 0.00395**,
тобто 0.4% бюджету при підлозі множника 10% ⟹ **запас ×25 не використовується**
(10.02% з'їдає МК). Кожен додатковий член ряду — це одне поелементне множення
матриці 256² плюс зовнішній добуток ⟹ ~3·65536 флопів на шар = **6.3e6 на всі
32 шари**, тобто `C/B ≈ 2e-5` за член. Можна додати ДЕСЯТКИ членів безкоштовно.
⚠️ Раніше цього не бачили, бо гейт ціни був завищений у 7× (`fnp` 30.6 мкс).

## ⛔ ПЛОМБИ (інструмент перевіряє СЕБЕ до висновків)
 P1 `K=1` мусить відтворити НАШ ланцюг: `lowrk88` дав **0.9299%** (24 мережі).
 P2 🔑АНАЛІТИЧНА, на шарі 1, де `m ≡ 0` ⟹ `t=0`: ряд мусить збігтися до
    класичної формули Чо–Саула
    `Cov = (s_j s_k/2π)(√(1−ρ²) + ρ(π/2 + arcsin ρ) − 1)`.
    Це НЕЗАЛЕЖНИЙ шлях до того самого числа (не наша алгебра).
 P3 ЗБІЖНІСТЬ: члени ряду мусять СПАДАТИ. Друкуємо норму кожного.
 P4 ДІАГОНАЛЬ вписується точною формулою при БУДЬ-ЯКОМУ `K` (як в естиматорі)
    ⟹ розбіжність діагоналі ~1e-15 незалежно від `K`.
 P5 ШУМ МК-істини: дві незалежні половини.

## 📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ
 A. P1 і P2 пройдуть. Шанс **80%**.
 B. `K=6` дасть **0.35–0.75%** (зараз 0.930%), тобто обрізання відповідає за
    20–60% похибки. Шанс **55%**.
    ⚠️ Наш запис «Мелер K=3..6 = нуль при великих `t`» стосувався поправки до
    `E[ReLU]`, а НЕ до `Cov`. Якщо він поширюється й сюди — виграшу не буде
    і гілка закриється одним прогоном. Шанс такого результату **30%**.
 C. Ряд збігатиметься ПОВІЛЬНО там, де `|ρ|→1` — а це саме глибокі шари
    (еф. ранг 3.1 ⟹ нейрони майже злиплися). Шанс **65%**.
    📌 Якщо так — потрібна не більша `K`, а пряма квадратура по `r`.
 D. 🎯 Стеля: `K=20` покаже, скільки взагалі можна забрати обрізанням.
    Різниця «`K=20` ↔ істина» = ЧИСТА негаусовість, якої рядом не дістати.
"""
import glob, math, sys
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
KS = (1, 2, 3, 4, 6, 8, 12, 20)
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
print(f'{len(FS)} мереж · 2^{LOG2N} · {DEV} · K={KS}\n', flush=True)

KMAX = max(KS)
WK = torch.tensor([1.0 / ((k + 1) * math.factorial(k)) for k in range(1, KMAX + 1)],
                  dtype=torch.float64, device=DEV)


def herm(t, kmax):
    """`He_0..He_{kmax−1}` за рекурсією `He_{n+1} = t·He_n − n·He_{n−1}`."""
    out = [torch.ones_like(t)]
    if kmax > 1:
        out.append(t.clone())
    for n in range(1, kmax - 1):
        out.append(t * out[n] - n * out[n - 1])
    return out[:kmax]


def closure(m, C, K, diag_norms=None, seal=None):
    """Замикання кроку з рядом Прайса порядку `K`. `K=1` = наш нинішній метод.
    ДІАГОНАЛЬ завжди вписується точною формулою (як `_closure_parts`)."""
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s
    P, f = Phi(t), phi(t)
    m_a = m * P + s * f
    Ca = C * torch.outer(P, P)
    R = C / torch.outer(s, s)                    # кореляційна
    Rp = R.clone()                               # ρ^1
    He = herm(t, K)
    sso = torch.outer(s, s)
    for k in range(1, K + 1):
        Rp = Rp * R                              # ρ^{k+1}
        c = He[k - 1] * f
        term = (WK[k - 1] * Rp) * torch.outer(c, c)
        Ca = Ca + term * sso
        if diag_norms is not None:
            diag_norms[k - 1] += float(torch.linalg.matrix_norm(term * sso))
    d = (m * m + s * s) * P + (m * s) * f - m_a * m_a
    Ca = Ca + torch.diag(d - torch.diagonal(Ca))
    if seal is not None:   # ⛔P4: після вписування діагональ ТОТОЖНО = `d`
        seal[0] = max(seal[0], float((torch.diagonal(Ca) - d).abs().max()))
    return m_a, Ca


# ---------- ⛔ P2: АНАЛІТИЧНА ПЛОМБА НА `t=0` (формула Чо–Саула) ----------
rho = torch.linspace(-0.98, 0.98, 21, dtype=torch.float64, device=DEV)
cho = (torch.sqrt(1 - rho * rho) + rho * (math.pi / 2 + torch.asin(rho)) - 1) / (2 * math.pi)
f0 = 1.0 / math.sqrt(2 * math.pi)
ser = rho * 0.25          # 🚨 член `ρ·Φ(t_j)Φ(t_k)`; при t=0 це рівно ρ/4.
#    Перша версія пломби його ЗАБУЛА і дала віднос. похибку 1.44 — пломба
#    впала на власній помилці, а не на замиканні. Записано як приклад.
He0 = herm(torch.zeros(1, dtype=torch.float64, device=DEV), KMAX)
for k in range(1, KMAX + 1):
    ser = ser + WK[k - 1] * rho ** (k + 1) * (float(He0[k - 1]) * f0) ** 2
p2 = float((ser - cho).abs().max())
p2r = float(((ser - cho).abs() / cho.abs().clamp(min=1e-12))[rho.abs() < 0.9].max())
print(f'⛔ P2 (АНАЛІТИЧНА, t=0, K={KMAX}): max|ряд − Чо–Саул| = {p2:.3e}, '
      f'віднос. при |ρ|<0.9 = {p2r:.3e}\n', flush=True)

NK = len(KS)
ERR = np.zeros((len(FS), NK))
ERRH = np.zeros((len(FS), NK, 2))
DIAG = np.zeros((len(FS), NK))
TNORM = np.zeros(KMAX)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                        device=DEV, dtype=torch.float64) for q in range(L_)]
    with torch.no_grad():
        SA = torch.zeros(2, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(5500 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            hlf = 0 if d < NS // 2 else 1
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                a = torch.relu(a @ W32[q].float())
            SA[hlf] += a.double().sum(0)
            d += b
        a_true = SA.sum(0) / NS
        a_h = SA * (2.0 / NS)
        for j, K in enumerate(KS):
            m = torch.zeros(W_, dtype=torch.float64, device=DEV)
            C = torch.eye(W_, dtype=torch.float64, device=DEV)
            dn = TNORM if (n == 0 and K == KMAX) else None
            sl = [0.0]
            for q in range(L_):
                w = W32[q]
                cp = w.T @ (C @ w); cp = 0.5 * (cp + cp.T)
                mp = m @ w
                m, C = closure(mp, cp, K, dn, sl)
            DIAG[n, j] = sl[0]
            ERR[n, j] = float(torch.sqrt(((m - a_true) ** 2).sum()
                                         / (a_true ** 2).sum()))
            for h in range(2):
                ERRH[n, j, h] = float(torch.sqrt(((m - a_h[h]) ** 2).sum()
                                                 / (a_h[h] ** 2).sum()))

    print(f'  net{n}  K=1 {ERR[n,0]:.4%} · K=4 {ERR[n,3]:.4%} · '
          f'K=20 {ERR[n,-1]:.4%}', flush=True)

print(f'\n{"=" * 74}\n⛔ ПЛОМБИ')
print(f' P1  K=1 (наш метод)      = {ERR[:,0].mean():.4%}   (lowrk88 дав 0.9299%)')
print(f' P2  аналітична, t=0      = {p2:.3e} абс · {p2r:.3e} віднос ✅' if p2r < 1e-3
      else f' P2  аналітична, t=0      = {p2:.3e} абс · {p2r:.3e} віднос ⛔')
print(f' P4  max |Δdiag| по всіх K = {DIAG.max():.3e}   (мусить бути ~1e-15)')
print(f' P5  розбіжність половин  = {np.abs(ERRH[:,:,0]-ERRH[:,:,1]).mean():.4%}')
print(f'\n P3 ЗБІЖНІСТЬ (норми членів, net0, шар-сума):')
for k in range(min(KMAX, 12)):
    print(f'     k={k+1:2d}  {TNORM[k]:.4e}' + ('  ← наш єдиний член' if k == 0 else ''))

print(f'\n{"=" * 74}\n📈 ПОХИБКА `E[a_last]` ЗАЛЕЖНО ВІД ПОРЯДКУ РЯДУ ({len(FS)} мереж)')
print(f'{"K":>4s} {"похибка":>10s} {"×нашої":>8s} {"дод. F":>11s} {"дод. C/B":>10s}')
b1 = ERR[:, 0].mean()
for j, K in enumerate(KS):
    fl = (K - 1) * 3 * W_ * W_ * L_
    print(f'{K:4d} {ERR[:,j].mean():10.4%} {b1/max(ERR[:,j].mean(),1e-12):8.3f} '
          f'{fl:11.3e} {fl/2.72e11:10.6f}')

print(f'\n{"=" * 74}\n📣 ВИРОК')
best = ERR[:, -1].mean()
print(f' B. K=20 дає {best:.4%} проти нашої {b1:.4%} ⟹ ×{b1/max(best,1e-12):.3f}')
print(f'    ⟹ ОБРІЗАННЯ відповідає за {(1-best/b1)*100:.1f}% похибки,')
print(f'       НЕГАУСОВІСТЬ — за решту {best/b1*100:.1f}%')
print(f' 🎯 пороги: наш рекорд 0.1456% · лідер 0.0760% · L4 0.1318%')
