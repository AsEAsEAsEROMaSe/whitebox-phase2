"""МІШАНІ КУМУЛЯНТИ `κ₂₁, κ₁₂` — КОПУЛА В ПЕРШОМУ ПОРЯДКУ (С88)

## ЗВІДКИ — ЗАУВАЖЕННЯ ЮЗЕРА
«можливо в коді є помилки або не всі мат можливості використані».
Перевірка знайшла три помилки (`chk88`), і одна математична можливість
справді пропущена: `cum88` використав ЛИШЕ МАРГІНАЛЬНІ `κ₃₀, κ₀₃`, а
**мішані `κ₂₁, κ₁₂` — ні**. Саме вони і є копулою в першому порядку:
маргіналі їх не бачать ЗА ПОБУДОВОЮ.

## 📐 ВИВІД ПОПРАВКИ
Розклад по Ерміту (`herm88`): `E[a_j a_k]/(s_j s_k) = Σ_{n,m} c_n c_m E[He_n(u_j)He_m(u_k)]`.
Для ГАУСОВОЇ пари `E[He_n He_m] = δ_{nm}·n!·ρ^n` — це наш ряд Прайса.
Для негаусової з'являються НЕДІАГОНАЛЬНІ члени, і найнижчі з них:
```
E[He_1(u_j)He_2(u_k)] = κ₁₂ ,  E[He_2(u_j)He_1(u_k)] = κ₂₁
```
З `c_1 = Φ(t)`, `c_2 = φ(t)/2` (аналітичні, `price88`):
```
🔑 δCov_jk = s_j s_k · [ Φ(t_j)·(φ(t_k)/2)·κ₁₂ + (φ(t_j)/2)·Φ(t_k)·κ₂₁ ]
```
## 📐 ЯК ДІСТАТИ `κ₂₁` — ОДИН МАТМУЛ
`h = Wᵀa`. За незалежності `a_l` (наближення, як у `cum88`):
```
κ₂₁(h_j,h_k) = Σ_l W_lj²·W_lk·κ₃(a_l)     ⟹  M = (W∘W∘κ₃)ᵀ · W
```
Матмул 256³ = 33.5e6 флопів на шар ⟹ 1.07e9 на 32 шари = **подвоєння
ланцюга**. Наш `C/B` = 0.00395 при підлозі 0.1 ⟹ **запас ×25, вміщається**.
Стандартизація: `κ₂₁(u_j,u_k) = M_jk/(s_j²·s_k)`, `κ₁₂ = M_kj/(s_k²·s_j)`.

## ⛔ ПЛОМБИ
 P1 🔑ШАР 1: вхід ТОЧНО гаусів ⟹ істинні `κ₂₁ ≡ 0`. Міряємо і друкуємо.
 P2 🔑СИМЕТРІЯ: `κ₂₁(j,k) = κ₁₂(k,j)` тотожно — перевірка індексації.
 P3 ЯКІСТЬ наближення незалежності: істинний `κ₂₁` (з МК) проти `Σ W²Wκ₃`.
    Корел і віднос. похибка по шарах. ⚠️Вимірювати `κ₂₁` шумно — ставимо
    пломбу масштабування (N проти 4N), як у `cum88`.
 P4 ЛАНЦЮГ: база мусить дати 0.9404% (`cum88`, 24 мережі).
 P5 ОРАКУЛЬНИЙ `κ₂₁` (з МК) — стеля гілки.

## 📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ
 A. Оракульні мішані `κ` дадуть **×1.3–2.5** на замиканні. Шанс **40%**.
    Підстава: маргінальні дали ×1.336, а копула зміряна як ГОЛОВНЕ джерело
    (`herm88`: гаусова копула + точні маргіналі ГІРША за повністю гаусову).
 B. Наближення незалежності з'їсть половину. Шанс **60%**.
 C. 🎯У балі (за ВИПРАВЛЕНОЮ константою `1/r=10.33`): `k=2` → ×1.088.
    Сабміта це не дає, але вперше зачепить копулу прицільно.
 D. ⚠️Якщо оракульні дадуть &lt;×1.15 — копула НЕ описується третім порядком,
    і весь моментний підхід закритий остаточно. Шанс **35%**.
"""
import glob, math, sys
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
PROBE = (0, 8, 16, 24, 31)
assert DEV not in ('cuda:0', 'cuda:4')
print(__doc__.split('📣')[1], flush=True)

I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
NF = len(FS)
print(f'{NF} мереж · 2^{LOG2N} · {DEV}\n', flush=True)


def relu_k3(m, s):
    """`κ₃` активації `ReLU(N(m,s²))` — замкнена формула."""
    t = m / s; P, f = Phi(t), phi(t); s2 = s * s
    M1 = m * P + s * f
    M2 = (m * m + s2) * P + m * s * f
    M3 = (m ** 3 + 3 * m * s2) * P + (m * m + 2 * s2) * s * f
    return M3 - 3 * M1 * M2 + 2 * M1 ** 3


def step(m, C, K21=None):
    """Замикання + (опційно) поправка від МІШАНИХ кумулянтів."""
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s; P, f = Phi(t), phi(t)
    m_a = m * P + s * f
    Ca = C * torch.outer(P, P) + 0.5 * (C * C) * torch.outer(f / s, f / s)
    if K21 is not None:
        # κ₂₁ стандартизований: M_jk / (s_j² s_k)
        k21 = K21 / (torch.outer(s * s, s))
        corr = torch.outer(P, f * 0.5) * k21.T + torch.outer(f * 0.5, P) * k21
        Ca = Ca + corr * torch.outer(s, s)
    d = (m * m + s * s) * P + (m * s) * f - m_a * m_a
    Ca = Ca + torch.diag(d - torch.diagonal(Ca))
    return m_a, Ca, s, t


MODES = ('база', 'мішані κ (діаг)', 'мішані κ (ОРАКУЛ)')
ERR = np.zeros((NF, 3))
P1S = np.zeros(NF); P2S = np.zeros(NF)
P3REL = np.zeros((NF, len(PROBE))); P3COR = np.zeros((NF, len(PROBE)))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                        device=DEV, dtype=torch.float64) for q in range(L_)]
    Wf = [w.float() for w in W32]
    with torch.no_grad():
        H1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        H2 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        M21 = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        A1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(4200 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                h = (a @ Wf[q]).double()
                H1[q] += h.sum(0); H2[q] += (h * h).sum(0)
                M21[q] += (h * h).T @ h                 # E[h_j² h_k]·NS
                aq = torch.clamp(h, min=0.0)
                A1[q] += aq.sum(0)
                a = aq.float()
            d += b
        MH = H1 / NS
        SH = torch.sqrt(torch.clamp(H2 / NS - MH * MH, min=1e-30))
        AM = A1 / NS
        # центрований мішаний момент: E[(h_j−m_j)²(h_k−m_k)]
        K21T = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(4200 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                h = (a @ Wf[q]).double()
                u = h - MH[q]
                K21T[q] += (u * u).T @ u
                a = torch.clamp(h, min=0.0).float()
            d += b
        K21T /= NS
        P1S[n] = float((K21T[0] / torch.outer(SH[0] ** 2, SH[0])).abs().mean())
        # ⛔P2: κ₂₁(j,k) проти κ₁₂(k,j) — має бути транспонування того самого
        P2S[n] = float((K21T[16] - K21T[16].T).abs().max()
                       / K21T[16].abs().max().clamp(min=1e-30))
        for j, q in enumerate(PROBE):
            if q == 0:
                P3REL[n, j] = float('nan'); P3COR[n, j] = float('nan'); continue
            k3a = relu_k3(MH[q - 1], SH[q - 1])
            appr = ((W32[q] * W32[q]) * k3a.view(-1, 1)).T @ W32[q]
            tru = K21T[q]
            P3REL[n, j] = float(torch.sqrt(((appr - tru) ** 2).sum()
                                           / (tru * tru).sum()))
            P3COR[n, j] = float(torch.corrcoef(torch.stack(
                [appr.flatten(), tru.flatten()]))[0, 1])
        a_true = AM[L_ - 1]
        for mi in range(3):
            m = torch.zeros(W_, dtype=torch.float64, device=DEV)
            C = torch.eye(W_, dtype=torch.float64, device=DEV)
            for q in range(L_):
                cp = W32[q].T @ (C @ W32[q]); cp = 0.5 * (cp + cp.T)
                mp = m @ W32[q]
                if mi == 0:
                    K = None
                elif mi == 1:
                    if q == 0:
                        K = torch.zeros(W_, W_, dtype=torch.float64, device=DEV)
                    else:
                        sprev = torch.sqrt(torch.clamp(torch.diagonal(cprev),
                                                       min=1e-30))
                        k3a = relu_k3(mprev, sprev)
                        K = ((W32[q] * W32[q]) * k3a.view(-1, 1)).T @ W32[q]
                else:
                    K = K21T[q]
                cprev, mprev = cp, mp
                m, C, s, t = step(mp, cp, K)
            ERR[n, mi] = float(torch.sqrt(((m - a_true) ** 2).sum()
                                          / (a_true ** 2).sum()))
        del M21, K21T
    print(f'  net{n}  база {ERR[n,0]:.4%} · діаг {ERR[n,1]:.4%} · '
          f'оракул {ERR[n,2]:.4%}', flush=True)

print(f'\n{"=" * 72}\n⛔ ПЛОМБИ')
print(f' P1  |κ₂₁| на шарі 1 = {P1S.mean():.5f}  (вхід ТОЧНО гаусів ⟹ ~0)')
print(f' P2  асиметрія κ₂₁ = {P2S.mean():.4f}  (мусить бути ≠0: це НЕ симетр.)')
print(f' P4  база = {ERR[:,0].mean():.4%}  (cum88 дав 0.9404%)')
print(f'\n📈 P3: ЯКІСТЬ `κ₂₁ ≈ Σ_l W_lj²W_lk κ₃(a_l)`')
print(f'{"шар":>5s} {"віднос.похиб":>13s} {"корел":>8s}')
for j, q in enumerate(PROBE):
    if q == 0: continue
    print(f'{q+1:5d} {P3REL[:,j].mean():13.2%} {P3COR[:,j].mean():8.4f}')

print(f'\n{"=" * 72}\n📈 ЛАНЦЮГ ({NF} мереж)')
b = ERR[:, 0].mean()
for mi, md in enumerate(MODES):
    k = b / max(ERR[:, mi].mean(), 1e-12)
    print(f'{md:>20s} {ERR[:,mi].mean():10.4%}  ×{k:.3f}   '
          f'у балі ×{(k / 10.33 + 1) / (1 / 10.33 + 1):.4f}')
print(f'\n📣 ВИРОК: оракульні мішані κ дають '
      f'×{b/max(ERR[:,2].mean(),1e-12):.3f} (передбачав ×1.3–2.5)')
