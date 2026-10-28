"""ЧИСЕЛЬНЕ ІНТЕГРУВАННЯ ЩІЛЬНОСТІ ЗАМІСТЬ КУМУЛЯНТНОГО РОЗКЛАДУ. (С87)

## ЗВІДКИ — ВКАЗІВКА ЮЗЕРА
«Лідер не рахує кумулянтні тензори — радше інтегрує саму щільність чисельно.
Виведи і застосуй.»

## 📐 ВИВЕДЕННЯ (точне; єдине наближення — число факторів `k`)
Крок шару: `g` = передактивація попереднього шару, `a = ReLU(g)`, `p = Wᵀa`.
Наше замикання робить ДВА гаусових припущення: (1) `g` гаусів, (2) `p` гаусів.
Друге зайве — його можна ЗНЯТИ, і саме воно тут знімається.

Нехай `g ~ N(m, C)` і `C = D + VVᵀ`, `D = diag`, `V ∈ R^{n×k}` (top-k власні).
    `D_i = Σ_{r>k} λ_r U_ir² ≥ 0`  ⟹ розклад ЗАВЖДИ коректний, дисперсія ЦІЛА.
Тоді ТОЧНО:  `g = m + Vη + D^{1/2}ξ`,  `η ~ N(0,I_k)`, `ξ ~ N(0,I_n)` незалежні.
🔑 **УМОВНО НА `η` КОМПОНЕНТИ `g` НЕЗАЛЕЖНІ** ⟹ незалежні і `a_j = ReLU(g_j)`.
Кумулянти суми незалежних АДИТИВНІ:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    `κ_r(p_i | η) = Σ_j W_ji^r · κ_r(a_j | η)`        ← МАТВЕК з `W^r`
    `E[ReLU(p_i)] = ∫ E[ReLU(p_i)|η] · N(η;0,I_k) dη` ← КВАДРАТУРА
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔑 ЦЕ ОБХОДИТЬ ТЕНЗОРИ: `κ₃` коштує МАТВЕК `n²`, а не згортку `n³`
([[finding-non-gaussianity-lives-in-the-spectrum-tail]]: `R≈190` = 20.3% бюджету,
`κ₄` = 10.2 ГБ). Тут `κ₃` — це `W³ᵀ·κ₃(a|η)`, тобто 1.3e5 флопів на вузол.

## 📐 1D-МОМЕНТИ `ReLU(N(μ,σ²))` — ВИВЕДЕНО, НЕ ВЗЯТО З ГОЛОВИ
`E[Y^n 1(Y>0)]`, підстановка `y=μ+σu`, `t=μ/σ`:
    `I_k = ∫_{-t}^∞ u^k φ(u)du`,  `I_0=Φ`, `I_1=φ`, `I_k=(k−1)I_{k−2}+(−t)^{k−1}φ`
⟹ `I_2=Φ−tφ`, `I_3=(2+t²)φ`, `I_4=3Φ−3tφ−t³φ`. Біноміальна згортка дає
    `M1 = μΦ + σφ`
    `M2 = (μ²+σ²)Φ + μσφ`
    `M3 = μ(μ²+3σ²)Φ + σ(μ²+2σ²)φ`
    `M4 = (μ⁴+6μ²σ²+3σ⁴)Φ + σμ(μ²+5σ²)φ`
(усі чотири звірені алгебраїчно через `σ²t=σμ`; ПЛОМБА S2 звіряє чисельно.)

## 📐 УМОВНИЙ ЕДЖВОРТ — ТУТ ВІН ЗАКОННИЙ
`G_n(t) = (−1)^(n−2)φ^(n−2)(−t)` ⟹ `G₀=tΦ+φ`, `G₃=−tφ`, `G₄=(t²−1)φ`,
`G₆=(t⁴−6t²+3)φ`;  `E[ReLU] = σ[G₀ + (γ₃/6)G₃ + (γ₄/24)G₄ + (γ₃²/72)G₆]`.
🔑 ЧОМУ ТУТ ЗАКОННИЙ, А РАНІШЕ НІ: умовно на `η` це сума `n` НЕЗАЛЕЖНИХ
доданків ⟹ ЦГТ дає малі `γ` і ряд збігається. Безумовно `γ₃≈0.44` — не мале.

## ⚠️ ЧИМ ЦЕ НЕ Є (щоб не повторити закритий клас)
⛔НЕ кубатура у просторі входу `z` ([[finding-cubature-deterministic-flat]]:
детермінований шлях ПЛОСКИЙ, 1.48× за 256× вузлів). Там вузли жили у R²⁵⁶ і
мали пінити моменти; тут квадратура — у `k`≤3 вимірах ЛАТЕНТНОГО фактора, а
розподіл усередині береться АНАЛІТИЧНО.
⛔НЕ низькорангове наближення НЕГАУСОВОСТІ ([[finding-non-gaussianity-lives-in-the-spectrum-tail]]:
ранг оманливий). Там рангом `R` наближали тензор `κ₃`. Тут рангом `k` наближають
КОВАРІАЦІЮ ГАУСОВОГО `g` — а вона справді низькорангова (еф. ранг 3.2), і вся
негаусовість `p` народжується з ReLU ТОЧНО, а не апроксимується.

## 🎯 ЩО МІРЯЄМО (однокроковий тест, обидва методи мають ТОЧНИЙ вхід)
| метод | що знає | що припускає |
|---|---|---|
| `GAUS` | ТОЧНІ `m_p`,`C_p` з МК | `p` гаусів |
| `FAC-k` | ТОЧНІ `m_g`,`C_g` з МК | `g` гаусів, `C_g` = `D+VVᵀ` |
| `FAC-k+E` | те саме | + умовний Еджворт 4 порядку |
🔑 `FAC` має СЛАБШУ інформацію (2 моменти `g`, а не `p`) — якщо він усе одно
виграє, це не артефакт.

⛔ПЛОМБИ:
 S1 `k = n` ⟹ `D = 0`, залишок `‖C − VVᵀ‖` = машинний нуль.
 S2 1D-моменти проти `scipy.integrate.quad` — машинна точність.
 S3 РЕЖИМ ОБ'ЄКТА: синтетика з СПРАВЖНІМ рангом 1 (`C = vvᵀ + D`) ⟹ `FAC-1`
    мусить дати похибку на рівні квадратурної (≪ `GAUS`). Якщо не дає — баг.
 S4 МАСШТАБУВАННЯ: `N×4` — істина стоїть, шум падає вдвічі.
 S5 ⛔КОНТРОЛЬ: `V` від ЧУЖОЇ мережі ⟹ `FAC` мусить стати ГІРШИМ за `GAUS`.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (щоб не тлумачити заднім числом):
 (а) S1,S2 пройдуть — це алгебра. Шанс 97%.
 (б) `FAC-1` побиває `GAUS` на ГЛИБОКИХ шарах (еф. ранг 3–5), і НЕ побиває на
     ранніх (еф. ранг 127). Шанс, що на шарі 31 похибка впаде ≥2×: **55%**.
 (в) Умовний Еджворт додасть мало (ЦГТ уже зробив `γ|η` малими). Шанс, що
     `+E` дає ще ≥1.3× поверх `FAC`: 30%.
 (г) ГОЛОВНИЙ РИЗИК: `FAC` може програти `GAUS` НАВІТЬ на глибині, бо `GAUS`
     годують ТОЧНОЮ `C_p`, у якій уже сидить частина негаусовості `g`. Шанс
     цього: 35%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/onefac.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
BATCH = 1 << 12
L_, W_ = 32, 256
KS = (1, 2, 3)
NQ = {1: 24, 2: 16, 3: 10}
LAYERS = [2, 4, 8, 16, 24, 31]
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

SQ2PI = math.sqrt(2.0 * math.pi)
INV_SQ2PI = 1.0 / SQ2PI


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def relu_moments(mu, sg):
    """M1..M4 для `ReLU(N(mu,sg²))` — виведення в докстрінгу."""
    t = mu / sg
    P, f = Phi(t), phi(t)
    s2 = sg * sg
    m1 = mu * P + sg * f
    m2 = (mu * mu + s2) * P + mu * sg * f
    m3 = mu * (mu * mu + 3.0 * s2) * P + sg * (mu * mu + 2.0 * s2) * f
    m4 = ((mu ** 4 + 6.0 * mu * mu * s2 + 3.0 * s2 * s2) * P
          + sg * mu * (mu * mu + 5.0 * s2) * f)
    return m1, m2, m3, m4


def gaus_relu(m, s):
    t = m / s
    return m * Phi(t) + s * phi(t)


def edge_relu(m, s, k3, k4):
    """Еджворт 4 порядку через `G_n(t) = (−1)^(n−2)φ^(n−2)(−t)`."""
    t = m / s
    P, f = Phi(t), phi(t)
    g3 = k3 / (s ** 3)
    g4 = k4 / (s ** 4)
    G0 = t * P + f
    G3 = -t * f
    G4 = (t * t - 1.0) * f
    G6 = (t ** 4 - 6.0 * t * t + 3.0) * f
    return s * (G0 + (g3 / 6.0) * G3 + (g4 / 24.0) * G4 + (g3 * g3 / 72.0) * G6)


def herm_nodes(nq, k, dev):
    """Тензорний Гаусс-Ерміт для ваги `N(0,I_k)`."""
    x, w = np.polynomial.hermite_e.hermegauss(nq)
    w = w / w.sum()
    if k == 1:
        return (torch.tensor(x[:, None], dtype=torch.float64, device=dev),
                torch.tensor(w, dtype=torch.float64, device=dev))
    grids = np.meshgrid(*([x] * k), indexing='ij')
    wg = w
    for _ in range(k - 1):
        wg = np.multiply.outer(wg, w)
    pts = np.stack([q.ravel() for q in grids], axis=1)
    return (torch.tensor(pts, dtype=torch.float64, device=dev),
            torch.tensor(wg.ravel(), dtype=torch.float64, device=dev))


def gaus_closure_step(m_g, C_g, W):
    """РІВНО те, що робить `_closure_parts` сабміта: Мелер K=2 + точна діагональ.

    Це ЧЕСНА БАЗА: той самий вхід `(m_g, C_g)`, що й у факторної моделі."""
    s = torch.sqrt(torch.clamp(torch.diagonal(C_g), min=1e-30))
    t = m_g / s
    P, f = Phi(t), phi(t)
    m_a = m_g * P + s * f
    C_a = C_g * torch.outer(P, P) + (0.5 * (C_g * C_g)) * torch.outer(f / s, f / s)
    d = (m_g * m_g + s * s) * P + (m_g * s) * f - m_a * m_a
    C_a = C_a + torch.diag(d - torch.diagonal(C_a))
    m_p = m_a @ W
    v_p = torch.clamp(((C_a @ W) * W).sum(0), min=1e-30)
    return m_p, torch.sqrt(v_p)


def factor_cumulants(m_g, C_g, W, k, nq, V_alt=None):
    """ПОВНІ `(m, σ, κ₃, κ₄)` для `p_i` із суміші по `η` — закон повних моментів.

    Умовно на `η` компоненти незалежні ⟹ `κ_r(p|η) = Σ_j W_ji^r κ_r(a_j|η)`.
    Далі МОМЕНТИ (не кумулянти) усереднюються по `η` ЛІНІЙНО:
        `E[p^r] = ∫ E[p^r|η] dη`,   `E[p³|η] = m³+3mv+κ₃`,
        `E[p⁴|η] = m⁴+6m²v+4mκ₃+κ₄+3v²`
    і вже з повних моментів беруться повні кумулянти."""
    ev, U = torch.linalg.eigh(C_g)
    idx = torch.argsort(ev, descending=True)
    ev, U = torch.clamp(ev[idx], min=0.0), U[:, idx]
    V = U[:, :k] * torch.sqrt(ev[:k])
    if V_alt is not None:
        V = V_alt                                     # S5: чужий фактор
    D = torch.clamp(torch.diagonal(C_g) - (V * V).sum(1), min=1e-12)
    sg = torch.sqrt(D)
    pts, wq = herm_nodes(nq, k, C_g.device)
    mu = m_g[None, :] + pts @ V.T                     # (Q,n)
    m1, m2, m3, m4 = relu_moments(mu, sg[None, :])
    c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
    c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4 - 3.0 * c2 * c2
    W2 = W * W
    mq = m1 @ W                                       # (Q,n) умовне середнє
    vq = torch.clamp(c2 @ W2, min=1e-30)
    k3q = c3 @ (W2 * W)
    k4q = c4 @ (W2 * W2)
    w = wq[:, None]
    M1 = (w * mq).sum(0)
    M2 = (w * (mq * mq + vq)).sum(0)
    M3 = (w * (mq ** 3 + 3.0 * mq * vq + k3q)).sum(0)
    M4 = (w * (mq ** 4 + 6.0 * mq * mq * vq + 4.0 * mq * k3q
               + k4q + 3.0 * vq * vq)).sum(0)
    K2 = torch.clamp(M2 - M1 * M1, min=1e-30)
    K3 = M3 - 3.0 * M1 * M2 + 2.0 * M1 ** 3
    K4 = M4 - 4.0 * M1 * M3 + 6.0 * M1 * M1 * M2 - 3.0 * M1 ** 4 - 3.0 * K2 * K2
    # ПРЯМА квадратура (умовно-гаусова, без Еджворта) — окремий вихід
    direct = (w * gaus_relu(mq, torch.sqrt(vq))).sum(0)
    return M1, torch.sqrt(K2), K3, K4, direct


def factor_predict(m_g, C_g, W, k, nq, use_edge, V_alt=None):
    M1, S, K3, K4, direct = factor_cumulants(m_g, C_g, W, k, nq, V_alt)
    return direct if not use_edge else edge_relu(M1, S, K3, K4)


# ══════════════════ ПЛОМБА S2: 1D-моменти проти quad ══════════════════
try:
    from scipy.integrate import quad
    worst = 0.0
    for mu0, sg0 in ((0.0, 1.0), (0.7, 1.0), (-1.3, 0.6), (2.5, 2.0), (-0.2, 0.3)):
        mm = relu_moments(torch.tensor([mu0], dtype=torch.float64),
                          torch.tensor([sg0], dtype=torch.float64))
        for nn in range(1, 5):
            ref = quad(lambda y, n=nn: max(y, 0.0) ** n
                       * math.exp(-0.5 * ((y - mu0) / sg0) ** 2)
                       / (sg0 * SQ2PI), -40 * sg0 + mu0, 40 * sg0 + mu0,
                       limit=400)[0]
            worst = max(worst, abs(float(mm[nn - 1][0]) - ref) / max(abs(ref), 1e-12))
    print(f'⛔S2 1D-моменти vs quad: найгірша віднос.розб = {worst:.3e}', flush=True)
except Exception as e:                                # noqa: BLE001
    print(f'⛔S2 пропущено: {e}', flush=True)

# ══════════════════ ПЛОМБА S3: РЕЖИМ ОБ'ЄКТА (справжній ранг 1) ══════════════
gS = torch.Generator(device=DEV)
gS.manual_seed(4242)
n_s = 64
DT = torch.float64
v_s = torch.randn(n_s, generator=gS, device=DEV, dtype=DT) * 0.8
d_s = torch.rand(n_s, generator=gS, device=DEV, dtype=DT) * 0.5 + 0.1
C_s = torch.outer(v_s, v_s) + torch.diag(d_s)
m_s = torch.randn(n_s, generator=gS, device=DEV, dtype=DT) * 0.3
W_s = torch.randn(n_s, n_s, generator=gS, device=DEV, dtype=DT) * math.sqrt(2.0 / n_s)
Ls = torch.linalg.cholesky(C_s)
acc = torch.zeros(n_s, device=DEV, dtype=DT)
accp = torch.zeros(n_s, device=DEV, dtype=DT)
accp2 = torch.zeros(n_s, device=DEV, dtype=DT)
NSS = 1 << 23
done = 0
while done < NSS:
    b = min(1 << 16, NSS - done)
    gg = m_s + torch.randn(b, n_s, generator=gS, device=DEV, dtype=DT) @ Ls.T
    pp = torch.relu(gg) @ W_s
    acc += torch.relu(pp).sum(0)
    accp += pp.sum(0)
    accp2 += (pp * pp).sum(0)
    done += b
tru_s = acc / NSS
mp_s = accp / NSS
sp_s = torch.sqrt(torch.clamp(accp2 / NSS - mp_s * mp_s, min=1e-30))
gau_s = gaus_relu(mp_s, sp_s)
fac_s = factor_predict(m_s, C_s, W_s, 1, 24, False)
nrm = float(torch.sqrt((tru_s ** 2).mean()))
e_g = float(torch.sqrt(((gau_s - tru_s) ** 2).mean())) / nrm
e_f = float(torch.sqrt(((fac_s - tru_s) ** 2).mean())) / nrm
print(f'⛔S3 режим-об\'єкта (СПРАВЖНІЙ ранг 1, n=64): GAUS {e_g:.4%} · '
      f'FAC-1 {e_f:.4%} · відношення {e_g / max(e_f, 1e-12):.2f}×', flush=True)

# ══════════════════ ОСНОВНИЙ ВИМІР ══════════════════
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'\n{len(FS)} мереж · 2^{LOG2N} семплів · k={KS} · NQ={NQ} · {DEV}\n',
      flush=True)

SH = len(FS), len(LAYERS)
E_G = np.zeros(SH)          # оракульні m_p,C_p + «p гаусів»  (нижня межа GAUS)
E_CL = np.zeros(SH)         # ЧЕСНА БАЗА: замикання сабміта з (m_g,C_g)
E_F = {(k, e): np.zeros(SH) for k in KS for e in (0, 1)}
E_SO = {k: np.zeros(SH) for k in KS}   # форма від FAC, масштаб ОРАКУЛЬНИЙ
E_SC = {k: np.zeros(SH) for k in KS}   # форма від FAC, масштаб від ЗАМИКАННЯ
D_MP = {k: np.zeros(SH) for k in KS}   # віднос.похибка m_p у FAC
D_SP = {k: np.zeros(SH) for k in KS}   # віднос.похибка s_p у FAC
D_MPC = np.zeros(SH)
D_SPC = np.zeros(SH)
G3T = np.zeros(SH)
R_CL = np.zeros(SH)                    # оптимальний СКАЛЯРНИЙ масштаб бази
R_SC = {k: np.zeros(SH) for k in KS}   # ... і в SC-варіанта
E_ALT = np.zeros(SH)
SEAL1 = np.zeros(len(FS))
ERANK = np.zeros(SH)
HALF = np.zeros(SH)

V_REF = {}
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        SG = {L: torch.zeros(W_, dtype=torch.float64, device=DEV) for L in LAYERS}
        CG = {L: torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for L in LAYERS}
        SP = {L: torch.zeros(W_, dtype=torch.float64, device=DEV) for L in LAYERS}
        CP = {L: torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for L in LAYERS}
        TR = {L: torch.zeros(W_, dtype=torch.float64, device=DEV) for L in LAYERS}
        TRH = {L: torch.zeros(W_, dtype=torch.float64, device=DEV) for L in LAYERS}
        g = torch.Generator(device=DEV)
        g.manual_seed(7700 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            prev = None
            for q in range(L_):
                h = a @ W[q]
                if q in LAYERS and prev is not None:
                    SG[q] += prev.sum(0, dtype=torch.float64)
                    CG[q] += (prev.T @ prev).double()
                    SP[q] += h.sum(0, dtype=torch.float64)
                    CP[q] += (h.T @ h).double()
                    rl = torch.relu(h).sum(0, dtype=torch.float64)
                    TR[q] += rl
                    if d < NS // 2:
                        TRH[q] += rl
                prev = h
                a = torch.relu(h)
            d += b

    for j, L in enumerate(LAYERS):
        m_g = SG[L] / NS
        C_g = CG[L] / NS - torch.outer(m_g, m_g)
        m_p = SP[L] / NS
        C_p = CP[L] / NS - torch.outer(m_p, m_p)
        tru = TR[L] / NS
        nrm = float(torch.sqrt((tru ** 2).mean()))
        HALF[n, j] = float(torch.sqrt((((TRH[L] / (NS // 2)) - tru) ** 2).mean())) / nrm
        ev = torch.linalg.eigvalsh(C_g)
        ev = torch.clamp(ev, min=0)
        ERANK[n, j] = float(ev.sum() ** 2 / (ev * ev).sum())
        s_p = torch.sqrt(torch.clamp(torch.diagonal(C_p), min=1e-30))
        gau = gaus_relu(m_p, s_p)
        E_G[n, j] = float(torch.sqrt(((gau - tru) ** 2).mean())) / nrm
        Wd = W[L].double()

        def rel(a, b):
            return float(torch.sqrt(((a - b) ** 2).mean())
                         / torch.sqrt((b * b).mean()))

        def rat(a):
            return float((a * tru).sum() / (tru * tru).sum())

        mc, sc = gaus_closure_step(m_g, C_g, Wd)
        cl = gaus_relu(mc, sc)
        E_CL[n, j] = float(torch.sqrt(((cl - tru) ** 2).mean())) / nrm
        R_CL[n, j] = rat(cl)
        D_MPC[n, j], D_SPC[n, j] = rel(mc, m_p), rel(sc, s_p)
        for k in KS:
            M1, S, K3, K4, direct = factor_cumulants(m_g, C_g, Wd, k, NQ[k])
            E_F[(k, 0)][n, j] = float(torch.sqrt(((direct - tru) ** 2).mean())) / nrm
            ed = edge_relu(M1, S, K3, K4)
            E_F[(k, 1)][n, j] = float(torch.sqrt(((ed - tru) ** 2).mean())) / nrm
            D_MP[k][n, j], D_SP[k][n, j] = rel(M1, m_p), rel(S, s_p)
            # 🔑 ФОРМА від FAC (`γ₃`,`γ₄`), МАСШТАБ звідки-інде
            g3 = K3 / (S ** 3)
            g4 = K4 / (S ** 4)
            so = edge_relu(m_p, s_p, g3 * s_p ** 3, g4 * s_p ** 4)
            sc2 = edge_relu(mc, sc, g3 * sc ** 3, g4 * sc ** 4)
            E_SO[k][n, j] = float(torch.sqrt(((so - tru) ** 2).mean())) / nrm
            E_SC[k][n, j] = float(torch.sqrt(((sc2 - tru) ** 2).mean())) / nrm
            R_SC[k][n, j] = rat(sc2)
        # ⛔S5 КОНТРОЛЬ: фактор `V` від ЧУЖОЇ мережі при своїх `m_g`,`D`
        evx, Ux = torch.linalg.eigh(C_g)
        ix = torch.argsort(evx, descending=True)
        Vown = (Ux[:, ix] * torch.sqrt(torch.clamp(evx[ix], min=0.0)))[:, :1]
        if L not in V_REF:
            V_REF[L] = Vown
        else:
            pr = factor_predict(m_g, C_g, Wd, 1, NQ[1], False, V_alt=V_REF[L])
            E_ALT[n, j] = float(torch.sqrt(((pr - tru) ** 2).mean())) / nrm

    # ⛔S1: k = n ⟹ D = 0, залишок машинний
    C0 = CG[LAYERS[-1]] / NS - torch.outer(SG[LAYERS[-1]] / NS, SG[LAYERS[-1]] / NS)
    ev, U = torch.linalg.eigh(C0)
    idx = torch.argsort(ev, descending=True)
    ev, U = torch.clamp(ev[idx], min=0.0), U[:, idx]
    Vf = U * torch.sqrt(ev)
    SEAL1[n] = float(torch.linalg.norm(C0 - Vf @ Vf.T) / torch.linalg.norm(C0))
    print(f'  net{n} (seed {sd0}) готова · S1 залишок {SEAL1[n]:.3e}', flush=True)

print(f'\n⛔S1 `k=n` ⟹ ‖C−VVᵀ‖/‖C‖ = {SEAL1.mean():.3e} (машинний нуль)')
print(f'⛔S4 масштаб: піврозмір-МК проти повного = {HALF.mean():.4%} '
      f'(шум істини; похибки нижче мусять бути ВИЩИМИ)\n')

print('══ ПОХИБКА `E[ReLU(p)]`, віднос. RMS. `CL` = ЧЕСНА БАЗА (замикання сабміта) ══')
hdr = ('шар ефранг   GAUS*      CL   ' + '  '.join(
    f' FAC{k}{"E" if e else " "}' for k in KS for e in (0, 1))
    + '  ' + '  '.join(f'  SO{k}' for k in KS)
    + '  ' + '  '.join(f'  SC{k}' for k in KS))
print(hdr)
for j, L in enumerate(LAYERS):
    row = (f'{L:3d} {ERANK[:, j].mean():6.1f}  {E_G[:, j].mean():6.4%}  '
           f'{E_CL[:, j].mean():6.4%}')
    for k in KS:
        for e in (0, 1):
            row += f'  {E_F[(k, e)][:, j].mean():6.4%}'
    for k in KS:
        row += f'  {E_SO[k][:, j].mean():6.4%}'
    for k in KS:
        row += f'  {E_SC[k][:, j].mean():6.4%}'
    print(row)
print('GAUS* = оракульні m_p,C_p (недосяжна нижня межа) · SOk = форма FAC-k + '
      'оракульний масштаб · SCk = форма FAC-k + масштаб замикання')

print('\n══ ДІАГНОСТИКА: віднос. похибка МАСШТАБУ (m_p, s_p) ══')
print('шар   CL:m_p   CL:s_p  ' + '  '.join(f'F{k}:m_p  F{k}:s_p' for k in KS))
for j, L in enumerate(LAYERS):
    row = f'{L:3d}  {D_MPC[:, j].mean():6.3%}  {D_SPC[:, j].mean():6.3%}'
    for k in KS:
        row += f'  {D_MP[k][:, j].mean():6.3%}  {D_SP[k][:, j].mean():6.3%}'
    print(row)

print('\n📊 ВИГРАШ проти ЧЕСНОЇ БАЗИ `CL` (× по MSE):')
for j, L in enumerate(LAYERS):
    row = f'{L:3d} {ERANK[:, j].mean():6.1f}'
    for k in KS:
        for tag, arr in (('F', E_F[(k, 1)]), ('SC', E_SC[k])):
            r = (E_CL[:, j] / np.maximum(arr[:, j], 1e-30)) ** 2
            row += f'  {tag}{k}={r.mean():7.3f}'
    print(row)


def loo_resid(err, rat_):
    """Похибка після зняття СПІЛЬНОГО скалярного масштабу, LOO по мережах.

    🔑 Це вирішальний фільтр: `λ`-регресія сабміта поглинає будь-який спільний
    скаляр АВТОМАТИЧНО ([[finding-lambda-absorbs-any-mechanism-you-fix]]) ⟹
    виграш, що зникає після цієї операції, у сабміті дає РІВНО НУЛЬ."""
    out = np.zeros_like(err)
    for i in range(len(err)):
        msk = np.arange(len(err)) != i
        out[i] = math.sqrt(max(err[i] ** 2 - (rat_[i] - 1.0) ** 2
                               + (rat_[i] - rat_[msk].mean()) ** 2, 1e-30))
    return out


print('\n🔑 ВИРІШАЛЬНЕ: ЧИ ВИГРАШ SC — ЦЕ ПРОСТО СКАЛЯРНИЙ МАСШТАБ?')
print('(λ-регресія і `_ALPHA` знімають спільний скаляр самі ⟹ те, що зникає '
      'тут, у сабміті дає нуль)')
print('шар   CL сирий  CL−скаляр  ' + '  '.join(
    f'SC{k} сир  SC{k}−ск  ×чисте' for k in KS))
for j, L in enumerate(LAYERS):
    rb = loo_resid(E_CL[:, j], R_CL[:, j])
    row = f'{L:3d}  {E_CL[:, j].mean():7.4%}  {rb.mean():7.4%}  '
    for k in KS:
        rk = loo_resid(E_SC[k][:, j], R_SC[k][:, j])
        row += (f'  {E_SC[k][:, j].mean():6.4%}  {rk.mean():6.4%}  '
                f'×{((rb / np.maximum(rk, 1e-30)) ** 2).mean():5.3f}')
    print(row)

print('\n⛔S5 КОНТРОЛЬ (фактор `V` від ЧУЖОЇ мережі, k=1) — мусить бути ГІРШЕ GAUS:')
print('шар    свій FAC1     чужий V       GAUS')
for j, L in enumerate(LAYERS):
    alt = E_ALT[1:, j]
    print(f'{L:3d}  {E_F[(1, 0)][1:, j].mean():9.4%}  {alt.mean():9.4%}  '
          f'{E_G[1:, j].mean():9.4%}')

print('\n📊 ВИГРАШ проти GAUS (× по MSE, тобто (e_G/e_F)²):')
print(hdr.replace('GAUS  ', '      '))
for j, L in enumerate(LAYERS):
    row = f'{L:3d}  {ERANK[:, j].mean():6.1f}        '
    for k in KS:
        for e in (0, 1):
            r = (E_G[:, j] / np.maximum(E_F[(k, e)][:, j], 1e-30)) ** 2
            row += f'  {r.mean():7.3f}'
    print(row)
