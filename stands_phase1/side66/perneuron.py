"""ПОНЕЙРОННА СХЕМА: ЧИ ВИСТАЧАЄ 4 МОМЕНТІВ НА НЕЙРОН. (С86)

## ЗВІДКИ — ВКАЗІВКА ЮЗЕРА
«Перевір варіант понейронного. Поки на бюджет не дивись, спробуй дізнатись
суть, що є у такій понейронній версії.»

## ЩО ЦЕ ЗА СХЕМА
Реконструкція методу лідера: він НЕ веде матрицю `C` (2.15e9 флопів проти його
1.3e7) і НЕ веде пари (2.1e6 пар × 32 шари не влазять у 2.65 с). Лишається
єдине — ПОНЕЙРОННІ маргінальні характеристики, вектори по 256.

## 📐 ФОРМУЛА, ЯКУ ПЕРЕВІРЯЄМО (виведена С86)
Через `E[f(Z)He_n(Z)] = E[f⁽ⁿ⁾(Z)]` і `ReLU⁽ⁿ⁾ = δ⁽ⁿ⁻²⁾`:
    `G_n(t) = E[ReLU(t+Z)He_n(Z)] = (−1)^(n−2)·φ^(n−2)(−t)`
    G_0 = tΦ(t)+φ(t) ·  G_3 = −tφ(t) ·  G_4 = (t²−1)φ(t) ·  G_6 = (t⁴−6t²+3)φ(t)
Еджворт по стандартизованій `h`:
    `E[ReLU(h)] = σ·[G_0 + (γ₃/6)G_3 + (γ₄/24)G_4 + (γ₃²/72)G_6]`
🔑 ПРИ `t→0`: `G_3 → 0`, тобто **`κ₃` входить із множником `t`, а `κ₄` — без**.
   `E → σφ(0)·[1 − γ₄/24 + γ₃²/24]` ⟹ якщо `t` малий, `κ₃` СТРУКТУРНО пригнічений.
   Це пояснювало б, чому `κ₃` у нас помер з трьох боків (С86).

## 🎯 ТРИ НЕЗАЛЕЖНІ ВИМІРИ
A. ЛОКАЛЬНА ТОЧНІСТЬ (оракульна подача): беремо ІСТИННІ понейронні моменти `h`
   з МК і дивимось, наскільки точно кожна схема дає `E[ReLU(h_i)]`.
   Схеми: G(гаус) · G+κ₃ · G+κ₄ · G+κ₃+κ₄ · +γ₃² (повний Еджворт-4).
   ⟹ відповідає на «чи ВИСТАЧАЄ 4 моментів на нейрон».
B. ПЕРЕНОС МОМЕНТІВ (рекурсія): чи відтворює діагональна рекурсія
   `σ²(h_i) = Σ_j W_ji²·var(a_j)` і `κ₄(h_i) = Σ_j W_ji⁴·κ₄(a_j)` істину.
   ⟹ це аналог гейту 1 для `κ₃`, де діагональ дала лише 7.8%.
C. СТАТИСТИКА `t = m/σ` — наскільки він справді малий.

⛔ПЛОМБИ:
 (1) 🔑ШАР 1: `h₁ = zᵀW₀` з `z~N(0,I)` ⟹ ТОЧНО гаусів. Гаусова схема мусить
     дати похибку на рівні МК-шуму. Якщо ні — баг в індексах або моментах.
 (2) МК-шум: дві незалежні половини вибірки; `RMS(m¹−m²)/√2` мусить бути
     помітно МЕНШИЙ за виміряну похибку, інакше вимір нічого не значить.
 (3) `γ₃`,`γ₄` шару 1 мусять бути ≈0 (гаус) — контроль самих кумулянтів.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ (щоб не підганяти тлумачення):
   `t` малий (|t|~0.1–0.3) · внесок `κ₃` у 3–10× менший за `κ₄` ·
   гаусова схема дає ~0.2% локально · `+κ₄` знімає більшу частину ⟹ ~0.05–0.1%
   · рекурсія `κ₄` по діагоналі ПРОМАХНЕТЬСЯ (як `κ₃`), бо змішані `(j,j,k,k)`
   члени мають вагу `Σ_j W_ji²·Σ_k W_ki² = 4` проти діагональних ~1.
   Шанс, що `+κ₄` знімає ≥2× локальної похибки: **65%**.
   Шанс, що діагональна рекурсія `κ₄` дає ≥50% істини: **25%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/perneuron.py [cuda:N] [NETS] [LOG2N]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
NS = 1 << LOG2N
BATCH = 1 << 13
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений (ТАБУ/монітор)'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж, вибірка 2^{LOG2N} = {NS}, {DEV}\n', flush=True)

SQ2PI = math.sqrt(2.0 * math.pi)


def phi(t):
    return torch.exp(-0.5 * t * t) / SQ2PI


def Phi(t):
    return 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))


def collect(W, dev, ns, batch, seed):
    """Сирі моменти 1..4 понейронно для `h` і для `a=relu(h)`, ПО ДВОХ ПОЛОВИНАХ.

    Повертає `S[half][шар][степінь]` — суми, і `cnt` на половину.
    Накопичення у fp64 (редукція memory-bound, fp32-акумуляція на 2^24 зразках
    втратила б саме ті цифри, які ми міряємо).
    """
    S = [[torch.zeros(4, W_, dtype=torch.float64, device=dev) for _ in range(L_)]
         for _ in range(2)]
    T = [[torch.zeros(4, W_, dtype=torch.float64, device=dev) for _ in range(L_)]
         for _ in range(2)]
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    d = 0
    while d < ns:
        b = min(batch, ns - d)
        half = 0 if d < ns // 2 else 1
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            h = a @ W[i]
            hd = h.double()
            p = hd
            for k in range(4):
                S[half][i][k] += p.sum(0)
                p = p * hd
            a = torch.relu(h)
            ad = a.double()
            p = ad
            for k in range(4):
                T[half][i][k] += p.sum(0)
                p = p * ad
        d += b
    return S, T, ns // 2


def cumulants(S, n):
    """Центральні кумулянти 1..4 понейронно з сирих сум."""
    m1 = S[0] / n
    m2 = S[1] / n
    m3 = S[2] / n
    m4 = S[3] / n
    c2 = m2 - m1 * m1
    c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    # 🚨 μ₄ = m₄ − 4m₃m₁ + 6m₂m₁² − 3m₁⁴ ;  κ₄ = μ₄ − 3c₂²
    #    (С86: перша редакція мала +12m₁²m₂ − 6m₁⁴ ⟹ γ₄ ~ +790, зловила ПЛОМБА)
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
    c4 = mu4 - 3.0 * c2 * c2
    return m1, c2, c3, c4


def predict(m, c2, c3, c4, mode):
    """`E[ReLU(h)]` за понейронними моментами.

    mode: 'G','K3','K4','K34','FULL' — понейронні `γ`;
    🔑'SCAL' — `γ₃`,`γ₄` замінені СЕРЕДНІМ по нейронах шару (ОДИН скаляр на шар).
       Якщо SCAL ≈ K34, то понейронний розкид `γ` НЕ ПОТРІБЕН, і вести треба
       лише скалярну траєкторію `γ(ℓ)` — а це знімає потребу в рекурсії векторів.
    """
    s = torch.sqrt(torch.clamp(c2, min=1e-30))
    t = m / s
    g3 = c3 / s ** 3
    g4 = c4 / s ** 4
    if mode == 'SCAL':
        g3 = torch.full_like(g3, float(g3.mean()))
        g4 = torch.full_like(g4, float(g4.mean()))
        mode = 'K34'
    elif mode in ('REG', 'REG3'):
        # 🔑 ОРАКУЛЬНА СТЕЛЯ «γ ІЗ t»: corr(γ₃,t)=0.93 ⟹ R²=0.87. Тут γ₃,γ₄
        #    замінені НАЙКРАЩОЮ регресією на t (коефіцієнти підібрані на ЦЬОМУ
        #    ж шарі з ІСТИНИ — верхня межа). REG: базис [1,t]; REG3: [1,t,t²,t³].
        cols = [torch.ones_like(t), t] + ([t * t, t ** 3] if mode == 'REG3' else [])
        X = torch.stack(cols, 1)
        G = X.T @ X + torch.eye(X.shape[1], dtype=X.dtype, device=X.device) * 1e-12
        g3 = X @ torch.linalg.solve(G, X.T @ g3)
        g4 = X @ torch.linalg.solve(G, X.T @ g4)
        mode = 'K34'
    elif mode == 'TWOS':
        # 🔑 ГІПОТЕЗА RANK-COLLAPSE: h_i ≈ c_i·u(z) ⟹ γ₃ᵢ = sign(c_i)·γ₃(u),
        #    tᵢ = sign(c_i)·t_u. Тоді на шар досить ДВОХ чисел: |γ₃| і γ₄.
        g3 = torch.sign(t) * float(torch.sqrt((g3 * g3).mean()))
        g4 = torch.full_like(g4, float(g4.mean()))
        mode = 'K34'
    ph, Ph = phi(t), Phi(t)
    out = t * Ph + ph
    if mode in ('K3', 'K34', 'FULL'):
        out = out + (g3 / 6.0) * (-t * ph)
    if mode in ('K4', 'K34', 'FULL'):
        out = out + (g4 / 24.0) * ((t * t - 1.0) * ph)
    if mode == 'FULL':
        out = out + (g3 * g3 / 72.0) * ((t ** 4 - 6.0 * t * t + 3.0) * ph)
    return s * out


MODES = ('G', 'K3', 'K4', 'K34', 'FULL', 'SCAL', 'TWOS', 'REG', 'REG3')
ERR = {k: np.zeros((len(FS), L_)) for k in MODES}
NOISE = np.zeros((len(FS), L_))
TABS = np.zeros((len(FS), L_))
G3A = np.zeros((len(FS), L_))   # RMS γ₃
G3M = np.zeros((len(FS), L_))   # mean γ₃
CTG = np.zeros((len(FS), L_))   # корел(γ₃, t) понейронно
CSG = np.zeros((len(FS), L_))   # корел(γ₃, sign t) — тест rank-collapse
C34 = np.zeros((len(FS), L_))   # корел(γ₃, γ₄) понейронно
G4A = np.zeros((len(FS), L_))   # mean γ₄
G4R = np.zeros((len(FS), L_))   # RMS  γ₄
TSPR = np.zeros((len(FS), L_))  # std  t
RECV = np.zeros((len(FS), L_))       # діагональна рекурсія var
RECK4 = np.zeros((len(FS), L_))      # діагональна рекурсія k4
RECV_S = np.zeros((len(FS), L_))     # її масштаб
RECK4_S = np.zeros((len(FS), L_))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        S, T, cnt = collect(W, DEV, NS, BATCH, 4400 + n)
        for i in range(L_):
            # --- повна вибірка = сума половин ---
            Sf = [S[0][i][k] + S[1][i][k] for k in range(4)]
            Tf = [T[0][i][k] + T[1][i][k] for k in range(4)]
            m, c2, c3, c4 = cumulants(Sf, 2 * cnt)
            ma = Tf[0] / (2 * cnt)                       # ІСТИНА: E[relu(h)]
            s = torch.sqrt(torch.clamp(c2, min=1e-30))
            # 🔑 RMS, а не середнє: понейронний РОЗКИД `γ` — це те, що входить
            #    у похибку, а середнє по нейронах може бути ~0 при великому розкиді.
            tt = m / s
            TABS[n, i] = float(tt.abs().mean())
            TSPR[n, i] = float(tt.std())
            gg3 = c3 / s ** 3
            gg4 = c4 / s ** 4
            G3M[n, i] = float(gg3.mean())
            for _a, _x, _y in ((CTG, gg3, tt), (C34, gg3, gg4),
                               (CSG, gg3, torch.sign(tt))):
                _u = _x - _x.mean(); _v = _y - _y.mean()
                _d = float(torch.sqrt((_u * _u).sum() * (_v * _v).sum()))
                _a[n, i] = float((_u * _v).sum()) / _d if _d > 0 else 0.0
            G3A[n, i] = float(torch.sqrt(((c3 / s ** 3) ** 2).mean()))
            G4A[n, i] = float((c4 / s ** 4).mean())
            G4R[n, i] = float(torch.sqrt(((c4 / s ** 4) ** 2).mean()))
            base = float(torch.sqrt((ma * ma).mean()))
            for k in MODES:
                p = predict(m, c2, c3, c4, k)
                ERR[k][n, i] = float(torch.sqrt(((p - ma) ** 2).mean())) / base
            # ⛔ПЛОМБА(2): МК-шум із двох незалежних половин
            a1 = T[0][i][0] / cnt
            a2 = T[1][i][0] / cnt
            NOISE[n, i] = float(torch.sqrt(((a1 - a2) ** 2).mean())
                                / math.sqrt(2.0)) / base
            # --- ВИМІР B: діагональна рекурсія з шару i-1 у шар i ---
            if i > 0:
                Tp = [T[0][i - 1][k] + T[1][i - 1][k] for k in range(4)]
                _, av, _, ak4 = cumulants(Tp, 2 * cnt)
                w2 = (W[i].double()) ** 2
                pv = (w2 * av[:, None]).sum(0)
                pk4 = ((w2 * w2) * ak4[:, None]).sum(0)
                for arr, sarr, pr, tr in ((RECV, RECV_S, pv, c2),
                                          (RECK4, RECK4_S, pk4, c4)):
                    x = pr - pr.mean()
                    y = tr - tr.mean()
                    den = float(torch.sqrt((x * x).sum() * (y * y).sum()))
                    arr[n, i] = float((x * y).sum()) / den if den > 0 else 0.0
                    sarr[n, i] = float(pr.mean() / tr.mean())
    print(f'  … {n+1}/{len(FS)} seed {sd0} · '
          f'ПЛОМБА(1) шар1 G-похибка {ERR["G"][n,0]*100:.4f}% '
          f'(МК-шум {NOISE[n,0]*100:.4f}%) · '
          f'шар16 G {ERR["G"][n,15]*100:.3f}% → K4 {ERR["K4"][n,15]*100:.3f}%',
          flush=True)
    del W, S, T
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/perneuron.npz',
         **{f'err_{k}': ERR[k] for k in MODES},
         noise=NOISE, tabs=TABS, tspr=TSPR, g3=G3A, g3m=G3M, g4=G4A, g4r=G4R,
         ctg=CTG, csg=CSG, c34=C34,
         recv=RECV, reck4=RECK4, recv_s=RECV_S, reck4_s=RECK4_S)

print('\n' + '=' * 78)
print('⛔ПЛОМБА(1) ШАР 1 (h=zᵀW₀ ТОЧНО гаусів ⟹ G мусить = МК-шум):')
print(f'   G-похибка {ERR["G"][:,0].mean()*100:.4f}%   '
      f'МК-шум {NOISE[:,0].mean()*100:.4f}%   '
      f'відношення {ERR["G"][:,0].mean()/max(NOISE[:,0].mean(),1e-12):.2f}')
print(f'⛔ПЛОМБА(3) шар 1: RMS γ₃ = {G3A[:,0].mean():.5f}  '
      f'RMS γ₄ = {G4R[:,0].mean():.5f}  (гаус ⟹ обидва ≈0)')
print('=' * 78)

print('\n📊 A. ЛОКАЛЬНА ПОХИБКА `E[ReLU]` ПРИ ІСТИННИХ ПОНЕЙРОННИХ МОМЕНТАХ (%)')
print(f'{"шар":>4s} {"МК-шум":>9s} {"G":>9s} {"G+κ₃":>9s} {"G+κ₄":>9s} '
      f'{"G+κ₃+κ₄":>9s} {"повний":>9s} {"|t|":>6s} {"σ(t)":>6s} '
      f'{"RMSγ₃":>7s} {"⟨γ₄⟩":>8s} {"RMSγ₄":>7s}')
print('─' * 108)
for i in [0, 1, 2, 3, 5, 7, 11, 15, 19, 23, 27, 31]:
    print(f'{i+1:4d} {NOISE[:,i].mean()*100:9.4f} '
          f'{ERR["G"][:,i].mean()*100:9.4f} {ERR["K3"][:,i].mean()*100:9.4f} '
          f'{ERR["K4"][:,i].mean()*100:9.4f} {ERR["K34"][:,i].mean()*100:9.4f} '
          f'{ERR["FULL"][:,i].mean()*100:9.4f} {TABS[:,i].mean():6.3f} '
          f'{TSPR[:,i].mean():6.3f} {G3A[:,i].mean():7.4f} '
          f'{G4A[:,i].mean():+8.4f} {G4R[:,i].mean():7.4f}')
print('─' * 108)
sl = slice(4, 32)
print('СЕРЕДНЄ по шарах 5..32:')
for k, nm in zip(MODES, ('G (гаус)', 'G+κ₃', 'G+κ₄', 'G+κ₃+κ₄', 'повний Еджворт',
                         '🔑СКАЛЯРНИЙ γ (1 число/шар)',
                         '🔑ДВА СКАЛЯРИ sign(t)·|γ₃|, γ₄',
                         '🔑ОРАКУЛ γ=a+b·t', '🔑ОРАКУЛ γ=poly₃(t)')):
    v = ERR[k][:, sl].mean() * 100
    print(f'   {nm:<18s} {v:8.4f}%   ×{ERR["G"][:,sl].mean()/max(ERR[k][:,sl].mean(),1e-12):.3f} проти гауса')
print(f'   ⛔МК-шум          {NOISE[:,sl].mean()*100:8.4f}%')

print('\n📊 B. ДІАГОНАЛЬНА РЕКУРСІЯ МОМЕНТІВ (корел із істиною / масштаб)')
print(f'{"шар":>4s} {"corr var":>10s} {"scale var":>10s} {"corr κ₄":>10s} {"scale κ₄":>10s}')
print('─' * 50)
for i in [1, 2, 3, 5, 7, 11, 15, 19, 23, 27, 31]:
    print(f'{i+1:4d} {RECV[:,i].mean():10.4f} {RECV_S[:,i].mean():10.4f} '
          f'{RECK4[:,i].mean():10.4f} {RECK4_S[:,i].mean():10.4f}')
print('─' * 50)
print(f'СЕРЕДНЄ 5..32:  var corr {RECV[:,sl].mean():.4f} scale {RECV_S[:,sl].mean():.4f}  |  '
      f'κ₄ corr {RECK4[:,sl].mean():.4f} scale {RECK4_S[:,sl].mean():.4f}')
print('\n📊 C. СТРУКТУРА `γ` ПОНЕЙРОННО (чи потрібен вектор, чи вистачить скаляра)')
print(f'{"шар":>4s} {"⟨γ₃⟩":>8s} {"RMSγ₃":>8s} {"розкид":>8s} {"частка сер.":>12s} '
      f'{"corr(γ₃,t)":>11s} {"corr(γ₃,sgn t)":>14s} {"corr(γ₃,γ₄)":>12s}')
print('─' * 80)
for i in [1, 3, 5, 7, 11, 15, 19, 23, 27, 31]:
    mu = G3M[:, i].mean(); rms = G3A[:, i].mean()
    spr = math.sqrt(max(rms * rms - mu * mu, 0.0))
    print(f'{i+1:4d} {mu:8.4f} {rms:8.4f} {spr:8.4f} {mu/max(rms,1e-12):12.3f} '
          f'{CTG[:, i].mean():11.4f} {CSG[:, i].mean():14.4f} '
          f'{C34[:, i].mean():12.4f}')
print('─' * 80)
print('\n🎯 A каже, чи ВИСТАЧАЄ 4 моментів. B каже, чи можна їх ВЕСТИ без МК.')
print('🎯 C каже, чи `γ` — ВЕКТОР (треба рекурсія) чи СКАЛЯР (треба лише траєкторія).')
