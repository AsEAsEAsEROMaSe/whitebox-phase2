"""ГЕЙТ `β` ДЛЯ КАНАЛУ `n_act`: чи влазить опора в `β ≲ σ_g/√N`. (С86)

## ЗВІДКИ — ІДЕЯ ЮЗЕРА (С86)
«Нам треба на кожну вагу поставити імовірність спрацювання, і наші 6к семплів
мають створити картину активності.» `side66/actcv.py` зміряв СТЕЛЮ:
`R²(a_L | n_act₁..₃₂) = 0.2493` ⟹ ×1.332 дисперсії, **×1.258 у балі**.
⚠️Але то стеля при ІДЕАЛЬНІЙ опорі. Тут — гейт, якого бракувало.

## 📐 ЦЕНТРАЛЬНЕ РІВНЯННЯ (С85)
Оцінка з контрольною змінною: `â_j = mean_z a_j − λ_j·(mean_z g − ĝ)`,
де `g(z) = n_act_ℓ(z)`, а `ĝ = Σ_i Φ(t_i)` — опора ІЗ ЗАМИКАННЯ (зміщена на `β`).
```
MSE_j = (Var_j − 2λCov_j + λ²Var_g)/N + λ²β²
λ_opt = Cov_j/(Var_g + N·β²)
MSE_opt = Var_j/N − Cov_j²/(N·(Var_g + N·β²))
виграш  = 1/(1 − ρ²·Var_g/(Var_g + N·β²)),   ρ² = Cov²/(Var_j·Var_g)
```
⟹ 🔑 канал живий ЛИШЕ коли `N·β² ≲ Var_g`, тобто **`β ≲ σ_g/√N`**.

## 🔑 ЧОМУ ОПОРА МАЙЖЕ БЕЗКОШТОВНА
У сабміті `_closure_parts` уже рахує `Ph = Φ(t)` на КОЖНОМУ шарі (глибока
гілка йде до `depth`), просто зберігає лише для `i ≤ ksig`. `Σ_i Φ(t_i)` — це
одна редукція на шар поверх уже наявного.

## 🎯 ЩО МІРЯЄМО (замикання ТОЧНО як у сабміті: C₀=I, Мелер K=2, точна діагональ)
* `β_ℓ = ĝ_ℓ − E[n_act_ℓ]` — зміщення опори, ПОШАРОВО;
* `σ_g`, `ρ` (корел `n_act_ℓ` з виходом), `Var_j` — з МК;
* виграш при `N`=5000 з урахуванням `β` проти ідеального `β=0`.

⛔ПЛОМБИ:
 (1) шар 1: `h₁=zᵀW₀` ТОЧНО гаусів, `C₀=I` точна ⟹ `β₁` мусить бути ≈0
     (лише МК-шум). Якщо ні — баг у замиканні;
 (2) `E[n_act]` з МК мусить = `Σ_i gate_i` (тотожність) — контроль обліку;
 (3) ⛔контроль: опора від ЧУЖОЇ мережі ⟹ `β` мусить вирости на порядок.

📣 ПЕРЕДБАЧЕНО: `σ_g ≈ 8–20` (нейрони корельовані ⟹ більше за біноміальні 8),
   `σ_g/√5000 ≈ 0.11–0.28`. `β` від замикання з 0.19% похибкою в `m` дасть
   ~0.1–1.0 на глибині ⟹ ГЕЙТ, найімовірніше, НЕ ПРОЙДЕ на глибоких шарах,
   але може пройти на 1–8, де замикання точне.
   Шанс, що хоч 8 шарів дадуть сумарний виграш ≥×1.05: **35%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/actbeta.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
BATCH = 1 << 12
L_, W_ = 32, 256
N_WORK = 5000.0                     # робоче число зразків у сабміті
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
SQ2PI = math.sqrt(2.0 * math.pi)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} · N_work={N_WORK:.0f} · {DEV}\n', flush=True)


def closure_gates(W):
    """`Σ_i Φ(t_i)` на кожному шарі — ТОЧНО за схемою `_closure_parts` сабміту."""
    m = torch.zeros(W_, dtype=torch.float64, device=DEV)
    C = torch.eye(W_, dtype=torch.float64, device=DEV)
    out = []
    for i in range(L_):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(cp.diagonal(), min=1e-20))
        mp = m @ w
        t = mp / s
        ph = torch.exp(-0.5 * t * t) / SQ2PI
        Ph = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
        out.append(float(Ph.sum()))
        m = mp * Ph + s * ph
        if i == L_ - 1:
            break
        C = cp * torch.outer(Ph, Ph) + (0.5 * (cp * cp)) * torch.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + torch.diag(d - C.diagonal())
    return torch.tensor(out, dtype=torch.float64, device=DEV)


BETA = np.zeros((len(FS), L_))
SIGG = np.zeros((len(FS), L_))
RHO2 = np.zeros((len(FS), L_))
GAIN0 = np.zeros(len(FS))           # виграш при β=0 (усі 32 шари)
GAINB = np.zeros(len(FS))           # виграш з реальним β
GAINH = np.zeros(len(FS))           # лише шари 1..8
BETAX = np.zeros((len(FS), L_))     # ⛔контроль: чужа опора

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    Wx = [torch.tensor(np.asarray(WZ[have[int(np.load(FS[(n + 1) % len(FS)])['seed'])]][q],
                                  np.float32), device=DEV) for q in range(L_)]
    with torch.no_grad():
        ghat = closure_gates(W)
        ghat_x = closure_gates(Wx)                     # ⛔контроль(3)
        sg = torch.zeros(2, L_, dtype=torch.float64, device=DEV)
        sa = torch.zeros(2, W_, dtype=torch.float64, device=DEV)
        sga = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        gg = torch.zeros(L_, L_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(8200 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            na = torch.empty(b, L_, dtype=torch.float64, device=DEV)
            for q in range(L_):
                h = a @ W[q]
                na[:, q] = (h > 0).double().sum(1)
                a = torch.relu(h)
            ad = a.double()
            sg[0] += na.sum(0); sg[1] += (na * na).sum(0)
            sa[0] += ad.sum(0); sa[1] += (ad * ad).sum(0)
            sga += na.T @ ad
            gg += na.T @ na
            d += b
        mg = sg[0] / NS
        vg = sg[1] / NS - mg * mg
        ma = sa[0] / NS
        va = sa[1] / NS - ma * ma
        cov = sga / NS - torch.outer(mg, ma)
        Gm = gg / NS - torch.outer(mg, mg)
        BETA[n] = (ghat - mg).cpu().numpy()
        BETAX[n] = (ghat_x - mg).cpu().numpy()
        SIGG[n] = torch.sqrt(vg).cpu().numpy()
        RHO2[n] = ((cov * cov).sum(1) / (vg * va.sum())).cpu().numpy()

        # багатовимірний виграш: Δ = covᵀ (G + N·ββᵀ)⁻¹ cov, сумарно по виходах
        def gain(idx, beta):
            Gs = Gm[idx][:, idx].clone()
            if beta is not None:
                bb = beta[idx]
                Gs = Gs + N_WORK * torch.outer(bb, bb)
            Gs = Gs + torch.eye(len(idx), dtype=Gs.dtype, device=DEV) * 1e-8 * Gs.diagonal().mean()
            cv = cov[idx]
            red = float((cv * torch.linalg.solve(Gs, cv)).sum())
            return float(va.sum()) / max(float(va.sum()) - red, 1e-30)

        allidx = torch.arange(L_, device=DEV)
        bt = torch.tensor(BETA[n], dtype=torch.float64, device=DEV)
        GAIN0[n] = gain(allidx, None)
        GAINB[n] = gain(allidx, bt)
        GAINH[n] = gain(allidx[:8], bt)
    print(f'  … {n+1}/{len(FS)} seed {sd0} · ПЛОМБА(1) β₁={BETA[n,0]:+.4f} · '
          f'β₁₆={BETA[n,15]:+.3f} σg₁₆={SIGG[n,15]:.2f} · '
          f'виграш β=0 ×{GAIN0[n]:.3f} → з β ×{GAINB[n]:.3f}', flush=True)
    del W, Wx
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/actbeta.npz',
         beta=BETA, betax=BETAX, sigg=SIGG, rho2=RHO2,
         g0=GAIN0, gb=GAINB, gh=GAINH)
thr = SIGG / math.sqrt(N_WORK)
print('\n' + '=' * 84)
print(f'📊 ГЕЙТ `β ≲ σ_g/√N` при N={N_WORK:.0f}')
print(f'{"шар":>4s} {"|β|":>9s} {"σ_g":>8s} {"поріг σ_g/√N":>13s} {"|β|/поріг":>10s} '
      f'{"ρ²":>7s} {"⛔чужа |β|":>10s}')
print('─' * 84)
for i in [0, 1, 3, 5, 7, 11, 15, 19, 23, 27, 31]:
    b = np.abs(BETA[:, i]).mean()
    print(f'{i+1:4d} {b:9.4f} {SIGG[:,i].mean():8.3f} {thr[:,i].mean():13.4f} '
          f'{b/max(thr[:,i].mean(),1e-12):10.2f} {RHO2[:,i].mean():7.4f} '
          f'{np.abs(BETAX[:,i]).mean():10.4f}')
print('─' * 84)
print(f'⛔ПЛОМБА(1) шар 1: |β| = {np.abs(BETA[:,0]).mean():.5f} (мусить ≈0)')
print(f'⛔ПЛОМБА(3) чужа опора: |β| у {np.abs(BETAX).mean()/max(np.abs(BETA).mean(),1e-12):.1f}× більша')
print(f'\n🎯 ВИГРАШ ДИСПЕРСІЇ (усі 32 шари як регресори, N={N_WORK:.0f}):')
print(f'   при ІДЕАЛЬНІЙ опорі (β=0):  ×{GAIN0.mean():.4f}')
print(f'   з РЕАЛЬНИМ `β` замикання:   ×{GAINB.mean():.4f}')
print(f'   лише шари 1..8, реальний β: ×{GAINH.mean():.4f}')
print(f'\n📈 у балі (дисперсія = 82.5% MSE): '
      f'ідеал ×{1/(0.825/GAIN0.mean()+0.175):.4f} · реально ×{1/(0.825/GAINB.mean()+0.175):.4f}')
