"""ГІБРИДНЕ ЗАМИКАННЯ: γ₁ З ВИБІРКИ, ЯКУ САБМІТ УЖЕ МАЄ (С83)

## ЗВІДКИ
`ground.py`: похибка ФОРМИ ~0.2% і не росте вглиб; похибка ЛАНЦЮГА = A·√(ℓ−1)
до 1.0%; 🟢 Еджворт з ІСТИННИМ γ₁ ріже форму втричі (C/A=0.359 на ВСІХ шарах).
`kappa3.py`: ⛔ дешевий діагональний κ₃ мертвий — корел +0.646, але масштаб
недооцінює 1.7×→43× вглиб, бере 5.7% стелі.
⟹ лишається третій шлях: **γ₁ прямо з ВИБІРКИ**.

## 🔑 ЧОМУ ЦЕ МАЙЖЕ БЕЗКОШТОВНО
Сабміт і так жене вибірку крізь усі 32 шари й на кожному має передактивації
`p`. Третій центральний момент — ОДНА редукція `Σp³` на шар:
32 × 2·npair × 256 ≈ 6e7 операцій проти бюджету 2.2e10 = **0.3%**, а ми ще й
під підлогою 10% ⟹ 🔑 ціна РІВНО НУЛЬ у балі.

## 🚨 ЧОМУ С82 НЕ СПРОСТОВУЄ ЦЕ
`clsedge.py` (С82, ×0.945) рахував `gam = κ·((w³)·μ₃).sum(0)/s³` — це саме
ДІАГОНАЛЬНА згортка, яку `kappa3.py` зміряв як хибну в 1.7–43×. Той вирок
стосується діагоналі, а не Еджворта.

## ФОРМУЛА (виведена, не вгадана)
`X = μ + σZ`, `Z ~ φ(z)(1 + γ₁He₃(z)/6)`, `t = μ/σ`:
    ∫₋ₜ^∞ He₃φ = (t²−1)φ(t) ;  z·He₃ = He₄ + 3He₂ ⟹ ∫₋ₜ^∞ z·He₃φ = −t³φ(t)
    ⟹ поправка = (γ₁/6)φ(t)[μ(t²−1) − σt³] = **−(γ₁/6)·μ·φ(t)**
Для другого моменту (потрібен для `C`) аналогічно:
    E[ReLU²] += (γ₁/6)·σ²·[2φ(t) + t·(t²−1)... ] — тут беремо прямий вивід
    ∫₋ₜ^∞ (μ+σz)²He₃φ dz = φ(t)·σ²·(2 − t⁴ + ... ) ⟹ коефіцієнт `EQ` нижче.

## ВАРІАНТИ (6 мереж × 2²⁰ істина, γ₁ з вибірки РОЗМІРУ САБМІТА)
  A ⛔ γ₁ = 0                     — нинішнє замикання
  B 🎯 γ₁ з ВИБІРКИ, лише `m`     — одна редукція на шар
  C 🎯 γ₁ з ВИБІРКИ, `m` і `C`    — плюс поправка до другого моменту
  D ⛔ γ₁ ТОЧНИЙ (2²⁰), лише `m`  — СТЕЛЯ шляху
  E ⛔ γ₁ з ЧУЖОЇ мережі          — контроль
Міряємо ‖b‖/‖T‖ на виході (шар 32) — саме те, що входить у бленд.

📣 ПЕРЕДБАЧЕНО: D дасть падіння ‖b‖ у ~2.5–3× (це стеля з ground.py). B буде
   гіршим за D через шум γ₁ на npair≈3700: відносний шум κ₃ ~ √15/√npair ≈ 6%,
   а сам γ₁ малий на мілких шарах ⟹ на шарах 1–8 поправка потоне в шумі, на
   глибоких (|γ₁|≈0.4) має вижити. Чекаю B у 1.5–2.2×. Якщо B ≥ 1.5× — це
   ×1.3…1.5 у балі через бленд, і вперше за сесію щось іде в сабміт.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/edgesamp.py [cuda:N] [NETS] [log2]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import cut_budget91, _phi, _Phi, ALPHA           # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LG = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS, BS, L_, W_ = 1 << LG, 1 << 14, 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
VAR = ['A ⛔ γ₁=0 (нинішнє)', 'B 🎯 γ₁ з вибірки, лише m',
       'C 🎯 γ₁ з вибірки, m і C', 'D ⛔ γ₁ ТОЧНИЙ, лише m (СТЕЛЯ)',
       'E ⛔ γ₁ з ЧУЖОЇ мережі']
NV = len(VAR)
RES = np.zeros((NETS, NV))
G1S = np.zeros((NETS, L_))      # γ₁ з вибірки
G1T = np.zeros((NETS, L_))      # γ₁ точний
prev_g = None
CSW = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]
CS = np.zeros((NETS, len(CSW)))


def closure_edge(W, dev, g1, eq=False):
    """Замикання Мелера з Еджворт-поправкою `g1` (вектор на шар або None).

    `g1[i]` — скошеність передактивацій шару i. eq=True вмикає поправку і до
    другого моменту (тобто до діагоналі `C`).
    """
    n0 = W[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    for i in range(len(W)):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m_new = mp * Ph + s * ph
        d2 = (mp * mp + s * s) * Ph + mp * s * ph          # E[a²]
        if g1 is not None:
            g = g1[i]
            m_new = m_new - (g / 6.0) * mp * ph
            if eq:
                # 🚨 ВИВЕДЕНО НАНОВО (перша версія була хибна і дала ×0.122).
                #   (t+z)² = t²+1 + 2t·He₁ + He₂ ;  He₂He₃ = He₅+6He₃+6He₁
                #   ⟹ (t+z)²He₃ = He₅ + 2t·He₄ + (t²+7)He₃ + 6t·He₂ + 6He₁
                #   ∫₋ₜ^∞ He_nφ = He_{n−1}(−t)φ(t) ⟹ сума = **2φ(t)** (усе
                #   решта скорочується точно) ⟹ поправка = (γ₁/3)·σ²·φ(t).
                #   ✅ збігається з `clsedge.py` (С82), де стояло (gam/3)·s²·ph.
                d2 = d2 + (g / 3.0) * s * s * ph
        if i == len(W) - 1:
            return m_new
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s,
                                                                     ph / s)
        d = d2 - m_new * m_new
        C = C + torch.diag(d - C.diagonal())
        m = m_new
    return m


for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        # ── ІСТИНА і ТОЧНИЙ γ₁ великою вибіркою
        gen = torch.Generator(device=DEV); gen.manual_seed(31337 + n)
        S1 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        S2 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        S3 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        AM = torch.zeros(W_, dtype=torch.float64, device=DEV)
        done = 0
        while done < NS:
            b = min(BS, NS - done)
            z = torch.randn(b, W_, device=DEV, generator=gen)
            a = z
            for i in range(L_):
                h = a @ W[i]
                hd = h.double()
                S1[i] += hd.sum(0); S2[i] += (hd*hd).sum(0); S3[i] += (hd**3).sum(0)
                a = torch.relu(h)
                if i == L_ - 1:
                    AM += a.double().sum(0)
            done += b
        T = AM / NS
        g_true = []
        for i in range(L_):
            mh = S1[i] / NS
            vh = (S2[i] / NS - mh * mh).clamp_min(1e-30)
            k3 = S3[i] / NS - 3 * mh * (S2[i] / NS) + 2 * mh ** 3
            g_true.append(k3 / vh.sqrt() ** 3)
            G1T[n, i] = float(g_true[i].abs().mean())

        # ── γ₁ з ВИБІРКИ РОЗМІРУ САБМІТА (антитетичні пари, як у сабміті)
        wc, npair, rng = cut_budget91(W, DEV)
        zs = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                          device=DEV)
        a = torch.cat([zs, -zs], 0)
        den = float(2 * npair)
        g_samp = []
        for i in range(L_):
            h = a @ W[i]
            hd = h.double()
            mh = hd.sum(0) / den
            vh = (hd.pow(2).sum(0) / den - mh * mh).clamp_min(1e-30)
            k3 = hd.pow(3).sum(0) / den - 3 * mh * (hd.pow(2).sum(0) / den) \
                + 2 * mh ** 3
            g_samp.append(k3 / vh.sqrt() ** 3)
            G1S[n, i] = float(g_samp[i].abs().mean())
            a = torch.relu(h)

        # ⛔ПЛОМБА: моє замикання при γ₁=0 мусить дати РІВНО `closure91`
        from sub91_port import closure91 as _c91
        _ref = _c91(W, L_ - 1, DEV, kdeep=None)[4]
        _mine = closure_edge(W, DEV, None)
        _rel = float((_mine - _ref).norm() / _ref.norm())
        if n == 0:
            print(f'⛔ПЛОМБА closure_edge(γ₁=0) ≡ closure91: відн. {_rel:.2e} '
                  f'{"✅" if _rel < 1e-12 else "🚨 РОЗІЙШЛОСЬ — стенд не той"}',
                  flush=True)
        base = T.norm()
        # 🎯 ОПТИМАЛЬНИЙ МНОЖНИК на поправку: якщо гілка жива, він >0.
        #    Свіп по c, бо задача нелінійна (γ₁ входить у рекурсію).
        for ci, cc in enumerate(CSW):
            gg = [x * cc for x in g_true]
            CS[n, ci] = float(((1 + ALPHA) * closure_edge(W, DEV, gg) - T
                               ).norm() / T.norm())
        outs = [closure_edge(W, DEV, None),
                closure_edge(W, DEV, g_samp),
                closure_edge(W, DEV, g_samp, eq=True),
                closure_edge(W, DEV, g_true),
                closure_edge(W, DEV, prev_g if prev_g is not None else g_samp)]
        for v in range(NV):
            RES[n, v] = float(((1 + ALPHA) * outs[v] - T).norm() / base)
        prev_g = [x.clone() for x in g_samp]
    del W, wc, a
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  ‖b‖/‖T‖: '
          + '  '.join(f'{RES[n,v]:.4f}' for v in range(NV)), flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/edgesamp.npz',
         RES=RES, G1S=G1S, G1T=G1T, CS=CS, CSW=np.array(CSW))
print(f'\n{NETS} мереж, істина 2^{LG}, γ₁ з вибірки розміру сабміта (npair≈{npair})')
print(f'\n{"варіант":34s} {"‖b‖/‖T‖":>10s} {"× до A":>8s} {"‖b‖² ×":>8s}')
print('─' * 66)
a0 = RES[:, 0].mean()
for v in range(NV):
    r = RES[:, v].mean()
    print(f'{VAR[v]:34s} {r:10.5f} {a0/r:8.3f} {(a0/r)**2:8.3f}'
          + ('  🟢' if a0 / r > 1.05 else ''))
print(f'\n🎯 СВІП МНОЖНИКА НА ПОПРАВКУ (γ₁ ТОЧНИЙ × c), ‖b‖/‖T‖')
print('   ' + '  '.join(f'{c:+5.2f}' for c in CSW))
print('   ' + '  '.join(f'{CS[:,i].mean():.4f}' for i in range(len(CSW))))
_b = int(np.argmin(CS.mean(0)))
print(f'   ⟹ оптимум c = **{CSW[_b]:+.2f}**  '
      + ('🚨 НУЛЬ або МІНУС — поправка НЕ ПОТРІБНА, гілку закрито'
         if CSW[_b] <= 0 else '🟢 додатний — гілка жива'))
print(f'\n🔑 ЯКІСТЬ γ₁ З ВИБІРКИ (|γ₁| сер. по нейронах)')
print(f'{"шар":>4s} {"з вибірки":>10s} {"точний":>9s} {"відн.":>7s}')
for i in (0, 3, 7, 11, 15, 19, 23, 27, 31):
    print(f'{i+1:4d} {G1S[:,i].mean():10.4f} {G1T[:,i].mean():9.4f} '
          f'{G1S[:,i].mean()/max(G1T[:,i].mean(),1e-30):7.3f}')
b = a0 / RES[:, 1].mean()
sig2_ratio = 9.6 / b ** 2          # ‖b‖² було 9.6σ², стало 9.6/b²
w_new = 1.0 / (1.0 + sig2_ratio)
mse_new = (1 - w_new) ** 2 + w_new ** 2 * sig2_ratio
print(f'\n📊 ПЕРЕРАХУНОК У БАЛ (з `‖b‖²=9.6σ²` та MSE=0.909σ² при w=0.0883)')
print(f'   ‖b‖² падає в {b**2:.2f}× ⟹ {sig2_ratio:.2f}σ² ⟹ нове w* = {w_new:.3f}')
print(f'   MSE {mse_new:.3f}σ² проти 0.909σ² ⟹ **×{0.909/mse_new:.3f}** '
      f'⟹ бал {1.946e-07*mse_new/0.909:.4e}')
