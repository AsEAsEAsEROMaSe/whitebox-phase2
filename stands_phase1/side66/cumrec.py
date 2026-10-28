"""ТОЧНІШЕ `E[a²]` У САМІЙ РЕКУРСІЇ ЧЕРЕЗ ВЛАСНУ РЕКУРСІЮ КУМУЛЯНТІВ. (С85)

## ЗВІДКИ — ФОРМУЛЮВАННЯ ЮЗЕРА
«Шар зрізав, а далі ваги відновили заново.» Це БУКВАЛЬНИЙ опис нашої єдиної
апроксимації: `h = Wᵀa` і `Cov = WᵀCW` ТОЧНІ, наближаємо лише `E[ReLU(h)]`,
припускаючи, що ЦГТ наступного шару відновив гаусовість ПОВНІСТЮ.
🔑 Рамка ділить процес НАДВОЄ, і половини мають різну долю:
  ⛔ «наскільки ШВИДКО відновилось» = форма, κ₃/κ₄, Еджворт — **мертво тричі**
     (у сабміті ×0.875; [[finding-closure-is-a-balance-point-not-an-approximation]]),
     і `log n_eff` після центрування МЕРЕЖА×ШАР дав −0.396 → **−0.005**;
  🎯 «скільки ЛИШИЛОСЬ після зрізу» = `E[a²]` — **ось це досі не точне**.
РІЗНИЦЯ ПРИНЦИПОВА: перше — поправка до ОЗНАКИ (її `λ` всмокче), друге —
краща `C` У КОЖНОМУ КРОЦІ, накопичена 32 рази. Стеля ×12.1.

## 🔑 СХЕМА (замкнена, БЕЗ вибірки)
1. `L_k = E[h^k · 1{h>0}]` для гаусового `h~N(μ,σ²)` — ТОЧНА рекурсія
   `L_0 = Φ(t)`, `L_1 = μΦ + σφ`, `L_k = μ·L_{k-1} + (k-1)σ²·L_{k-2}`, `t=μ/σ`.
2. кумулянти АКТИВАЦІЇ: `κ₃(a) = m₃`, `κ₄(a) = m₄ − 3m₂²` через `L_1..L_4`.
3. поширення на наступний шар: `h'_i = Σ_j W_ji a_j` ⟹ ЗА НЕЗАЛЕЖНОСТІ
   `κ₃(h') = (W³)ᵀ κ₃(a)`, `κ₄(h') = (W⁴)ᵀ κ₄(a)` (степені ПОЕЛЕМЕНТНІ).
   🚨 це НАБЛИЖЕННЯ (активації корельовані) — і саме його перевіряє пломба.
4. поправлені `E[a]`, `E[a²]` за Еджвортом з ЦИХ κ (формули з С84).

## ⛔ ПЛОМБА ПЕРЕД БУДЬ-ЯКИМ УЖИВАННЯМ (гейт із NEXT_SESSION §0.000)
`κ₃,κ₄` з рекурсії проти `κ₃,κ₄` з вибірки 2²⁰ семплів:
  (а) кореляція ВСЕРЕДИНІ шару ≥ 0.9 — інакше рекурсія міряє не те;
  (б) масштаб `рекурсія/вибірка` СТАЛИЙ по шарах — інакше вона ловить глибину,
      а не механізм (рівно та хвороба, що вбила `log n_eff`);
  🚨 кореляцію рахувати ВСЕРЕДИНІ клітинки МЕРЕЖА×ШАР, НЕ наскрізь.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: пломба (а) пройде на дрібних шарах і посиплеться з
   глибиною, бо припущення незалежності активацій псується накопиченням
   кореляцій; (б) — головний ризик. Оцінка шансів довести до бала: ~25%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/cumrec.py [cuda:N] [NETS] [LOG2N]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import _phi, _Phi                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
BATCH = 1 << 15
L_, W_ = 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж, вибірка 2^{LOG2N} = {NS}, {DEV}\n', flush=True)


def trunc_moments(mu, s):
    """L_k = E[h^k·1{h>0}] для h~N(mu,s²), k=0..4. ТОЧНО."""
    t = mu / s
    ph, Ph = _phi(t), _Phi(t)
    s2 = s * s
    L0 = Ph
    L1 = mu * Ph + s * ph
    L2 = mu * L1 + s2 * L0
    L3 = mu * L2 + 2.0 * s2 * L1
    L4 = mu * L3 + 3.0 * s2 * L2
    return L1, L2, L3, L4


def act_cumulants(mu, s):
    """κ₃,κ₄ активації a=ReLU(h) при ГАУСОВОМУ h~N(mu,s²)."""
    L1, L2, L3, L4 = trunc_moments(mu, s)
    m2 = L2 - L1 * L1
    m3 = L3 - 3.0 * L1 * L2 + 2.0 * L1 ** 3
    m4 = L4 - 4.0 * L1 * L3 + 6.0 * L1 * L1 * L2 - 3.0 * L1 ** 4
    return m3, m4 - 3.0 * m2 * m2


def closure_cum(W, dev, kmax=None):
    """Замикання + ВЛАСНА рекурсія κ₃,κ₄ передактивацій. Без вибірки."""
    n0 = W[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    k3 = torch.zeros(n0, dtype=torch.float64, device=dev)
    k4 = torch.zeros(n0, dtype=torch.float64, device=dev)
    K3, K4, MU, SG = [], [], [], []
    nit = len(W) if kmax is None else kmax
    for i in range(nit):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        # 🎯 κ передактивації ЦЬОГО шару — з κ активацій попереднього
        k3h = (w ** 3).T @ k3
        k4h = (w ** 4).T @ k4
        K3.append(k3h.clone()); K4.append(k4h.clone())
        MU.append(mp.clone()); SG.append(s.clone())
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        if i < nit - 1:
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
            C = C + torch.diag(d - C.diagonal())
            k3, k4 = act_cumulants(mp, s)
    return K3, K4, MU, SG, m


def sample_cum(W, dev, ns, batch, seed):
    """κ₃,κ₄ передактивацій З ВИБІРКИ — еталон для пломби."""
    n = len(W)
    S1 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(n)]
    S2 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(n)]
    S3 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(n)]
    S4 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(n)]
    OUT = torch.zeros(W_, dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    done = 0
    while done < ns:
        b = min(batch, ns - done)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(n):
            h = a @ W[i]
            hd = h.double(); h2 = hd * hd
            S1[i] += hd.sum(0); S2[i] += h2.sum(0)
            S3[i] += (h2 * hd).sum(0); S4[i] += (h2 * h2).sum(0)
            a = torch.relu(h)
            if i == n - 1:
                OUT += a.double().sum(0)
        done += b
    K3, K4 = [], []
    for i in range(n):
        e1 = S1[i]/ns; e2 = S2[i]/ns; e3 = S3[i]/ns; e4 = S4[i]/ns
        v = e2 - e1*e1
        K3.append(e3 - 3.0*e1*e2 + 2.0*e1**3)
        K4.append(e4 - 4.0*e1*e3 + 6.0*e1*e1*e2 - 3.0*e1**4 - 3.0*v*v)
    return K3, K4, OUT/ns


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = float(x.norm() * y.norm())
    return float((x*y).sum()) / d if d > 0 else float('nan')


ROWS = []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        K3c, K4c, MU, SG, mdeep = closure_cum(W, DEV)
        K3s, K4s, out = sample_cum(W, DEV, NS, BATCH, 9000 + n)
        b1, b2 = TRUTH2[sd0]
        T = 0.5*(torch.tensor(b1, dtype=torch.float64, device=DEV)
                 + torch.tensor(b2, dtype=torch.float64, device=DEV))
        row = []
        for i in range(L_):
            g3 = corr(K3c[i], K3s[i]); g4 = corr(K4c[i], K4s[i])
            # масштаб — робастно, через відношення норм
            r3 = float(K3c[i].norm() / K3s[i].norm().clamp_min(1e-300))
            r4 = float(K4c[i].norm() / K4s[i].norm().clamp_min(1e-300))
            row.append((g3, g4, r3, r4))
        ROWS.append(row)
        eo = float((out - T).abs().mean() / T.abs().mean())
        em = float((mdeep - T).abs().mean() / T.abs().mean())
        print(f'  … {n+1}/{len(FS)} seed {sd0} · '
              f'⛔ПЛОМБА вибірки: |МК−істина|/|істина| = {eo:.2e} '
              f'(замикання {em:.2e})', flush=True)
    del W
    torch.cuda.empty_cache()

A = np.array(ROWS)                      # (нетів, шарів, 4)
np.save('/mnt/win_d/llm/White-Box/arena62/logs/cumrec.npy', A)
print(f'\n⛔ПЛОМБА (а) корел рекурсія↔вибірка ВСЕРЕДИНІ шару · '
      f'⛔ПЛОМБА (б) масштаб мусить бути СТАЛИЙ по шарах')
print(f'{"шар":>5s} {"корел κ₃":>10s} {"корел κ₄":>10s} '
      f'{"масштаб κ₃":>11s} {"масштаб κ₄":>11s}')
print('─' * 52)
for i in range(1, L_):
    if i <= 10 or i % 5 == 0 or i == L_ - 1:
        print(f'{i:5d} {A[:, i, 0].mean():10.4f} {A[:, i, 1].mean():10.4f} '
              f'{A[:, i, 2].mean():11.4f} {A[:, i, 3].mean():11.4f}')
print('─' * 52)
c3, c4 = A[:, 1:, 0].mean(), A[:, 1:, 1].mean()
s3 = A[:, 1:, 2].mean(0); s4 = A[:, 1:, 3].mean(0)
print(f'сер. корел: κ₃ {c3:+.4f} · κ₄ {c4:+.4f}   (гейт ≥0.90)')
print(f'масштаб: κ₃ {s3.mean():.4f} ± {s3.std():.4f} '
      f'({s3.std()/abs(s3.mean())*100:.1f}%) · '
      f'κ₄ {s4.mean():.4f} ± {s4.std():.4f} '
      f'({s4.std()/abs(s4.mean())*100:.1f}%)   (гейт: розкид <20%)')
ok = (c3 >= 0.9 and c4 >= 0.9 and s3.std()/abs(s3.mean()) < 0.2
      and s4.std()/abs(s4.mean()) < 0.2)
print(f'\n🎯 ПЛОМБА {"ПРОЙДЕНА ✅ — можна підставляти в E[a²]" if ok else "ПРОВАЛЕНА ⛔ — гілку закрито ДО реалізації"}')
