"""width90 — ЧИ ПАДАЄ ПОХИБКА ЗАМИКАННЯ ЯК 1/√n? (питання ЮЗЕРА про ЗВЧ)

════════════════ ПИТАННЯ ЮЗЕРА ════════════════
«чому не використовуємо властивості великих чисел? бо по факту вся суть у
тому, що хоч скільки варіантів не було б, їх середнє можна точно виявити від
вагів, бо вони при кількості самоусереднюються по закону».

ВІДПОВІДЬ ПО СУТІ: ми САМЕ це й робимо — замикання є застосуванням ЦГТ.
Але воно дає 0.935%, а проста МК 0.187%. ЦЕЙ СТЕНД ПИТАЄ: чому саме стільки,
і чи це фундаментальна межа `1/√n`.

📐 ЯКЩО ЗВЧ — ГОЛОВНИЙ МЕХАНІЗМ, то похибка замикання мусить падати як
`n^(−1/2)`. Логарифмічний нахил має бути **−0.5**.
🚨 АЛЕ сьогодні зміряно: анігіляція лишає ЕФЕКТИВНО ~53 доданки з 256 ⟹
похибка може йти не по `n`, а по `n_eff`. Нахил тоді все одно −0.5, але
константа вдвічі гірша.

Ми НІКОЛИ не міряли залежність від ширини — усі 90 сесій працювали при n=256.

════════════════ ПЛОМБИ ════════════════
P1 ВІДТВОРЕННЯ: при n=256 замикання мусить дати ~0.9% (наш робочий рівень)
   і МК на 5432 семплах ~0.24–0.34%. Якщо ні — прилад не той об'єкт.
P2 ГЛИБИНА 1: для ОДНОШАРОВОЇ мережі замикання мусить бути ТОЧНИМ
   (вхід рівно гаусів ⟹ ЦГТ не потрібна) — похибка на рівні МК-шуму істини.
P3 МАСШТАБУВАННЯ ІСТИНИ: подвоєння N істини не мусить рухати вирок.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
 (а) P2 пройде точно (шанс 90%) — це чиста алгебра;
 (б) нахил буде −0.5 ± 0.1 (шанс 55%). Якщо так, ЗВЧ підтверджено як
     головний механізм і межа відома;
 (в) 🚨 але я допускаю НАСИЧЕННЯ (шанс 35%): похибка перестане падати, бо
     негаусовість накопичується за 32 шари і живе в КОПУЛІ, а не в
     маргіналях (С88), а копулярна похибка може НЕ падати з `n`.
     Якщо так — це найцікавіше, бо означає, що збільшення «великого числа»
     задачу НЕ рятує в принципі.
 (г) МК має падати як 1/√N_семплів і НЕ залежати від `n` взагалі (шанс 70%).
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 4
L = 32
NMC = 5432
NTRUTH = 1 << 18
DT = torch.float32
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327
WIDTHS = [32, 64, 128, 256, 512, 1024]


def mknet(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    s = (2.0 / n) ** 0.5
    return [torch.randn(n, n, generator=g, device=DEV, dtype=DT) * s
            for _ in range(L)]


def closure(ws, n, depth=L):
    """Наш метод: m,C через ЦГТ + точна гаусова E[ReLU]; позадіагональ —
    перший член ряду Прайса `Cov(a_j,a_k) ≈ Φ_j Φ_k C_h[j,k]`."""
    m = torch.zeros(n, dtype=torch.float64, device=DEV)
    C = torch.eye(n, dtype=torch.float64, device=DEV)
    for i in range(depth):
        w = ws[i].double()
        mh = m @ w
        Ch = w.T @ (C @ w)
        d = torch.diagonal(Ch).clamp(min=1e-30)
        s = d.sqrt()
        t = mh / s
        P = 0.5 * (1.0 + torch.erf(t / SQ2))
        f = torch.exp(-0.5 * t * t) * INV_SQ2PI
        ma = s * (t * P + f)
        if i == depth - 1:
            return ma
        va = ((mh * mh + d) * P + mh * s * f - ma * ma).clamp(min=1e-30)
        Ca = Ch * torch.outer(P, P)
        C = Ca + torch.diag(va - torch.diagonal(Ca))
        m = ma


def fwd_mean(ws, n, N, seed, anti=True, whiten=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    if anti:
        z = torch.randn(N // 2, n, generator=g, device=DEV, dtype=DT)
        z = torch.cat([z, -z], 0)
    else:
        z = torch.randn(N, n, generator=g, device=DEV, dtype=DT)
    if whiten and z.shape[0] > n:
        cov = (z.T @ z).double() / z.shape[0]
        z = z @ torch.linalg.inv(torch.linalg.cholesky(cov)).T.to(DT)
    x = z
    for i in range(L):
        h = x @ ws[i]
        x = torch.relu(h) if i < L - 1 else h
    return torch.relu(h).mean(0).double()


def truth(ws, n, seed, depth=L):
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    done = 0
    B = max(1 << 10, min(1 << 13, (1 << 22) // n))
    while done < NTRUTH:
        b = min(B, NTRUTH - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(depth):
            h = x @ ws[i]
            if i == depth - 1:
                acc += torch.relu(h).sum(0).double()
                break
            x = torch.relu(h)
        done += b
    return acc / NTRUTH


def rel(a, b):
    return float(((a - b) ** 2).sum() / (b * b).sum()) ** 0.5


print(f"{DEV} · глибина {L} · мереж на ширину {NNET} · МК {NMC} · істина {NTRUTH}\n")
print(f"{'ширина':>7s} {'замикання':>12s} {'МК(5432)':>11s} {'відношення':>11s}"
      f" {'ЗАМ×√n':>10s} {'L=1 замик.':>12s}")
rows = []
t0 = time.time()
for n in WIDTHS:
    ec, em, e1 = [], [], []
    for q in range(NNET):
        ws = mknet(n, 24000 + 7 * q + n)
        tr = truth(ws, n, 5000 + q)
        ec.append(rel(closure(ws, n), tr))
        em.append(rel(fwd_mean(ws, n, NMC, 6000 + q), tr))
        tr1 = truth(ws, n, 5000 + q, 1)
        e1.append(rel(closure(ws, n, 1), tr1))
    c, m1, o = float(np.mean(ec)), float(np.mean(em)), float(np.mean(e1))
    rows.append((n, c, m1, o))
    print(f"{n:>7d} {100*c:>11.4f}% {100*m1:>10.4f}% {c/m1:>11.3f}"
          f" {100*c*np.sqrt(n):>9.3f} {100*o:>11.4f}%  ({time.time()-t0:.0f} с)",
          flush=True)

ns = np.array([r[0] for r in rows], float)
cs = np.array([r[1] for r in rows])
ms = np.array([r[2] for r in rows])
kc = np.polyfit(np.log(ns), np.log(cs), 1)[0]
km = np.polyfit(np.log(ns), np.log(ms), 1)[0]
print(f"\n⛔ P2 ГЛИБИНА 1 (вхід ТОЧНО гаусів ⟹ ЦГТ не потрібна):")
print(f"   середня похибка {100*np.mean([r[3] for r in rows]):.4f}%  "
      f"{'✅ замикання точне' if np.mean([r[3] for r in rows]) < 0.004 else '⛔ НЕ ТОЧНЕ'}")
print(f"\n⛔ P1 ВІДТВОРЕННЯ при n=256: замикання {100*cs[ns==256][0]:.4f}% "
      f"(робочий рівень ~0.9%)  МК {100*ms[ns==256][0]:.4f}% (~0.24–0.34%)")
print(f"\n📈 ЛОГ-НАХИЛИ ПО ШИРИНІ")
print(f"   замикання  n^({kc:+.3f})   "
      f"{'✅ ЗВЧ: −0.5' if abs(kc+0.5) < 0.12 else ('🚨 НАСИЧЕННЯ' if kc > -0.25 else '⚠️ швидше за ЗВЧ')}")
print(f"   МК         n^({km:+.3f})   (очікування: ~0, МК не залежить від n)")
print(f"\n📣 ВИРОК")
print(f"   замикання × √n: " + " · ".join(f"{100*c*np.sqrt(n):.2f}" for n, c in zip(ns, cs)))
print(f"   ⟹ {'константа СТОЇТЬ ⟹ це РІВНО закон 1/√n' if np.std(cs*np.sqrt(ns))/np.mean(cs*np.sqrt(ns)) < 0.18 else 'константа ПЛИВЕ ⟹ не чистий 1/√n'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/width90.npz",
         ns=ns, clo=cs, mc=ms, one=np.array([r[3] for r in rows]))
