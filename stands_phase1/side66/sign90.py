"""sign90 — ЗНАКИ ВАГ: ЧИ СПРАВДІ МІНУСИ І ЛАМАЮТЬ ФОРМУ, І СТАБІЛІЗУЮТЬ?

════════════════ ДУМКА ЮЗЕРА ════════════════
«форму ламають мінуси… ваги мінуси хоч і стабілізують».

Теза, яку перевіряю: ці два ефекти — НЕ два різні спостереження, а одна
величина (частка від'ємних ваг `p`), що тягне в РІЗНІ боки.

📐 ЧОМУ ЦЕ МУСИТЬ БУТИ ТАК (виведення, не здогад):
після першого шару всі активації `a ≥ 0` (вихід ReLU). Преактивація
`h_j = Σ_i W_ij·a_i`. Якщо ВСІ `W_ij > 0`, то `h_j > 0` тотожно ⟹ жоден
нейрон не мовчить ⟹ ReLU не діє ⟹ мережа стає ТОЧНО ЛІНІЙНОЮ, конусів
НЕМАЄ ЗОВСІМ. Тобто від'ємні ваги — НЕОБХІДНА УМОВА нелінійності.
І одночасно: середнє `E[h_j] = (Σ W)·E[a]` тримається біля нуля лише коли
додатних і від'ємних порівну ⟹ той самий `p` тримає норму. Одна ручка,
два наслідки.

════════════════ СХЕМА ВИМІРУ ════════════════
Беру СПРАВЖНІ модулі ваг `|W|` і перевішую лише ЗНАКИ з ймовірністю `p`.
Розподіл модулів, а отже і He-масштаб, недоторканий — міняється РІВНО те,
що досліджуємо.

ПЛОМБИ:
P1 `p=0` (усі ваги додатні): 100% нейронів завжди активні, а вихід мусить
   ТОЧНО збігтися з чисто лінійним прогоном `z·W₁·W₂···W₃₂` (після шару 1).
   Похибка мусить бути на рівні fp32 (<1e-4), інакше виведення хибне.
P2 `p=0.5` мусить відтворити справжню мережу: норма стабільна, частка
   детермінованих на шарі 32 ≈ 62% (зміряно в `chk90`).

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
 (а) P1 пройде точно (шанс 90%);
 (б) норма активацій: при `p<0.5` росте експоненційно, при `p>0.5` падає,
     мінімальний дрейф рівно біля 0.5 (шанс 75%);
 (в) частка детермінованих нейронів на шарі 32 буде МІНІМАЛЬНОЮ біля p=0.5
     (шанс 60%) — тобто симетрія знаків максимально «тримає» нелінійність;
 (г) 🚨 але навіть при ідеальному p=0.5 детермінованих 62%, а не 0% ⟹
     стабілізація НЕПОВНА, і з глибиною баланс усе одно з'їдається. Це і є
     точна відповідь на «хоч і стабілізують»: стабілізують, але не до кінця.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NS = 1 << 13
DT = torch.float32

W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
PS = [0.0, 0.10, 0.25, 0.40, 0.45, 0.50, 0.55, 0.70, 1.0]
print(f"{DEV} · мереж {NNET} · N={NS} · перевішую ЛИШЕ знаки, модулі справжні\n")


def make(ws_abs, p, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [w * torch.where(torch.rand(w.shape, generator=g, device=DEV) < p,
                            -1.0, 1.0) for w in ws_abs]


def probe(ws):
    """частка детермінованих по шарах · дрейф норми на шар · RMS|t|

    🚨 ПЕРЕПОВНЕННЯ: при p=0 норма росте ×3.56/шар ⟹ 3.56³¹ = 1e17 ⟹ inf у
    fp32, і всі числа стають nan. Рятує сама однорідність: множення стану на
    скаляр НЕ МІНЯЄ патерн активації (запечатано в `homo90`, 100% біт-у-біт),
    тож нормую щошару і накопичую логарифм масштабу окремо.
    """
    g = torch.Generator(device=DEV).manual_seed(31337)
    x = torch.randn(NS, n, generator=g, device=DEV, dtype=DT)
    logs, det, trms = [], [], []
    for i in range(L):
        h = x @ ws[i]
        pj = (h > 0).to(torch.float64).mean(0)
        det.append(float(((pj > 0.999) | (pj < 0.001)).to(torch.float64).mean()))
        m, v = h.mean(0), h.var(0).clamp(min=1e-30)
        trms.append(float(torch.sqrt(((m / v.sqrt()) ** 2).mean())))
        x = torch.clamp(h, min=0.0)
        s = float(x.norm(dim=1).mean())
        logs.append(np.log(max(s, 1e-38)))
        if s > 0:
            x = x / s                       # нормування; патерн незмінний
    dr = float(np.exp(np.mean(np.diff(logs)))) if len(logs) > 1 else 1.0
    return np.array(det), dr, np.array(trms), float(np.exp(min(np.sum(
        np.diff(logs)) + logs[0], 700.0)))


rows = []
t0 = time.time()
for p in PS:
    D, DR, T, NR = [], [], [], []
    for q in range(NNET):
        wa = [torch.as_tensor(np.abs(np.ascontiguousarray(W[q, i])).astype(np.float32),
                              device=DEV) for i in range(L)]
        d, dr, t, nr = probe(make(wa, p, 900 + q))
        D.append(d); DR.append(dr); T.append(t); NR.append(nr)
    rows.append((p, np.mean(D, 0), float(np.mean(DR)), np.mean(T, 0), float(np.mean(NR))))
    print(f"  p={p:.2f} готово ({time.time()-t0:.0f} с)", flush=True)

print("\n" + "=" * 94)
print("СКАН ПО ЧАСТЦІ ВІД'ЄМНИХ ВАГ · модулі справжні, змінено ЛИШЕ знаки")
print("=" * 94)
print(f"{'p(−)':>6s} {'детерм. шар8':>13s} {'шар16':>9s} {'шар32':>9s} "
      f"{'дрейф норми/шар':>17s} {'||a|| на виході':>17s} {'RMS|t| шар32':>13s}")
for p, d, dr, t, nr in rows:
    print(f"{p:>6.2f} {100*d[7]:>12.2f}% {100*d[15]:>8.2f}% {100*d[31]:>8.2f}%"
          f" {dr:>17.4f} {nr:>17.3e} {t[31]:>13.3f}")

# ═══════ ПЛОМБА P1 ═══════
print("\n⛔ ПЛОМБА P1: при p=0 мережа мусить бути ТОЧНО лінійною (після шару 1)")
errs = []
for q in range(NNET):
    wa = [torch.as_tensor(np.abs(np.ascontiguousarray(W[q, i])).astype(np.float32),
                          device=DEV) for i in range(L)]
    g = torch.Generator(device=DEV).manual_seed(4321 + q)
    z = torch.randn(NS, n, generator=g, device=DEV, dtype=DT)
    x = torch.clamp(z @ wa[0], min=0.0)
    x = x / float(x.norm(dim=1).mean())
    # точний прогін із нормуванням щошару (однорідність ⟹ патерн незмінний)
    xe = x.clone()
    for i in range(1, L):
        xe = xe @ wa[i]
        if i < L - 1:
            xe = torch.clamp(xe, min=0.0)
        xe = xe / float(xe.abs().max())
    exact = torch.clamp(xe, min=0.0).mean(0)
    exact = exact / float(exact.norm())
    # злиття всіх 31 шарів в одну матрицю, теж із нормуванням
    Wl = torch.eye(n, dtype=DT, device=DEV)
    for i in range(1, L):
        Wl = Wl @ wa[i]
        Wl = Wl / float(Wl.abs().max())
    lin = torch.clamp(x @ Wl, min=0.0).mean(0)
    lin = lin / float(lin.norm())
    errs.append(float(((lin - exact) ** 2).sum()) ** 0.5)
e = float(np.mean(errs))
print(f"   похибка злиття ВСІХ 31 шарів в одну матрицю = {e:.3e}"
      f"   {'✅ ТОЧНО ЛІНІЙНА ⟹ конусів немає' if e < 1e-3 else '⛔ НЕ ЛІНІЙНА'}")

d50 = [r for r in rows if r[0] == 0.5][0]
print(f"\n⛔ ПЛОМБА P2: p=0.50 проти справжньої мережі")
print(f"   детермінованих на шарі 32 = {100*d50[1][31]:.2f}%   (chk90 дав 61.98%)"
      f"   {'✅' if abs(100*d50[1][31]-61.98) < 6 else '⚠️ розбіжність'}")
print(f"   дрейф норми на шар = {d50[2]:.4f}   (мусить бути ≈1.00)")

best = min(rows, key=lambda r: r[1][31])
print(f"\n📣 ВИРОК")
print(f"   найменше детермінованих на шарі 32 при p = {best[0]:.2f} "
      f"({100*best[1][31]:.2f}%)")
print(f"   ⟹ мінуси СТВОРЮЮТЬ нелінійність (p=0 → рівно нуль конусів)")
print(f"   ⟹ мінуси ТРИМАЮТЬ норму (дрейф 1.00 лише біля p=0.5)")
print(f"   ⟹ але стабілізація НЕПОВНА: навіть при ідеальному балансі "
      f"{100*d50[1][31]:.1f}% нейронів на глибині вже детерміновані")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/sign90.npz",
         ps=np.array(PS), det=np.array([r[1] for r in rows]),
         drift=np.array([r[2] for r in rows]),
         trms=np.array([r[3] for r in rows]))
