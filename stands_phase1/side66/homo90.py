"""homo90 — ЧОМУ ПЛОМБА P1 ВПАЛА: ПОРУШЕННЯ ОДНОРІДНОСТІ ЧИ FP32?

`cone90` перевіряв `патерн(r·u) == патерн(u)` через `.all()` — бінарно, і
впав. Але 0.01 і 100.0 НЕ ПРЕДСТАВНІ ТОЧНО у fp32 ⟹ `u*0.01` — це трохи
ІНШИЙ напрямок (відносний зсув ~6e-8 на компоненту), і нейрон, що лежить
ближче за це до границі, чесно перемикається. Тоді провал — не про
однорідність, а про точність.

РОЗРІЗНЯЮ ДВА ПОЯСНЕННЯ:
  A. МАСШТАБ ЯК СТЕПІНЬ ДВІЙКИ (r = 2,4,...,2^20) — множення у fp32 міняє
     ЛИШЕ ЕКСПОНЕНТУ, мантиса недоторкана ⟹ напрямок ТОЧНО той самий.
     Якщо однорідність справжня, збіг мусить бути 100.0000% біт-у-біт.
  B. МАСШТАБ НЕ-СТЕПІНЬ (r = 0.01, 100, 3.7) — напрямок зсувається на ~1e-7
     ⟹ частка розбіжних мусить бути МАЛЕНЬКОЮ і, головне, вони мусять
     лежати БІЛЯ ГРАНИЦІ (мінімальний |t| серед нейронів — крихітний).

ПЕРЕДБАЧЕННЯ: A дасть рівно 100%, B — 99.9х% з розбіжностями лише на межі.
Якщо A дасть <100%, однорідність СПРАВДІ порушена і картинку треба викидати.
"""
import os
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
ws = [torch.as_tensor(np.ascontiguousarray(W[0, i]).astype(np.float32),
                      device=DEV) for i in range(L)]
NS = 1 << 14


def run(x):
    """Патерн (bool за всіма шарами) + мінімальний |t| = запас до границі."""
    pats, mind = [], torch.full((x.shape[0],), 1e30, device=DEV)
    for i in range(L):
        h = x @ ws[i]
        pats.append(h > 0)
        s = h.abs() / torch.clamp(h.abs().mean(1, keepdim=True), min=1e-30)
        mind = torch.minimum(mind, s.min(1).values)
        x = torch.relu(h)
    return torch.cat([p.reshape(x.shape[0], -1) for p in pats], 1), mind


g = torch.Generator(device=DEV).manual_seed(11)
u = torch.randn(NS, n, generator=g, device=DEV, dtype=torch.float32)
u = u / u.norm(dim=1, keepdim=True)
ref, mind = run(u)

print(f"{DEV} · {NS} напрямків · {L} шарів × {n} нейронів = {L*n} бітів патерну\n")
print("A. МАСШТАБ = СТЕПІНЬ ДВІЙКИ (мантиса недоторкана)")
print(f"{'r':>12s} {'напрямків із ІДЕНТИЧНИМ патерном':>34s} {'бітів різних':>14s}")
for e in (-20, -10, -1, 1, 10, 20, 40):
    r = float(2.0 ** e)
    p, _ = run(u * r)
    same = float((p == ref).all(1).double().mean())
    bits = float((p != ref).double().sum())
    print(f"{r:>12.3e} {100*same:>33.6f}% {bits:>14.0f}")

print("\nB. МАСШТАБ = НЕ-СТЕПІНЬ (напрямок зсувається на ~1e-7)")
print(f"{'r':>12s} {'ідентичних':>13s} {'бітів різних':>13s} "
      f"{'мед. запас у розбіжних':>24s} {'у решти':>10s}")
for r in (0.01, 3.7, 100.0, 1e4):
    p, _ = run(u * r)
    eq = (p == ref).all(1)
    bits = float((p != ref).double().sum())
    md = float(mind[~eq].median()) if int((~eq).sum()) else float("nan")
    mo = float(mind[eq].median()) if int(eq.sum()) else float("nan")
    print(f"{r:>12.3e} {100*float(eq.double().mean()):>12.6f}% {bits:>13.0f}"
          f" {md:>24.3e} {mo:>10.3e}")

print("\n📣 ВИРОК:")
p2, _ = run(u * 2.0 ** 20)
a_ok = bool((p2 == ref).all())
print(f"   A (степінь двійки, r=2^20): "
      f"{'✅ 100% БІТ-У-БІТ ⟹ ОДНОРІДНІСТЬ ТОЧНА' if a_ok else '⛔ ОДНОРІДНІСТЬ ПОРУШЕНА'}")
print("   ⟹ провал P1 у cone90 був артефактом fp32-подання 0.01 і 100.0,"
      if a_ok else "   ⟹ картинку конусів будувати НЕ МОЖНА.")
if a_ok:
    print("     а не властивістю мережі. Картинка конусів ДІЙСНА.")
