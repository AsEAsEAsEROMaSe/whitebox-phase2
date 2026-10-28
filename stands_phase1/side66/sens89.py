"""sens89 — BACKWARD: ЯК ЗБУРЕННЯ НА ШАРІ `l` ДОХОДИТЬ ДО ВИХОДУ.

════════════════ ЧОМУ ЦЕ НОВА ВІСЬ ════════════════
Уся наша робота (С84–С89) — FORWARD-маргіналі: тягнемо `m, C` від входу до
виходу. Жодного разу ми не питали ЗВОРОТНОГО: наскільки вихід взагалі
ЧУТЛИВИЙ до того, що діється на шарі `l`. Ідею подав ЮЗЕР («вести з кінця,
трійками»), і хоч його формулювання «далі 3 шарів — шум» наш власний вимір
не підтверджує (`hyb89`: похибка замикання росте як L^0.35 аж до 32), сама
вісь — backward — не розвідана.

════════════════ ЩО МІРЯЄМО ════════════════
1. ЗАТУХАННЯ: збурюємо стан на шарі `l` на δ і дивимось `|Δвихід|/|δ|`
   як функцію відстані `L−l`. Якщо падає експоненційно — задача РОЗПАДАЄТЬСЯ
   на локальні блоки, і все, крім останніх кількох шарів, можна рахувати
   грубо (тобто ДЕШЕВО, а бюджет піде в семпли).
2. РАНГ ПЕРЕДАЧІ: скільки НАПРЯМКІВ збурення реально доходить. Якщо
   ефективний ранг якобіана `∂вихід/∂a^(l)` малий, то вихід залежить від
   стану лише через кілька скалярів — і тоді точність потрібна ЛИШЕ в них.
   Це і є форма, у якій здогад ЮЗЕРА може виявитись правильним:
   `tprof88` уже зміряв, що еф. ранг АКТИВАЦІЙ падає 127.6 → 3.1.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
Ставлю до першого числа:
  (а) затухання БУДЕ степеневим, не експоненційним (шанс 65%) — бо He-init
      зберігає норму, а `gate` вирізає рівно половину;
  (б) ЕФЕКТИВНИЙ РАНГ ПЕРЕДАЧІ впаде до 3…10 вже за 8–10 шарів (шанс 55%);
  (в) якщо (б) справдиться — це НАЙЦІННІШЕ, бо означає, що похибка в 250
      напрямках з 256 просто НЕ ДОХОДИТЬ до виходу, а ми досі рівномірно
      витрачали точність на всі.
⚠️ Пастка, яку називаю наперед: малий ранг ЯКОБІАНА не означає, що можна
проєктувати СТАН (це вбило `proj89`: r=128 дало 13.3% похибки). Якобіан
лінеаризує, а `relu` — ні. Тому вирок робитиму лише про те, куди варто
вкладати ТОЧНІСТЬ, а не про те, що можна викидати зі стану.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 8
NS = 8192
DT = torch.float32

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · N={NS} · BACKWARD-чутливість")


def forward_gates(ws, seed):
    """Прогін + збереження масок активації (вони й задають якобіан)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(NS, n, generator=g, device=DEV, dtype=DT)
    x = torch.cat([z, -z], 0)
    gates = []
    for i in range(L - 1):
        p = x @ ws[i]
        m = (p > 0).to(DT)
        gates.append(m)
        x = p * m
    return gates


def jac_transfer(ws, gates, l):
    """Матриця передачі `∂h^(L)/∂a^(l)` (усереднена по семплах).

    Для ReLU без біасів якобіан — це добуток `W_k·diag(gate_k)`, тож
    середня передача = `E_s[ Π_k W_k diag(gate_k^s) ]`. Рахуємо дією на
    одиничні напрямки блоком, у fp32, з усередненням по семплах.
    """
    Vt = torch.eye(n, dtype=DT, device=DEV).unsqueeze(0)      # (1,n,n)
    acc = torch.zeros(n, n, dtype=torch.float64, device=DEV)
    CH = 512
    S = gates[0].shape[0]
    for s0 in range(0, S, CH):
        s1 = min(s0 + CH, S)
        V = Vt.expand(s1 - s0, n, n).clone()                  # (b,n,n)
        for k in range(l, L - 1):
            V = torch.einsum("bij,jk->bik", V, ws[k]) * gates[k][s0:s1].unsqueeze(1)
        V = torch.einsum("bij,jk->bik", V, ws[L - 1])
        acc += V.sum(0).double()
    return (acc / S).to(DT)


print(f"\n{'l':>4s} {'L−l':>5s} {'||J||_F':>11s} {'спад':>8s} "
      f"{'еф.ранг J':>11s} {'топ-1 частка':>13s} {'топ-3':>8s} {'топ-10':>8s}")
rows = []
t0 = time.time()
for q in range(min(NNET, 8)):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    gates = forward_gates(ws, 5150 + q)
    prev = None
    rec = []
    for l in (28, 26, 24, 20, 16, 8, 0):
        J = jac_transfer(ws, gates, l)
        sv = torch.linalg.svdvals(J.double())
        p = (sv / sv.sum())
        er = float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))
        fro = float(J.norm())
        rec.append((l, fro, er, float(p[0]), float(p[:3].sum()), float(p[:10].sum())))
        if q == 0:
            sp = "—" if prev is None else f"{fro / prev:.3f}"
            print(f"{l:>4d} {L - l:>5d} {fro:>11.4e} {sp:>8s} {er:>11.2f}"
                  f" {100 * p[0]:>12.2f}% {100 * p[:3].sum():>7.2f}%"
                  f" {100 * p[:10].sum():>7.2f}%")
            prev = fro
    rows.append(rec)

print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 80)
print(f"СЕРЕДНЄ ПО {len(rows)} МЕРЕЖАХ")
print("=" * 80)
print(f"{'l':>4s} {'L−l':>5s} {'еф.ранг J':>11s} {'топ-1':>9s} {'топ-3':>9s} {'топ-10':>9s}")
arr = np.array(rows)                                          # (nets, points, 6)
for t in range(arr.shape[1]):
    l = int(arr[0, t, 0])
    print(f"{l:>4d} {L - l:>5d} {arr[:, t, 2].mean():>11.2f}"
          f" {100 * arr[:, t, 3].mean():>8.2f}% {100 * arr[:, t, 4].mean():>8.2f}%"
          f" {100 * arr[:, t, 5].mean():>8.2f}%")
print()
print("  ВИСНОВОК ПРО ЗАТУХАННЯ:")
f0 = arr[:, -1, 1].mean()
f28 = arr[:, 0, 1].mean()
print(f"    ||J|| від l=28 до l=0 змінюється в {f0 / f28:.3f}× на {28} шарах")
print(f"    ⟹ на шар {(f0 / f28) ** (1 / 28):.4f}× "
      f"({'ЕКСПОНЕНЦІЙНЕ затухання' if (f0 / f28) ** (1 / 28) < 0.9 else 'НЕМАЄ експоненційного затухання'})")
print("  ВИСНОВОК ПРО РАНГ ПЕРЕДАЧІ:")
print(f"    еф.ранг на l=28 (близько до виходу) = {arr[:, 0, 2].mean():.1f}")
print(f"    еф.ранг на l=0  (весь ланцюг)       = {arr[:, -1, 2].mean():.1f}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/sens89.npz", rows=arr)
