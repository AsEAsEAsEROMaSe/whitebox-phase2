"""frozen89 — СТРУКТУРА, ЩО ПОВТОРЮЄТЬСЯ: СКІЛЬКИ НЕЙРОНІВ ДЕТЕРМІНОВАНІ?

════════════════ ІДЕЯ (ЮЗЕР) ════════════════
«ваги одні ⟹ в усіх семплах має бути та сама структура, що повторюється».
Математично: мережа без біасів КУСКОВО-ЛІНІЙНА — простір `z` розбитий на
конуси, і всередині конуса `f(z) = (Π W_k·diag(gate_k))·z` РІВНО лінійна.
«Одне й те саме» = патерн активації. МК щоразу заново женe семпл крізь 32
шари, наче структура невідома, хоч вона ОДНА на всі семпли.

Підстава, що структура вироджується з глибиною (`tprof88`):
    RMS|t| = 0.002 → 0.680 → 1.898 → 2.674 → **3.378**
а `Φ(3.378) = 0.9996` ⟹ нейрон активний у 99.96% семплів. Якщо так, то
на глибині мережа МАЙЖЕ ЛІНІЙНА, і ланцюг лінійних шарів згортається в
ОДНУ матрицю, перемножену раз наперед, замість 32 матмулів на семпл.

════════════════ ЩО МІРЯЄМО ════════════════
1. Розподіл `p_j = P(нейрон j активний)` пошарово: яка частка нейронів
   «майже детермінована» (`p>1−ε` або `p<ε`).
2. Скільки семплів мають ХОЧ ОДИН перемикач серед цих нейронів — бо саме
   вони ламають злиття шарів.
3. НАЙГОЛОВНІШЕ: якщо зліпити хвіст `[l..L]` в одну матрицю `W_eff`
   (заморозивши маски на найімовірніших), яка буде похибка виходу і скільки
   флопів це економить.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
🚨 Проти: С88 закрила «здешевлення семпла виморожуванням нейронів
(коваріація йде як √V)». Але там морозили заради КОВАРІАЦІЇ, а тут — заради
ЗЛИТТЯ ЛІНІЙНИХ ШАРІВ, інший об'єкт.
Ставлю: частка «майже детермінованих» на глибині буде 85…95% (шанс 60%),
АЛЕ частка семплів БЕЗ жодного перемикача впаде експоненційно з числом
нейронів (256 нейронів × 0.04% = 10% шансу перемикача на шар ⟹ на 12 шарах
майже кожен семпл матиме перемикач) ⟹ пряме злиття дасть велику похибку
(шанс 65%). Живим може лишитись гібрид: злити хвіст і ДОДАТИ поправку лише
від перемикачів, яких мало.
🎯 Шанс, що вийде здешевлення ≥×2 при похибці ≤0.25%: **20%**.
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
NS = 1 << 14
DT = torch.float32

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · N={NS}")


def act_stats(ws, seed):
    """p_j пошарово + частка семплів, де патерн збігається з наймовірнішим."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(NS, n, generator=g, device=DEV, dtype=DT)
    x = torch.cat([z, -z], 0)
    ps, masks = [], []
    for i in range(L - 1):
        p = x @ ws[i]
        mk = (p > 0)
        ps.append(mk.to(DT).mean(0))
        masks.append(mk)
        x = torch.clamp(p, min=0.0)
    return ps, masks


print("=" * 86)
print("A) ЧАСТКА «МАЙЖЕ ДЕТЕРМІНОВАНИХ» НЕЙРОНІВ ПО ГЛИБИНІ")
print("=" * 86)
print(f"{'шар':>5s} {'p>0.999':>9s} {'p<0.001':>9s} {'ДЕТЕРМ.разом':>13s} "
      f"{'граничні |p-.5|<.4':>19s} {'сер.|t|-екв':>12s}")
allp, allm, allw = [], [], []
for q in range(min(NNET, 4)):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    ps, masks = act_stats(ws, 4242 + q)
    allp.append(ps); allm.append(masks); allw.append(ws)
ps0 = allp[0]
for i in (0, 4, 8, 12, 16, 20, 24, 28, 30):
    p = torch.stack([a[i] for a in allp]).mean(0)
    hi = float((p > 0.999).to(DT).mean())
    lo = float((p < 0.001).to(DT).mean())
    mid = float(((p - 0.5).abs() < 0.4).to(DT).mean())
    teq = float(torch.erfinv(2 * torch.clamp(p, 1e-6, 1 - 1e-6) - 1).abs().mean() * 1.4142)
    print(f"{i:>5d} {100 * hi:>8.2f}% {100 * lo:>8.2f}% {100 * (hi + lo):>12.2f}%"
          f" {100 * mid:>18.2f}% {teq:>12.3f}")

print()
print("=" * 86)
print("B) ЧИ МОЖНА ЗЛИТИ ХВІСТ У ОДНУ МАТРИЦЮ? (частка семплів без перемикачів)")
print("=" * 86)
print(f"{'хвіст з шару':>13s} {'шарів':>7s} {'семплів БЕЗ перемикача':>24s}")
for l in (28, 24, 20, 16, 12, 8):
    fr = []
    for q in range(len(allm)):
        p = allp[q]
        ok = torch.ones(allm[q][0].shape[0], dtype=torch.bool, device=DEV)
        for i in range(l, L - 1):
            dom = (p[i] > 0.5)
            ok &= (allm[q][i] == dom.unsqueeze(0)).all(1)
        fr.append(float(ok.to(DT).mean()))
    print(f"{l:>13d} {L - l:>7d} {100 * np.mean(fr):>23.4f}%")

print()
print("=" * 86)
print("C) ГОЛОВНЕ: ПОХИБКА ВИХОДУ, ЯКЩО ХВІСТ ЗЛИТИ В `W_eff`")
print("=" * 86)
print(f"{'хвіст з шару':>13s} {'економія флопів':>17s} {'віднос.похибка виходу':>23s}")
for l in (30, 28, 26, 24, 20, 16):
    errs = []
    for q in range(len(allw)):
        ws, p = allw[q], allp[q]
        g = torch.Generator(device=DEV).manual_seed(9090 + q)
        z = torch.randn(NS, n, generator=g, device=DEV, dtype=DT)
        x = torch.cat([z, -z], 0)
        for i in range(l):
            x = torch.clamp(x @ ws[i], min=0.0)
        # ТОЧНО
        xe = x.clone()
        for i in range(l, L):
            h = xe @ ws[i]
            xe = torch.clamp(h, min=0.0) if i < L - 1 else h
        exact = torch.clamp(xe, min=0.0).mean(0)
        # ЗЛИТО: маски заморожені на наймовірніших
        Weff = torch.eye(n, dtype=DT, device=DEV)
        for i in range(l, L):
            Weff = Weff @ ws[i]
            if i < L - 1:
                Weff = Weff * (p[i] > 0.5).to(DT).unsqueeze(0)
        approx = torch.clamp(x @ Weff, min=0.0).mean(0)
        errs.append(float(((approx - exact) ** 2).mean() / (exact ** 2).mean()) ** 0.5)
    save = (L - l) / L
    print(f"{l:>13d} {100 * save:>16.1f}% {100 * np.mean(errs):>22.4f}%")
print("\n  (для довідки: повна МК має похибку 0.187–0.257%, замикання 0.935%)")
