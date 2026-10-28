"""conv90 — ЧИ ЗБІГЛАСЯ КІЛЬКІСТЬ СЕКТОРІВ У 2D-ЗРІЗІ?

ДВІ ТРИВОГИ З `cone90b`:
  (1) шар 1 дав 536 секторів, а ТЕОРІЯ каже РІВНО 512 (256 нейронів, кожен
      дає пряму через центр = 2 промені). Зайві 24 — підозра на «дрижання»:
      біля границі fp32 перемикає знак кілька разів поспіль;
  (2) при M 2^21→2^23 глибина стрибнула 23993→37883 (×1.58) ⟹ грубою сіткою
      ми ПРОПУСКАЛИ границі, і число не збіжне.

Обидва ефекти діють у РІЗНІ боки, тож без розділення жодному числу вірити не
можна. Розділяю: рахую при M = 2^20…2^24 і ОКРЕМО — з дедуплікацією
(перемикання ближчі за `KDUP` кроків сітки вважаю одним).

ПЛОМБА: після дедуплікації шар 1 мусить дати РІВНО 512 при БУДЬ-ЯКОМУ M.
Якщо так — дедуплікація правильна, і можна дивитись, чи збіглася глибина.
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
PR = torch.randint(-(1 << 62), 1 << 62, (L, n), device=DEV, dtype=torch.int64,
                   generator=torch.Generator(device=DEV).manual_seed(20260806)) | 1
CH = 1 << 17
D2 = [1, 2, 4, 8, 16, 32]
KDUP = 3

gg = torch.Generator(device=DEV).manual_seed(4242)
B = torch.linalg.qr(torch.randn(n, 2, generator=gg, device=DEV))[0]
e1, e2 = B[:, 0].contiguous(), B[:, 1].contiguous()


def sweep(M):
    acc = {d: torch.empty(M, dtype=torch.int64, device=DEV) for d in D2}
    for s0 in range(0, M, CH):
        s1 = min(s0 + CH, M)
        t = (torch.arange(s0, s1, device=DEV, dtype=torch.float64)
             * (2 * np.pi / M)).to(torch.float32)
        x = torch.outer(torch.cos(t), e1) + torch.outer(torch.sin(t), e2)
        cum = torch.zeros(s1 - s0, dtype=torch.int64, device=DEV)
        for i in range(L):
            h = x @ ws[i]
            cum = cum * 1000003 + ((h > 0).to(torch.int64) * PR[i]).sum(1)
            if (i + 1) in D2:
                acc[i + 1][s0:s1] = cum
            x = torch.relu(h)
    raw, ded = {}, {}
    for d in D2:
        c = acc[d]
        idx = torch.nonzero(c != torch.roll(c, -1)).flatten()
        raw[d] = int(idx.numel())
        if idx.numel() > 1:
            gap = torch.diff(idx, prepend=idx[-1:] - M)
            ded[d] = int((gap > KDUP).sum())
        else:
            ded[d] = raw[d]
    return raw, ded


print(f"{DEV} · дедуплікація: перемикання ближчі за {KDUP} кроків = одне\n")
print(f"{'M':>10s} " + " ".join(f"{'L'+str(d):>16s}" for d in D2))
print(f"{'':>10s} " + " ".join(f"{'сирих/дедуп':>16s}" for d in D2))
res = {}
for e in (20, 21, 22, 23, 24):
    M = 1 << e
    raw, ded = sweep(M)
    res[M] = ded
    print(f"2^{e:<8d} " + " ".join(f"{raw[d]:>7d}/{ded[d]:<8d}" for d in D2), flush=True)

print(f"\n⛔ ПЛОМБА ШАРУ 1 (теорія: РІВНО 512):")
for M in res:
    ok = "✅" if res[M][1] == 512 else "⛔"
    print(f"   M=2^{int(np.log2(M)):<2d}  дедуп = {res[M][1]:>5d}  {ok}")

print(f"\n📈 ЗБІЖНІСТЬ ГЛИБИНИ (дедуп, відношення до попереднього M):")
Ms = sorted(res)
print(f"{'шар':>5s} " + " ".join(f"{'2^'+str(int(np.log2(M))):>9s}" for M in Ms)
      + f" {'ост.×':>8s}")
for d in D2:
    row = " ".join(f"{res[M][d]:>9d}" for M in Ms)
    r = res[Ms[-1]][d] / max(res[Ms[-2]][d], 1)
    print(f"{d:>5d} {row} {r:>8.3f}")
print("\n  збіглося ⟺ останнє відношення ≈ 1.00; інакше число — НИЖНЯ ОЦІНКА")
