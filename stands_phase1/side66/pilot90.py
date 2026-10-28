"""pilot90 — СКАН ПО РОЗМІРУ ПІЛОТА. Найдешевша можлива зміна: одна константа.

Пілот коштує `2·PILOT·повний_семпл`; при 128 парах це 6.47% ціни семпла.
Менший пілот: дешевший І ріже АГРЕСИВНІШЕ (менше шансів, що нейрон спрацює),
але ризикує відрізати живих ⟹ зміщення. Питання суто емпіричне.
ПЛОМБА: PILOT=128 мусить відтворити сабміт (0.2446%, дисперсія ~90%).
ПЕРЕДБАЧЕННЯ: оптимум при 24..48 пар, виграш ×1.03..1.08 (шанс 55%).
"""
import os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, torch
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 60
MREP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
NMC, NTRUTH, DT = 5432, 1 << 19, torch.float32
W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
FULL = 2.0 * L * n * n
PILOTS = [8, 16, 24, 32, 48, 64, 128, 192]
print(f"{DEV} · мереж {NNET} · повний семпл {FULL:.3e}\n")

def alive_of(ws, P, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    zp = torch.randn(P, n, generator=g, device=DEV, dtype=DT)
    a = torch.cat([zp, -zp], 0); al = []
    for i in range(L):
        h = a @ ws[i]
        if i < L - 1: al.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    return al

def cut(ws, al):
    out, fl = [], 0.0
    prev = torch.ones(n, dtype=torch.bool, device=DEV)
    for i in range(L):
        k = al[i] if i < L - 1 else torch.ones(n, dtype=torch.bool, device=DEV)
        out.append(ws[i][prev][:, k].contiguous())
        fl += 2.0 * int(prev.sum()) * int(k.sum()); prev = k
    return out, fl

def fwd(ws, z):
    x = z
    for i in range(len(ws)):
        h = x @ ws[i]; x = torch.relu(h) if i < len(ws) - 1 else h
    return torch.relu(h)

def mc(ws, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(NMC // 2, n, generator=g, device=DEV, dtype=DT)
    z = torch.cat([z, -z], 0)
    cov = (z.T @ z).double() / z.shape[0]
    return fwd(ws, z @ torch.linalg.inv(torch.linalg.cholesky(cov)).T.to(DT)).mean(0).double()

def truth(ws, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV); done = 0
    while done < NTRUTH:
        b = min(1 << 13, NTRUTH - done)
        acc += fwd(ws, torch.randn(b, n, generator=g, device=DEV, dtype=DT)).sum(0).double()
        done += b
    return acc / NTRUTH

V = {p: [] for p in PILOTS}; B = {p: [] for p in PILOTS}; F = {p: [] for p in PILOTS}
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32), device=DEV) for i in range(L)]
    tr = truth(ws, 7000 + q); den = float((tr * tr).sum())
    for P in PILOTS:
        pw, fl = cut(ws, alive_of(ws, P, 4242 + q))
        F[P].append(fl + 2.0 * P * FULL / NMC)
        acc, ms = 0.0, torch.zeros(n, dtype=torch.float64, device=DEV)
        for r in range(MREP):
            p = mc(pw, 808000 + 977 * q + r)
            acc += float(((p - tr) ** 2).sum() / den); ms += p
        V[P].append(acc / MREP); B[P].append(float(((ms / MREP - tr) ** 2).sum() / den))
    if q % 15 == 0: print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

f0, t128 = np.mean(F[128]), np.mean(V[128]); b0 = np.mean(B[128]); v0 = t128 - b0
print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 86)
print(f"СКАН ПО РОЗМІРУ ПІЛОТА · {NNET} мереж × {MREP} повторів")
print("=" * 86)
print(f"{'пар':>6s} {'ціна/семпл':>12s} {'× до 128':>9s} {'похибка':>10s} {'зміщення':>10s} {'БАЛ ×':>9s}")
best = (128, 1.0)
for P in PILOTS:
    f = np.mean(F[P]); tt = np.mean(V[P]); bb = np.mean(B[P]); vv = tt - bb
    sc = f / f0; g = (b0 + v0) / max(bb + vv * sc, 1e-30)
    print(f"{P:>6d} {f:>12.4e} {sc:>9.4f} {100*tt**0.5:>9.4f}% {100*bb**0.5:>9.4f}% {g:>9.4f}")
    if g > best[1]: best = (P, g)
print(f"\n⛔ ПЛОМБА PILOT=128: похибка {100*t128**0.5:.4f}% (сабміт 0.2446%) "
      f"дисперсія {100*v0/(b0+v0):.1f}%")
print(f"\n📣 ВИРОК: PILOT={best[0]} → ×{best[1]:.4f} ⟹ {1.86762e-07/best[1]:.4e}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/pilot90.npz",
         P=np.array(PILOTS), V=np.array([V[p] for p in PILOTS]),
         B=np.array([B[p] for p in PILOTS]), F=np.array([F[p] for p in PILOTS]))
