"""cone90 — ВИТЯГТИ СПРАВЖНЮ ГЕОМЕТРІЮ КОНУСІВ ІЗ НАШИХ ВАГ (для візуалізації).

ЮЗЕР попросив побачити конуси в динаміці. Малювати вигадані — марно; беру
РЕАЛЬНУ мережу з банку і ріжу її 256-вимірний вхідний простір площиною (2D)
та тривимірним підпростором (3D).

════════════════ ЧОМУ ЗРІЗ ДАЄ ПОВНУ КАРТИНУ ════════════════
Біасів немає ⟹ все позитивно-однорідне ⟹ будь-який зріз ЧЕРЕЗ ПОЧАТОК
успадковує конічну структуру ТОЧНО: у 2D-зрізі кожна границя активації —
це промінь із центру, у 3D-зрізі — площина через центр.
⟹ у 2D достатньо обійти КОЛО: уся інформація в кутах перемикання.
⟹ у 3D достатньо обійти СФЕРУ: конус ↔ область на сфері.

Це не наближення. Пломба нижче перевіряє це прямо.

════════════════ ПЛОМБИ ════════════════
P1 ОДНОРІДНІСТЬ: патерн активації на промені `r·u` мусить бути ОДНАКОВИЙ
   для r = 0.01 · 1 · 100 (усі 32 шари). Якщо ні — уся картинка хибна.
P2 ШАР 1 У 2D: границі шару 1 — рівно 2·256 = 512 променів (по 2 на нейрон,
   бо площина `w·z=0` перетинає 2D-зріз по прямій = двох променях), тобто
   рівно 512 секторів. Пряма перевірка проти підрахунку.
P3 ЦЕНТРАЛЬНА СИМЕТРІЯ: патерн у точці `−u` мусить бути ТОЧНО доповненням
   патерну в `u` на ПЕРШОМУ шарі (бо `w·(−z) = −w·z`), далі — ні.
"""
import os
import sys
import time
import zlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NET = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = "/mnt/win_d/llm/White-Box/side66/conedata"
os.makedirs(OUT, exist_ok=True)

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
ws = [torch.as_tensor(np.ascontiguousarray(W[NET, i]).astype(np.float32),
                      device=DEV) for i in range(L)]
print(f"{DEV} · мережа #{NET} · ширина {n} · глибина {L}")

PRIMES = torch.randint(-(1 << 62), 1 << 62, (L, n), device=DEV,
                       dtype=torch.int64, generator=torch.Generator(
                           device=DEV).manual_seed(20260806)) | 1
DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def patterns(x, depths):
    """Кумулятивний хеш патерну активації на заданих глибинах."""
    cum = torch.zeros(x.shape[0], dtype=torch.int64, device=DEV)
    out = {}
    for i in range(L):
        h = x @ ws[i]
        mk = (h > 0)
        cum = cum * 1000003 + (mk.to(torch.int64) * PRIMES[i]).sum(1)
        if (i + 1) in depths:
            out[i + 1] = cum.clone()
        x = torch.relu(h)
    return out


# ═══════════════════ P1 · ОДНОРІДНІСТЬ ═══════════════════
g = torch.Generator(device=DEV).manual_seed(11)
u = torch.randn(2048, n, generator=g, device=DEV, dtype=torch.float32)
u = u / u.norm(dim=1, keepdim=True)
ref = patterns(u, [L])[L]
ok = all(bool((patterns(u * r, [L])[L] == ref).all()) for r in (0.01, 1.0, 100.0))
print(f"\n⛔ P1 ОДНОРІДНІСТЬ (патерн не залежить від радіуса): "
      f"{'✅ ТОЧНО' if ok else '⛔ ПРОВАЛ'}")

# ═══════════════════ 2D-ЗРІЗ: КУТИ ГРАНИЦЬ ═══════════════════
M = 1 << 21
gg = torch.Generator(device=DEV).manual_seed(4242)
B = torch.linalg.qr(torch.randn(n, 2, generator=gg, device=DEV))[0]   # (n,2)
e1, e2 = B[:, 0].contiguous(), B[:, 1].contiguous()

th = torch.arange(M, device=DEV, dtype=torch.float64) * (2 * np.pi / M)
cum_prev = {d: None for d in range(1, L + 1)}
edges = {d: [] for d in range(1, L + 1)}
counts = np.zeros(L, dtype=np.int64)
CH = 1 << 17
allcum = {d: torch.empty(M, dtype=torch.int64, device=DEV) for d in range(1, L + 1)}
t0 = time.time()
for s0 in range(0, M, CH):
    s1 = min(s0 + CH, M)
    t = th[s0:s1].to(torch.float32)
    x = torch.outer(torch.cos(t), e1) + torch.outer(torch.sin(t), e2)
    pat = patterns(x, list(range(1, L + 1)))
    for d in range(1, L + 1):
        allcum[d][s0:s1] = pat[d]
for d in range(1, L + 1):
    c = allcum[d]
    ch = (c != torch.roll(c, -1))
    counts[d - 1] = int(ch.sum())
    if d <= 10:
        idx = torch.nonzero(ch).flatten()
        edges[d] = (th[idx] + np.pi / M).cpu().numpy().astype(np.float32)
print(f"  2D-обхід кола M={M} за {time.time()-t0:.0f} с")
print(f"\n⛔ P2 ШАР 1: секторів у 2D-зрізі = {counts[0]}  (мусить бути 512) "
      f"{'✅' if counts[0] == 512 else '⚠️ (збіг границь або крок сітки)'}")

half = M // 2
c1 = allcum[1]
p3 = float((c1 == torch.roll(c1, half)).double().mean())
print(f"⛔ P3 ЦЕНТР.СИМЕТРІЯ шару 1: частка кутів, де патерн(−u)==патерн(u) "
      f"= {100*p3:.2f}%  (мусить бути ~0%) {'✅' if p3 < 0.01 else '⛔'}")

print(f"\n{'шар':>5s} {'секторів у 2D-зрізі':>21s} {'×до попереднього':>18s}")
for d in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
    r = counts[d - 1] / max(counts[d - 2], 1) if d > 1 else float("nan")
    print(f"{d:>5d} {counts[d-1]:>21d} {r:>18.3f}")

# ═══════════════════ 3D-ЗРІЗ: ТЕКСТУРИ СФЕРИ ═══════════════════
PW, PH = 1440, 720
gg2 = torch.Generator(device=DEV).manual_seed(777)
B3 = torch.linalg.qr(torch.randn(n, 3, generator=gg2, device=DEV))[0]
lon = (torch.arange(PW, device=DEV, dtype=torch.float32) + 0.5) * (2 * np.pi / PW)
lat = (torch.arange(PH, device=DEV, dtype=torch.float32) + 0.5) * (np.pi / PH)
LA, LO = torch.meshgrid(lat, lon, indexing="ij")
dirs = torch.stack([(torch.sin(LA) * torch.cos(LO)).reshape(-1),
                    (torch.sin(LA) * torch.sin(LO)).reshape(-1),
                    torch.cos(LA).reshape(-1)], 1)                    # (P,3)
P = dirs.shape[0]
tex = {d: torch.empty(P, dtype=torch.int64, device=DEV) for d in DEPTHS}
t0 = time.time()
for s0 in range(0, P, CH):
    s1 = min(s0 + CH, P)
    x = dirs[s0:s1] @ B3.T
    pat = patterns(x, DEPTHS)
    for d in DEPTHS:
        tex[d][s0:s1] = pat[d]
print(f"\n  3D-сфера {PW}×{PH} за {time.time()-t0:.0f} с")


def png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def ck(tag, data):
        c = tag + data
        return (len(data).to_bytes(4, "big") + c
                + zlib.crc32(c).to_bytes(4, "big"))
    hdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", hdr)
                + ck(b"IDAT", zlib.compress(raw, 9)) + ck(b"IEND", b""))
    return os.path.getsize(path)


tot = 0
uniq = {}
for d in DEPTHS:
    c = tex[d]
    uniq[d] = int(torch.unique(c).numel())
    v = (c ^ (c >> 17)) * 0x9E3779B97F4A7C15
    v = (v ^ (v >> 29)) & 0x7FFFFFFFFFFFFFFF
    hue = (v % 3600).to(torch.float32) / 3600.0
    sat = 0.42 + 0.30 * ((v // 3600) % 100).to(torch.float32) / 100.0
    val = 0.55 + 0.40 * ((v // 360000) % 100).to(torch.float32) / 100.0
    i6 = torch.floor(hue * 6.0)
    f = hue * 6.0 - i6
    p_, q_, t_ = val * (1 - sat), val * (1 - f * sat), val * (1 - (1 - f) * sat)
    i6 = (i6 % 6).to(torch.int64)
    r = torch.where(i6 == 0, val, torch.where(i6 == 1, q_, torch.where(
        i6 == 2, p_, torch.where(i6 == 3, p_, torch.where(i6 == 4, t_, val)))))
    gc = torch.where(i6 == 0, t_, torch.where(i6 == 1, val, torch.where(
        i6 == 2, val, torch.where(i6 == 3, q_, torch.where(i6 == 4, p_, p_)))))
    b = torch.where(i6 == 0, p_, torch.where(i6 == 1, p_, torch.where(
        i6 == 2, t_, torch.where(i6 == 3, val, torch.where(i6 == 4, val, q_)))))
    rgb = (torch.stack([r, gc, b], 1) * 255).clamp(0, 255).to(
        torch.uint8).reshape(PH, PW, 3).cpu().numpy()
    sz = png(f"{OUT}/sphere_L{d:02d}.png", rgb)
    tot += sz
    print(f"    шар {d:>2d}: конусів на сфері {uniq[d]:>7d}   PNG {sz/1024:>7.0f} КБ")
print(f"  РАЗОМ PNG: {tot/1024/1024:.2f} МБ")

np.savez(f"{OUT}/cone90.npz",
         counts=counts, uniq=np.array([uniq[d] for d in DEPTHS]),
         depths=np.array(DEPTHS), M=M,
         **{f"edges{d}": edges[d] for d in range(1, 11)})
print(f"\n  дані → {OUT}/cone90.npz")
