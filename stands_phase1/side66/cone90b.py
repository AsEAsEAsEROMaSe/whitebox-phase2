"""cone90b — ДАНІ ДЛЯ ВІЗУАЛІЗАЦІЇ: 2D-кути точно + 3D-сфери з ЗУМОМ.

Однорідність запечатано в `homo90` (100% біт-у-біт на 18 порядках масштабу),
тож зріз через нуль передає структуру ТОЧНО.

🔑 ЩО ПОКАЗАВ ПЕРШИЙ ПРОГІН І ЧОМУ ПОТРІБЕН ЗУМ:
на сфері 1440×720 (1.04e6 пікселів) кількість конусів росте 51298 (шар 1)
→ 708786 (шар 8) → 981411 (шар 32). Тобто вже з шару ~6 конус ДРІБНІШИЙ
ЗА ПІКСЕЛЬ, і картинка вироджується в шум. Це не вада рендера — це і є
головний факт. Щоб побачити структуру на глибині, треба ЗУМ.
"""
import json
import os
import sys
import time
import zlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
OUT = "/mnt/win_d/llm/White-Box/side66/conedata"
os.makedirs(OUT, exist_ok=True)
W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
ws = [torch.as_tensor(np.ascontiguousarray(W[0, i]).astype(np.float32),
                      device=DEV) for i in range(L)]
PR = torch.randint(-(1 << 62), 1 << 62, (L, n), device=DEV, dtype=torch.int64,
                   generator=torch.Generator(device=DEV).manual_seed(20260806)) | 1
CH = 1 << 17
print(f"{DEV} · мережа #0 · {n}×{L}")


def patterns(x, depths):
    cum = torch.zeros(x.shape[0], dtype=torch.int64, device=DEV)
    out = {}
    for i in range(L):
        h = x @ ws[i]
        cum = cum * 1000003 + ((h > 0).to(torch.int64) * PR[i]).sum(1)
        if (i + 1) in depths:
            out[i + 1] = cum.clone()
        x = torch.relu(h)
    return out


# ════════════ 2D: КУТИ ГРАНИЦЬ, ВИСОКА ТОЧНІСТЬ ════════════
M = 1 << 23
D2 = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]
gg = torch.Generator(device=DEV).manual_seed(4242)
B = torch.linalg.qr(torch.randn(n, 2, generator=gg, device=DEV))[0]
e1, e2 = B[:, 0].contiguous(), B[:, 1].contiguous()
acc = {d: torch.empty(M, dtype=torch.int64, device=DEV) for d in D2}
t0 = time.time()
for s0 in range(0, M, CH):
    s1 = min(s0 + CH, M)
    t = (torch.arange(s0, s1, device=DEV, dtype=torch.float64)
         * (2 * np.pi / M)).to(torch.float32)
    p = patterns(torch.outer(torch.cos(t), e1) + torch.outer(torch.sin(t), e2), D2)
    for d in D2:
        acc[d][s0:s1] = p[d]
print(f"  2D M={M} за {time.time()-t0:.0f} с")
edges, cnt2 = {}, {}
for d in D2:
    c = acc[d]
    idx = torch.nonzero(c != torch.roll(c, -1)).flatten()
    cnt2[d] = int(idx.numel())
    ang = ((idx.double() + 0.5) * (2 * np.pi / M)).cpu().numpy()
    edges[d] = [round(float(a), 7) for a in ang]
print(f"{'шар':>5s} {'секторів':>10s} {'×':>7s}")
prev = None
for d in D2:
    print(f"{d:>5d} {cnt2[d]:>10d} {(cnt2[d]/prev if prev else float('nan')):>7.3f}")
    prev = cnt2[d]

# ════════════ 3D: СФЕРА + ЗУМИ, ПАЛІТРОВИЙ PNG ════════════
PW, PH = 1024, 512
gg2 = torch.Generator(device=DEV).manual_seed(777)
B3 = torch.linalg.qr(torch.randn(n, 3, generator=gg2, device=DEV))[0]


def png8(path, idx, pal):
    h, w = idx.shape
    raw = b"".join(b"\x00" + idx[y].tobytes() for y in range(h))
    def ck(tag, data):
        c = tag + data
        return len(data).to_bytes(4, "big") + c + zlib.crc32(c).to_bytes(4, "big")
    hdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 3, 0, 0, 0])
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", hdr) + ck(b"PLTE", pal)
                + ck(b"IDAT", zlib.compress(raw, 9)) + ck(b"IEND", b""))
    return os.path.getsize(path)


rng = np.random.default_rng(5)
PAL = bytes(rng.integers(40, 250, 768, dtype=np.uint8))


def render(depth, span_deg, tag):
    """span_deg=180 → уся сфера (рівнокутна); інакше вікно навколо полюса."""
    if span_deg >= 180:
        lo = (torch.arange(PW, device=DEV, dtype=torch.float32) + .5) * (2 * np.pi / PW)
        la = (torch.arange(PH, device=DEV, dtype=torch.float32) + .5) * (np.pi / PH)
    else:
        s = np.deg2rad(span_deg)
        lo = np.pi + torch.linspace(-s, s, PW, device=DEV)
        la = np.pi / 2 + torch.linspace(-s / 2, s / 2, PH, device=DEV)
    LA, LO = torch.meshgrid(la, lo, indexing="ij")
    dirs = torch.stack([(torch.sin(LA) * torch.cos(LO)).reshape(-1),
                        (torch.sin(LA) * torch.sin(LO)).reshape(-1),
                        torch.cos(LA).reshape(-1)], 1)
    P = dirs.shape[0]
    out = torch.empty(P, dtype=torch.int64, device=DEV)
    for s0 in range(0, P, CH):
        s1 = min(s0 + CH, P)
        out[s0:s1] = patterns(dirs[s0:s1] @ B3.T, [depth])[depth]
    u = int(torch.unique(out).numel())
    v = (out ^ (out >> 17)) * 0x9E3779B97F4A7C15
    idx = (((v ^ (v >> 31)) & 0xFF).to(torch.uint8)
           .reshape(PH, PW).cpu().numpy())
    sz = png8(f"{OUT}/{tag}.png", idx, PAL)
    print(f"    {tag:>14s} шар {depth:>2d} span {span_deg:>6.2f}°  "
          f"конусів {u:>7d}  PNG {sz/1024:>6.0f} КБ")
    return sz, u


tot, uniq = 0, {}
JOBS = [(1, 180, "sph_L01"), (2, 180, "sph_L02"), (3, 180, "sph_L03"),
        (4, 180, "sph_L04"), (8, 180, "sph_L08"), (32, 180, "sph_L32"),
        (8, 12, "zoom_L08"), (16, 2.5, "zoom_L16"), (32, 0.5, "zoom_L32")]
for d, sp, tg in JOBS:
    s, u = render(d, sp, tg)
    tot += s
    uniq[tg] = u
print(f"  РАЗОМ PNG {tot/1024/1024:.2f} МБ")

json.dump({"edges": {str(d): edges[d] for d in D2},
           "counts": {str(d): cnt2[d] for d in D2},
           "uniq": uniq, "M": M, "width": n, "depth": L},
          open(f"{OUT}/cone2d.json", "w"))
print(f"  2D → {OUT}/cone2d.json "
      f"({os.path.getsize(f'{OUT}/cone2d.json')/1024:.0f} КБ)")
