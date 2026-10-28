"""21.10.2026 white-box-f9: МК-істина c3/c4 передактивацій (2^24, сід 22000+k як chpre2/chprune) для навчання коректора D3/g4row.
Пише /home/trim/wblogs/c34/c34_n<k>.npz (c3, c4, var, mean (16,1024) float32). usage: mc_c34.py <a-b> [--step s --off o]"""
import os, sys, math, time
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
lo, hi = (int(x) for x in sys.argv[1].split("-")); step = int(sys.argv[2]) if len(sys.argv) > 2 else 1; off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
D = "/home/trim/wblogs/c34"; os.makedirs(D, exist_ok=True); N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192; dev = "cuda"
for k in range(lo, hi + 1):
    if (k - lo) % step != off: continue
    out = "%s/c34_n%d.npz" % (D, k)
    if os.path.exists(out): continue
    t0 = time.time()
    g = torch.Generator(device="cpu").manual_seed(770000 + k)
    W = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)).to(dev)
    gg = torch.Generator(device=dev).manual_seed(22000 + k)
    S = torch.zeros(16, 4, 1024, dtype=torch.float64, device=dev); sh = {}; n = 0
    while n < N:
        h = torch.randn(B, 1024, generator=gg, device=dev)
        for l in range(16):
            p = h @ W[l]
            if l not in sh: sh[l] = p.mean(0)
            x = (p - sh[l]).double()
            S[l, 0] += x.sum(0); S[l, 1] += (x * x).sum(0); S[l, 2] += (x ** 3).sum(0); S[l, 3] += (x ** 4).sum(0)
            h = torch.relu(p)
        n += B
    a = (S / n).cpu().numpy(); m1, m2, m3, m4 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    var = m2 - m1 ** 2; c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2
    mean = m1 + torch.stack([sh[l] for l in range(16)]).double().cpu().numpy()
    np.savez(out + ".tmp.npz", c3=c3.astype(np.float32), c4=c4.astype(np.float32), var=var.astype(np.float32), mean=mean.astype(np.float32))
    os.replace(out + ".tmp.npz", out)
    print("net %d %.0fs" % (k, time.time() - t0), flush=True)
