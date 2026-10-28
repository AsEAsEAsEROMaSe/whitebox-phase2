"""saddlechk96 — ЧИ ВЗАГАЛІ ІСНУЄ СЕДЛОВА ТОЧКА НА 4 КУМУЛЯНТАХ (порада АРХІВУ).

`[ЦИТАТА АРХІВ→ВИМІР, A3]` обрізаний 4-кумулянтний CGF не є валідним CGF:
`K(θ) = mθ + σ²θ²/2 + κ₃θ³/6 + κ₄θ⁴/24` ⟹ `K''(θ) = σ² + κ₃θ + κ₄θ²/2`.
Це парабола за θ; вона додатна СКРІЗЬ ⟺ `κ₄ > 0` І дискримінант < 0:
```
   κ₃² < 2σ²κ₄        у стандартизованих:   γ₃² < 2γ₄
```
Середні значення (С95/С96): γ₃≈0.42, γ₄≈0.23 ⟹ 0.176 < 0.46 ✅ проходить.
🚨 АЛЕ це СЕРЕДНЄ. Досить ОДНОГО нейрона з `γ₄≤0` або `γ₃²≥2γ₄`, щоб седлова
стала ill-posed саме там. Рахуємо ЧАСТКУ таких нейронів — це вирок гілці ДО
будь-якого стенду (протокол п.3: пломба мусить жити в режимі об'єкта).
"""
import glob
import os
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N = 1 << (int(sys.argv[3]) if len(sys.argv) > 3 else 17)
CH = 1 << 14
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")

G3A, G4A = [], []
with torch.no_grad():
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        for i in range(t.num_rows):
            if got >= NNET:
                break
            W = [torch.as_tensor(np.ascontiguousarray(w[i][L]), device=DEV)
                 for L in range(32)]
            g = torch.Generator(device=DEV).manual_seed(4200 + got)
            X = torch.randn(N, 256, generator=g, device=DEV)
            for L in range(32):
                Wd = W[L]
                s1 = torch.zeros(256, dtype=torch.float64, device=DEV)
                s2 = torch.zeros(256, dtype=torch.float64, device=DEV)
                for o in range(0, N, CH):
                    pb = (X[o:o + CH] @ Wd).double()
                    s1 += pb.sum(0)
                    s2 += (pb * pb).sum(0)
                mu = s1 / N
                sd = (s2 / N - mu * mu).clamp_min(1e-300).sqrt()
                s3 = torch.zeros(256, dtype=torch.float64, device=DEV)
                s4 = torch.zeros(256, dtype=torch.float64, device=DEV)
                for o in range(0, N, CH):
                    c = ((X[o:o + CH] @ Wd).double() - mu) / sd
                    s3 += (c ** 3).sum(0)
                    s4 += (c ** 4).sum(0)
                if L >= 8:
                    G3A.append((s3 / N).cpu())
                    G4A.append((s4 / N - 3.0).cpu())
                X = torch.relu(X @ Wd)
            got += 1
        if got >= NNET:
            break

g3 = torch.cat(G3A)
g4 = torch.cat(G4A)
n = g3.numel()
bad4 = float((g4 <= 0).double().mean())
disc = g3 * g3 - 2.0 * g4
badd = float((disc >= 0).double().mean())
bad = float(((g4 <= 0) | (disc >= 0)).double().mean())
print(f"нейронів (шари 8..31, {NNET} мереж, N=2^{int(np.log2(N))}): {n}")
print(f"γ₃: медіана {float(g3.median()):+.4f}  5%..95% "
      f"[{float(g3.quantile(0.05)):+.4f}, {float(g3.quantile(0.95)):+.4f}]")
print(f"γ₄: медіана {float(g4.median()):+.4f}  5%..95% "
      f"[{float(g4.quantile(0.05)):+.4f}, {float(g4.quantile(0.95)):+.4f}]")
print(f"  мін γ₄ = {float(g4.min()):+.4f}")
print()
print(f"🚨 частка нейронів із γ₄ ≤ 0 (K'' парабола ВНИЗ): {bad4*100:.3f}%")
print(f"🚨 частка з γ₃² ≥ 2γ₄ (дискримінант ≥ 0):        {badd*100:.3f}%")
print(f"🚨 РАЗОМ ill-posed для седлової на 4 кумулянтах:  {bad*100:.3f}%")
print()
if bad < 0.001:
    print("✅ ВЕРДИКТ: седлова на 4 кумулянтах КОРЕКТНА майже скрізь ⟹ гілку")
    print("   МОЖНА будувати; блокер r444 у нашому режимі не спрацьовує.")
else:
    print("⛔ ВЕРДИКТ: седлова ill-posed на помітній частці нейронів ⟹ стенд")
    print("   упаде з причини, що до справи не стосується. Потрібен CGF, що")
    print("   лишається опуклим за побудовою (напр. експоненційний нахил).")
