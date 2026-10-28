"""КРОК 1: ваги+істина для ПРОМІЖНИХ ширин навколо обриву (640, 768, 896)."""
import sys
import numpy as np
import torch
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
GRID = [(640, 32), (768, 32), (896, 32)]
NPAIR, CH = 1 << 19, 8192
out = {}
for width, depth in GRID:
    for n in range(NETS):
        rng = np.random.default_rng(90_000 + 977 * n + 13 * width + depth)
        sc = np.sqrt(2.0 / width)
        W = [(rng.standard_normal((width, width)) * sc).astype(np.float32)
             for _ in range(depth)]
        Wt = [torch.tensor(w, device=DEV) for w in W]
        acc = torch.zeros(width, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV).manual_seed(555 + n)
        done = 0
        with torch.no_grad():
            while done < NPAIR:
                k = min(CH, NPAIR - done)
                z = torch.randn(k, width, device=DEV, generator=g)
                x = torch.cat([z, -z], 0)
                for w in Wt:
                    x = torch.relu(x @ w)
                acc += x.double().sum(0)
                done += k
        key = f'{width}x{depth}x{n}'
        out[key + '_T'] = (acc / (2.0 * done)).cpu().numpy()
        for i, w in enumerate(W):
            out[f'{key}_w{i}'] = w
        del Wt
        torch.cuda.empty_cache()
    print(f'  … {width}×{depth} готово', flush=True)
np.savez('/mnt/win_d/llm/White-Box/side66/logs/dims2.npz', **out)
print('збережено side66/logs/dims2.npz')
