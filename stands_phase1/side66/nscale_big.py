#!/usr/bin/env python3
"""Довге плече для показника `p`: n = 512, 1024 (асимптотичний режим).

Ті самі стенд і метрика, що в `nscale.py` (імпорт, не копія).
"""
import os
import math
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')

import numpy as np
import torch
from nscale import make_net, run_group, gmean, PROF_LAYERS, L_

def main():
    logp = '/mnt/win_d/llm/White-Box/arena62/logs/nscale_big.log'
    fh = open(logp, 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        fh.write(s + '\n')
        fh.flush()

    dev = torch.device(f'cuda:{int(os.environ.get("GPU", 2))}')
    npair = int(os.environ.get('NPAIR', 2_000_000))
    log(f'# довге плече, {dev}, {npair} пар/мережу')
    log('# 📣 ПЕРЕДБАЧЕНО при p=1 (якір 0.849% на 256): '
        '512→0.425% · 1024→0.212%\n')
    out = {}
    for n, k in ((512, 12), (1024, 12)):
        log(f'  n={n} ({k} мереж):')
        nets = [make_net(n, 700_000 + 131 * n + j) for j in range(k)]
        out[n] = run_group(f'n={n}', nets, dev, npair, PROF_LAYERS, log)
        e = out[n]['err']
        log(f'    → n={n}: сер={e.mean():.4f}%  геом={gmean(e):.4f}%  '
            f'SEM={e.std(ddof=1) / math.sqrt(len(e)):.4f}  '
            f'шум={out[n]["nz"].mean():.4f}%  прогноз p=1: '
            f'{0.849 * 256 / n:.4f}%  факт/прогноз={e.mean() / (0.849 * 256 / n):.3f}')
        for l in PROF_LAYERS:
            log(f'       шар {l:2d}: {out[n]["prof"][l].mean():.4f}% '
                f'(шум {out[n]["pnz"][l].mean():.4f}%)')
    np.savez('/mnt/win_d/llm/White-Box/arena62/logs/nscale_big.npz',
             **{f'e{n}': out[n]['err'] for n in out},
             **{f'z{n}': out[n]['nz'] for n in out})
    log('\n# локальний показник 512→1024: p = '
        f'{math.log(out[512]["err"].mean() / out[1024]["err"].mean()) / math.log(2):.4f}')
    fh.close()


if __name__ == '__main__':
    main()
