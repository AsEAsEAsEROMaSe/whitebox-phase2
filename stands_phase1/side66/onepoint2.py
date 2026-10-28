"""КОНТРОЛЬ ДО `onepoint.py`: з чим порівнювати похибку `z*`.

Без базового рівня числа «0.05 своя / 0.16 чужа / 0.41 чужа» не читаються.
Міряємо: (1) ВИПАДКОВА точка тієї самої норми; (2) випадкова точка норми √n;
(3) найкращий МАСШТАБ випадкової точки (позитивна однорідність: `h(cz)=c·h(z)`,
тож масштаб можна підігнати аналітично — і саме він може давати всю «якість»);
(4) профіль частки активних ПО ШАРАХ при `z*` проти типової точки — перевірка
гіпотези «`z*` це точка, де КОЖЕН шар поводиться середньо».
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget                              # noqa: E402
from tilt_wide import truth_last                                  # noqa: E402
from onepoint import solve, fwd                                   # noqa: E402

L_ = 32
DEV = 'cuda:3'


def act_profile(z, wc):
    x = z.unsqueeze(0)
    out = []
    for w in wc:
        p = x @ w
        out.append(float((p > 0).float().mean()))
        x = torch.relu(p)
    return out


def main(nets=3, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'{"мережа":>7s} {"z* (бекпроп)":>13s} {"ВИПАДК ‖z*‖":>12s} '
          f'{"ВИПАДК √n":>10s} {"ВИПАДК+масштаб":>15s} {"200 точок":>10s}')
    print('-' * 74)
    for n in range(nets):
        sd0 = int(np.load(fs[n])['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            T = truth_last(wc, dev, 510_000 + 7919 * n).float()
        n0 = wc[0].shape[0]
        z, rel = solve(wc, T, 1000, dev)
        with torch.no_grad():
            g = torch.Generator(device=dev); g.manual_seed(555 + n)
            r = torch.randn(1, n0, generator=g, device=dev)
            rn = r / r.norm() * z.norm()
            o1 = fwd(rn, wc)
            e1 = float((o1 - T).norm() / T.norm())
            r2 = r / r.norm() * np.sqrt(n0)
            o2 = fwd(r2, wc)
            e2 = float((o2 - T).norm() / T.norm())
            # 🔑 найкращий МАСШТАБ: `h(cz)=c·h(z)` ⟹ `c* = ⟨o,T⟩/⟨o,o⟩`
            c = float((o2 * T).sum() / (o2 * o2).sum())
            e3 = float((c * o2 - T).norm() / T.norm())
            rm = torch.randn(200, n0, generator=g, device=dev)
            om = fwd(rm, wc).mean(0, keepdim=True)
            e4 = float((om - T).norm() / T.norm())
        print(f'{n:7d} {rel:13.5f} {e1:12.5f} {e2:10.5f} {e3:15.5f} '
              f'{e4:10.5f}')
        if n == 0:
            with torch.no_grad():
                pz = act_profile(z, wc)
                pr = act_profile(r2[0], wc)
            print('\nчастка АКТИВНИХ по шарах (мережа 0):')
            print('  z*      :', ' '.join(f'{v:.2f}' for v in pz[:12]), '…')
            print('  випадк. :', ' '.join(f'{v:.2f}' for v in pr[:12]), '…')
            print(f'  середнє по z*: {np.mean(pz):.3f}, '
                  f'по випадковій: {np.mean(pr):.3f}\n')
        del W
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
