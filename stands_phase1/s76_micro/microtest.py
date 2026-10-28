"""№5: МІКРОМОДЕЛЬ z → x₂₈ + СПРАВЖНІЙ ХВІСТ 29..32. (С76, ідея юзера)

════════ СХЕМА ════════
Вчимо малу модель g: вхід z (256) → активації шару 28 (d₂₈). Тоді
«дешевий семпл» = t(g(z)) — прогін лише хвоста 29..32 крізь СПРАВЖНІ ваги.
Ціна: c ≈ 0.1 повного проходу ⟹ якщо corr(t(g), f) висока, двохрівнева
схема (мала пачка виправляє зміщення сурогата, велика дає дешеву масу)
дає аж до 1/c = 10×. Все вирішує ρ².

Міряємо (тренування БЕЗКОШТОВНЕ — свідомо на користь ідеї):
  1) R²(x₂₈) — як добре g вгадує шар 28 поелементно (центровано);
  2) R²cv = 1 − Var[f − t(g)]/Var[f] на СВІЖІЙ пачці — сила контрольної
     змінної (β=1), вона ж ρ²;
  3) BIAS сирої версії юзера: mean[t(g)] на великій пачці проти істини —
     у одиницях MSE бази;
  4) implied двохрівневий виграш при вільному тренуванні й оптимальному
     розподілі бюджету: gain = V₀ / (√(V₁(1+c)) + √(V₂c))².

Моделі: LIN (гребенева, замкнена форма) і MLP 256→512→d₂₈ (Adam).
📣 ПЕРЕДБАЧЕННЯ: могили кажуть R²cv ≤ 0.3 ⟹ виграш ≤ 1.1× (і то до
   сплітів бюджету), bias сирої версії >> MSE бази. Якщо ж R²cv > 0.7 —
   ідея ЖИВА і це головна новина ночі.
🚨 ПЛОМБИ: ✅оцінка на свіжій пачці (не тренувальній) · ✅крос-ціль для
   bias · ✅LIN поруч із MLP (задача карає складність — перевіримо й тут).
"""
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import cut_and_budget, W_                                    # noqa: E402
from biassplit import truth_full                                         # noqa: E402

L_ = 32
SPLIT = 28                    # head = шари 0..27, tail = 28..31
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:1'
KSIG = 8


def head_fwd(wc, x):
    for i in range(SPLIT):
        x = torch.relu(x @ wc[i])
    return x


def tail_fwd(wc, x):
    for i in range(SPLIT, len(wc)):
        x = torch.relu(x @ wc[i])
    return x


def draw(npair, seed, dev):
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    return torch.cat([z, -z], 0)


def fit_lin(Z, Y, lam=1e-2):
    X = torch.cat([Z, torch.ones(Z.shape[0], 1, device=Z.device)], 1)
    G = X.T @ X + lam * X.shape[0] * torch.eye(X.shape[1], device=Z.device)
    th = torch.linalg.solve(G, X.T @ Y)
    return lambda q: torch.cat(
        [q, torch.ones(q.shape[0], 1, device=q.device)], 1) @ th


def fit_mlp(Z, Y, dev, epochs=400, hid=512):
    d = Y.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(W_, hid), torch.nn.ReLU(),
        torch.nn.Linear(hid, d)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for e in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(Z), Y)
        loss.backward()
        opt.step()
    net.eval()
    return net


def main(nets=8, dev=DEV):
    t0 = time.time()
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'№5 МІКРОМОДЕЛЬ 0..27 + СПРАВЖНІЙ ХВІСТ 28..31 · {nets} мереж, {dev}')
    print('📣 ПЕРЕДБАЧЕНО: R²cv ≤ 0.3 ⟹ ≤1.1×; якщо >0.7 — ЖИВА\n', flush=True)

    hdr = (f'{"мережа":>6} {"R²x28":>7} {"R²cv":>7} {"bias/MSE":>9} '
           f'{"gain2lvl":>9}   модель')
    print(hdr)
    R = {'LIN': [], 'MLP': []}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            T1 = truth_full(W, dev, 810_000 + 7919 * n).float()
            T2 = truth_full(W, dev, 910_000 + 7919 * n).float()
            # вартість хвоста відносно повного проходу (флопи матмулів)
            c_full = sum(w.shape[0] * w.shape[1] for w in wc)
            c_tail = (sum(w.shape[0] * w.shape[1] for w in wc[SPLIT:])
                      + W_ * 512 + 512 * wc[SPLIT].shape[0])
            c = c_tail / c_full
            # тренувальна пачка A і свіжа пачка B (та сама вартість, що в бою)
            xa = draw(npair, 500_000 + n, dev)
            xb = draw(npair, 600_000 + n, dev)
            ya = head_fwd(wc, xa)
            yb = head_fwd(wc, xb)
            fb = tail_fwd(wc, yb)                    # справжні виходи пачки B
            x_big = draw(4 * npair, 700_000 + n, dev)
        for name in ('LIN', 'MLP'):
            if name == 'LIN':
                g = fit_lin(xa, ya)
            else:
                g = fit_mlp(xa, ya, dev)
            with torch.no_grad():
                yh = g(xb)
                tb = tail_fwd(wc, torch.relu(yh))     # t(g(z)) на свіжій B
                # 1) якість x28 (центровано)
                r2x = 1.0 - float(((yb - yh) ** 2).sum()
                                  / ((yb - yb.mean(0)) ** 2).sum())
                # 2) сила КЗ на виході
                vf = float(((fb - fb.mean(0)) ** 2).sum())
                vd = float((((fb - tb) - (fb - tb).mean(0)) ** 2).sum())
                vt = float(((tb - tb.mean(0)) ** 2).sum())
                r2cv = 1.0 - vd / vf
                # 3) сире зміщення юзерської версії (велика пачка, крос-ціль)
                tbig = tail_fwd(wc, torch.relu(g(x_big)))
                mb = tbig.double().mean(0).float()
                bias2 = float(torch.dot(mb - T1, mb - T2))
                xbar = fb.double().mean(0).float()
                mse0 = float(torch.dot(xbar - T1, xbar - T2))
                # 4) двохрівнева стеля (вільне тренування, оптимальний розподіл)
                num = float(np.sqrt(max(vd, 0) * (1 + c))
                            + np.sqrt(max(vt, 0) * c))
                gain = vf / max(num * num / 1.0, 1e-30)
                R[name].append((r2x, r2cv, bias2 / max(mse0, 1e-30), gain))
                print(f'{n+1:>6} {r2x:>7.3f} {r2cv:>7.3f} '
                      f'{bias2/max(mse0,1e-30):>9.1f} {gain:>9.3f}   {name}',
                      flush=True)
        del W, wc, xa, xb, ya, yb, x_big
        torch.cuda.empty_cache()

    print(f'\n(вартість дешевого семпла c ≈ {c:.3f} повного)')
    for name in ('LIN', 'MLP'):
        a = np.array(R[name])
        print(f'{name}: R²x28 {np.mean(a[:,0]):.3f} · R²cv '
              f'{np.mean(a[:,1]):.3f} · bias/MSE {np.mean(a[:,2]):.1f}× · '
              f'двохрівневий виграш **{np.mean(a[:,3]):.3f}×**')
    print(f'   час {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
