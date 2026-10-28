"""ЧИ ПРАЦЮЄ ЯКІР, ЯКЩО ЗАМИКАННЯ ЖИВИТЬСЯ ВИБІРКОЮ? (С73)

ЗВІДКИ. Похибка `C` = 6–14%, і дешево виправити її аналітично НЕ вийшло
(марджинальні `γ` пояснюють 7–18%). Але вибіркові моменти шару `k` беруться
ОДНИМ матмулом і вони НАБАГАТО точніші за замикальні (шум вибірки 0.0024%
проти похибки замикання 0.861% на шарі 16 — у 359× краще).

🎯 ІДЕЯ: ПЕРЕЗАВАНТАЖИТИ рекурсію вибірковими `(m,C)` шару `k`, далі йти
звичайною формулою. Тоді `m_cl₃₀` стає точнішим.
🚨 АЛЕ: `m_cl₃₀` тепер КОРЕЛЮЄ з тією самою вибіркою, з якої взято `x̄₃₀` ⟹
усадка `pred = (1−λ)·x̄ + λ·m_cl` може виродитись у тавтологію. ЩО ПЕРЕВАЖИТЬ —
ВИРІШУЄ ВИМІР, а не міркування.

ВАРІАНТИ: `k = −1` (база, замикання без вибірки) · `k = 0,1,2,4,8`.
📊 МІРЯЄМО ДВА ЧИСЛА ОДРАЗУ:
   * точність `m_cl₃₀` (чи справді краще);
   * ВИГРАШ усадки проти чистого `x̄₃₀` (чи це можна ВЗЯТИ).
🔑 обидва — проти НЕЗАЛЕЖНОЇ істини, `λ` підбирається на ІНШИХ жеребах.

🚨 ПЛОМБИ
  ✅ `k=−1` мусить відтворити відомий виграш глибокого якоря ≈1.04×;
  ⛔ ТАВТОЛОГІЯ `k=29`: замикання перезавантажене вибірковим шаром 30 —
     `m_cl₃₀ ≈ x̄₃₀` ⟹ виграш мусить впасти до ≈1.00× (виродження).
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_, _phi, _Phi          # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NPAIR = 3100
NDRAW = 80
KS = (-1, 0, 1, 2, 4, 8, 29)
TARGET = 29          # 0-based шар 30


def closure_from(wc, start, m0, C0):
    """Рекурсія; `start=-1` — штатний старт, інакше з `(m0,C0)` ПІСЛЯ шару `start`."""
    dev = wc[0].device
    if start < 0:
        m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
        C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
        i0 = 0
    else:
        m, C, i0 = m0, C0, start + 1
    for i in range(i0, TARGET + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        dg = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(dg - C.diagonal())
    return m


def draw(wc, dev, seed):
    """Жереб: активації всіх шарів (для вибіркових моментів) + `x̄` шару 30."""
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(NPAIR, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    a = x
    acts = []
    for i in range(TARGET + 1):
        a = torch.relu(a @ wc[i])
        acts.append(a)
    return acts


def truth30(wc, dev, seed, npair=131_072, chunk=16384):
    acc = torch.zeros(wc[TARGET].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(seed)
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(TARGET + 1):
            a = torch.relu(a @ wc[i])
        acc += a.double().sum(0)
        done += k
    return acc / float(2 * npair)


def main(nets=4, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    acc = {k: {'err': [], 'gain': [], 'lam': []} for k in KS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            T = truth30(wc, dev, 660_000 + 7919 * n)
            XB, MC = {k: [] for k in KS}, {k: [] for k in KS}
            for d in range(NDRAW):
                acts = draw(wc, dev, 220_000 + 977 * n + d)
                xb = acts[TARGET].double().mean(0)
                for k in KS:
                    XB[k].append(xb)
                    if k < 0:
                        MC[k].append(closure_from(wc, -1, None, None))
                    else:
                        A = acts[k].double()
                        mk = A.mean(0)
                        Ck = (A.T @ A) / A.shape[0] - torch.outer(mk, mk)
                        MC[k].append(closure_from(wc, k, mk, Ck))
                del acts
            for k in KS:
                X = torch.stack(XB[k])
                M = torch.stack(MC[k])
                acc[k]['err'].append(100.0 * float(
                    ((M.mean(0) - T) ** 2).mean().sqrt() / (T ** 2).mean().sqrt()))
                h = NDRAW // 2
                # λ з першої половини, вимір — на другій
                d1, e1 = (X[:h] - M[:h]), (X[:h] - T)
                lam = float((d1 * e1).sum() / (d1 * d1).sum().clamp_min(1e-30))
                base = float(((X[h:] - T) ** 2).mean())
                got = float(((X[h:] - lam * (X[h:] - M[h:]) - T) ** 2).mean())
                acc[k]['gain'].append(base / max(got, 1e-30))
                acc[k]['lam'].append(lam)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж × {NDRAW} жеребів по {2*NPAIR} семплів\n')
    print(f'{"перезавантаження":>22s} {"похибка m₃₀":>13s} {"λ":>8s} '
          f'{"ВИГРАШ якоря":>14s}')
    print('-' * 62)
    for k in KS:
        nm = ('база (без вибірки)' if k < 0
              else ('⛔ТАВТОЛОГІЯ шар 30' if k == 29 else f'вибіркові моменти шару {k+1}'))
        print(f'{nm:>22s} {np.median(acc[k]["err"]):12.4f}% '
              f'{np.median(acc[k]["lam"]):8.4f} {np.median(acc[k]["gain"]):13.4f}×')
    print('-' * 62)
    print('🎯 ГЕЙТ: якщо якийсь рядок дає ВИГРАШ помітно >1.04× (база) — '
          'гібрид ЖИВИЙ.\n⛔ рядок «ТАВТОЛОГІЯ» мусить впасти до ≈1.00×, '
          'інакше стенд не розсуджує.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
