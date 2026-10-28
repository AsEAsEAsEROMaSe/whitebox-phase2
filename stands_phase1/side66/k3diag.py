"""ЧИ ВИСТАЧИТЬ ДІАГОНАЛЬНОГО НАБЛИЖЕННЯ ЗМІШАНИХ `κ₃`? (С74)

ЗВІДКИ. [[finding-edgeworth-correction-is-the-live-lever]] + M46 §D:
  * уся похибка замикання — негаусовість (при гаусовому вході формула точна);
  * вузьке місце НЕ в `m`, а в `C` (похибка 6–14% проти ~1%);
  * ✅ЗМІШАНІ `κ₃` — ПРАВИЛЬНА величина: пояснюють похибку `C` на **0.35–0.59**,
    тоді як сама `ρ` — на **0.00**;
  * 🛑АЛЕ дешеве `κ₃ ≈ cₐ·ρₐb` пояснює лише **0.03–0.10**.
Клас був закритий як `O(n⁴)`: точні `κ₃` — це тензор `n³` на шар.

🔑 СТРУКТУРНА ДУМКА, ЯКА ПОЯСНЮЄ ПРОВАЛ ДЕШЕВОГО ВАРІАНТА. Головний член
Прайса — `cp_ab·Φ(tₐ)Φ(t_b)`, тобто ∝ `ρab`. Наближення `cₐ·ρab` — ТЕЖ ∝ `ρab`
⟹ воно відрізняється від наявного члена лише МНОЖНИКОМ, а множник уже
підігнаний через `λ`. Лінійне по `ρ` ПОГЛИНАЄТЬСЯ калібруванням (узгоджено з
тим, що `ρ²`-член виявився зайвим). ⟹ **шукати треба те, що `ρ` НЕ пропорційне.**

🎯 КАНДИДАТ, ЯКОГО НЕ МІРЯЛИ. Якщо активації попереднього шару НАБЛИЖЕНО
незалежні, тензор `κ₃(xᵢ,xⱼ,x_k)` діагональний ⟹
    **`κ₃(pₐ,pₐ,p_b) = Σᵢ wᵢₐ² · wᵢb · κ₃(xᵢ)`**
тобто потрібен не тензор `n³`, а ВЕКТОР марджинальних кумулянтів, а згортка —
ОДИН матмул `n³`: `K = (W∘W)ᵀ @ (κ₃ ⊙ W)`. Це ТОЙ САМИЙ порядок, що вже
платиться в замиканні (`Wᵀ C W`) ⟹ ціна +1.1% флопів, не «пів бюджету».

📐 ГЕЙТ (числа ДО прогону): R²(діагональне проти точного `κ₃`) 0.5–0.8;
частка похибки `C` 0.15–0.35 при оракульних 0.35–0.59.
    >0.30 ⟹ КАНАЛ ЖИВИЙ (поріг, за яким світить 2×);
    <0.15 ⟹ незалежність активацій НЕ виконується, клас закрито остаточно.

ПЛОМБИ
  ✅ ОРАКУЛ мусить відтворити 0.35–0.59 — інакше стенд не той, що в С73;
  ✅ сама `ρ` мусить дати ≈0.00 — це вже зміряно і має повторитись;
  ⛔ ВИПАДКОВІ ознаки тієї ж форми ⟹ R²≈0;
  🔑 фіт і вимір — на РІЗНИХ парах (розділені за індексом).
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import cut_and_budget, W_, _phi, _Phi                # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NP_, CH = 262_144, 16384
LAYERS = (2, 4, 8, 16, 24, 29)
KGRID = (4, 8, 16)           # ранг підпростору, де параметризуємо тензор κ₃


def sample_stats(wc, dev, seed, upto):
    """Вибіркові моменти шару `upto`: `p`-моменти, `C(relu)`, `κ₃` активацій
    ПОПЕРЕДНЬОГО шару і ТОЧНІ змішані `κ₃(pₐ,pₐ,p_b)`."""
    rng = np.random.default_rng(seed)
    n = wc[upto].shape[1]
    npv = wc[upto].shape[0]
    z1 = torch.zeros(n, dtype=torch.float64, device=dev)
    sp, sa = z1.clone(), z1.clone()
    gp = torch.zeros(n, n, dtype=torch.float64, device=dev)
    ga = torch.zeros(n, n, dtype=torch.float64, device=dev)
    sx = torch.zeros(npv, dtype=torch.float64, device=dev)
    qx = torch.zeros(npv, dtype=torch.float64, device=dev)
    cx = torch.zeros(npv, dtype=torch.float64, device=dev)
    done = 0
    chunks = []
    while done < NP_:
        k = min(CH, NP_ - done)
        zz = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                          device=dev)
        a = torch.cat([zz, -zz], 0)
        for i in range(upto):
            a = torch.relu(a @ wc[i])
        p = a @ wc[upto]
        pd, ad, xd = p.double(), torch.relu(p).double(), a.double()
        sp += pd.sum(0)
        gp += pd.T @ pd
        sa += ad.sum(0)
        ga += ad.T @ ad
        sx += xd.sum(0)
        qx += (xd * xd).sum(0)
        cx += (xd ** 3).sum(0)
        chunks.append((seed, done, k))
        done += k
    N = float(2 * NP_)
    mp, ma = sp / N, sa / N
    cp = gp / N - torch.outer(mp, mp)
    ca = ga / N - torch.outer(ma, ma)
    mx = sx / N
    k3x = cx / N - 3.0 * mx * (qx / N) + 2.0 * mx ** 3     # κ₃ активацій
    # ── ТОЧНІ змішані κ₃ + НИЗЬКОРАНГОВИЙ тензор у підпросторі активацій ──
    # 🔑 базис `U` — головні напрямки АКТИВАЦІЙ попереднього шару (там, де вони
    #    живуть). У ньому κ₃ — це `k³` чисел замість `n³`.
    cxx = (qx * 0).clone()      # заглушка, реальна коваріація нижче
    mx = sx / N
    Cx = torch.zeros(npv, npv, dtype=torch.float64, device=dev)
    rng2 = np.random.default_rng(seed)
    T = torch.zeros(n, n, dtype=torch.float64, device=dev)
    done = 0
    while done < NP_:
        k = min(CH, NP_ - done)
        zz = torch.tensor(rng2.standard_normal((k, W_)).astype(np.float32),
                          device=dev)
        a = torch.cat([zz, -zz], 0)
        for i in range(upto):
            a = torch.relu(a @ wc[i])
        xd = a.double() - mx
        Cx += xd.T @ xd
        pc = (a @ wc[upto]).double() - mp
        T += (pc * pc).T @ pc
        done += k
    Cx /= N
    ev, V = torch.linalg.eigh(Cx)
    U = {kk: V[:, -kk:].contiguous() for kk in KGRID if kk < npv}
    G = {kk: torch.zeros(kk, kk, kk, dtype=torch.float64, device=dev)
         for kk in U}
    rng3 = np.random.default_rng(seed)
    done = 0
    while done < NP_:
        k = min(CH, NP_ - done)
        zz = torch.tensor(rng3.standard_normal((k, W_)).astype(np.float32),
                          device=dev)
        a = torch.cat([zz, -zz], 0)
        for i in range(upto):
            a = torch.relu(a @ wc[i])
        xd = a.double() - mx
        for kk, Uk in U.items():
            u = xd @ Uk
            G[kk] += torch.einsum('ma,mb,mc->abc', u, u, u)
        done += k
    for kk in G:
        G[kk] /= N
    del cxx, Cx
    return mp, cp, ca, T / N, k3x, U, G


def gauss_cov(mp, cp, s):
    t = mp / s
    ph, Ph = _phi(t), _Phi(t)
    m = mp * Ph + s * ph
    C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
    dg = (mp * mp + s * s) * Ph + mp * s * ph - m * m
    return C + torch.diag(dg - C.diagonal()), t, ph, Ph


def r2_of(X, y):
    """R² з фітом на першій половині пар і виміром на другій."""
    h = X.shape[0] // 2
    co = torch.linalg.lstsq(X[:h], y[:h].unsqueeze(1)).solution
    pr = (X[h:] @ co).squeeze(1)
    yb = y[h:]
    return 1.0 - float(((yb - pr) ** 2).sum() / ((yb - yb.mean()) ** 2).sum())


def main(nets=3, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    ROW = {l: {k: [] for k in ('err', 'k3r2', 'rho', 'ora', 'dia', 'rnd')}
           for l in LAYERS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            for l in LAYERS:
                if l >= len(wc) - 1:
                    continue
                mp, cp, ca, T, k3x, U, G = sample_stats(
                    wc, dev, 880_000 + 7919 * n + l, l)
                s = cp.diagonal().clamp_min(1e-30).sqrt()
                cg, t, ph, Ph = gauss_cov(mp, cp, s)
                w = wc[l].double()
                # 🔑 ДІАГОНАЛЬНЕ НАБЛИЖЕННЯ: K_ab = Σᵢ wᵢₐ²·wᵢb·κ₃(xᵢ)
                K = (w * w).T @ (k3x.unsqueeze(1) * w)
                off = ~torch.eye(len(s), dtype=torch.bool, device=dev)
                sc = torch.outer(s, s)
                y = ((ca - cg) / sc)[off]
                ROW[l]['err'].append(float(((ca - cg) / sc)[off].abs().mean()
                                           / (ca / sc)[off].abs().mean()) * 100)
                # ✅ЯКІСТЬ САМОГО НАБЛИЖЕННЯ κ₃ (нормуємо на s²ₐ·s_b)
                nz = (s * s).unsqueeze(1) * s.unsqueeze(0)
                ROW[l]['k3r2'].append(r2_of(
                    torch.stack([(K / nz)[off], torch.ones_like(y)], 1),
                    (T / nz)[off]))

                r = (cp / sc)[off]
                Ti, Tj = (torch.outer(t, torch.ones_like(t))[off],
                          torch.outer(torch.ones_like(t), t)[off])
                Pa, Pb = (torch.outer(ph, torch.ones_like(ph))[off],
                          torch.outer(torch.ones_like(ph), ph)[off])
                one = torch.ones_like(y)

                def block(Q):
                    """Ознаки з джерела `κ₃`: сам член і його `t`-модуляції."""
                    Qs = Q[off]
                    Qt = Q.T[off]                      # κ₃(pₐ,p_b,p_b)
                    return [Qs, Qt, Qs * Ti, Qt * Tj, Qs * Pa * Pb,
                            (Qs + Qt) * Ti * Tj]

                base = [r, r * r, one]
                Xr = torch.stack(base, 1)
                Xo = torch.stack(base + block(T / nz), 1)
                Xd = torch.stack(base + block(K / nz), 1)
                gg = torch.Generator(device=dev).manual_seed(17 + l)
                Xn = torch.stack(base + [torch.randn(y.shape, dtype=y.dtype,
                                                     device=dev, generator=gg)
                                         for _ in range(6)], 1)
                ROW[l]['rho'].append(r2_of(Xr, y))
                ROW[l]['ora'].append(r2_of(Xo, y))
                ROW[l]['dia'].append(r2_of(Xd, y))
                ROW[l]['rnd'].append(r2_of(Xn, y))
                # ── НИЗЬКОРАНГОВИЙ тензор: κ₃ у підпросторі активацій ──
                # `κ₃(pₐ,pₐ,p_b) = Σ_αβγ G_αβγ V_αa V_βa V_γb`, `V = Uᵀ W`
                for kk, Uk in U.items():
                    Vk = Uk.T @ w                     # (k × n)
                    H = torch.einsum('abg,am,bm->gm', G[kk], Vk, Vk)
                    Klr = H.T @ Vk                    # (n × n)
                    ROW[l].setdefault(f'k3r2_{kk}', []).append(r2_of(
                        torch.stack([(Klr / nz)[off], one], 1), (T / nz)[off]))
                    ROW[l].setdefault(f'lr{kk}', []).append(
                        r2_of(torch.stack(base + block(Klr / nz), 1), y))
                    del Vk, H, Klr
                del T, K, ca, cg, cp, U, G
                torch.cuda.empty_cache()
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    m = np.median
    print(f'\n{nets} мереж, істина {2*NP_} семплів\n')
    print('А. ЯКІСТЬ САМОГО НАБЛИЖЕННЯ `κ₃` (R² проти ТОЧНОГО змішаного)\n')
    print(f'{"шар":>4s} {"діагональне":>12s} '
          + ' '.join(f'{"ранг " + str(k):>10s}' for k in KGRID))
    print('-' * 50)
    for l in LAYERS:
        if not ROW[l]['ora']:
            continue
        r = ROW[l]
        print(f'{l:4d} {m(r["k3r2"]):12.4f} '
              + ' '.join(f'{m(r[f"k3r2_{k}"]):10.4f}'
                         if f'k3r2_{k}' in r else f'{"—":>10s}' for k in KGRID))
    print('-' * 50)
    print('\nБ. СКІЛЬКИ ПОХИБКИ `C` ПОЯСНЮЄ (те, що вирішує)\n')
    print(f'{"шар":>4s} {"похибка C":>10s} │ {"⛔ρ,ρ²":>8s} {"⚠️ОРАКУЛ":>9s} '
          f'{"✅ДІАГ":>8s} ' + ' '.join(f'{"ранг " + str(k):>9s}' for k in KGRID)
          + f' {"⛔ШУМ":>8s}')
    print('-' * 92)
    for l in LAYERS:
        if not ROW[l]['ora']:
            continue
        r = ROW[l]
        print(f'{l:4d} {m(r["err"]):9.2f}% │ {m(r["rho"]):8.4f} '
              f'{m(r["ora"]):9.4f} {m(r["dia"]):8.4f} '
              + ' '.join(f'{m(r[f"lr{k}"]):9.4f}' if f'lr{k}' in r
                         else f'{"—":>9s}' for k in KGRID)
              + f' {m(r["rnd"]):8.4f}')
    print('-' * 92)
    print('🎯 ГЕЙТ: будь-яка дешева колонка >0.30 ⟹ КАНАЛ ЖИВИЙ.\n'
          '   Усі <0.15 ⟹ клас `κ₃` закрито остаточно.\n'
          '✅ПЛОМБИ: ⚠️ОРАКУЛ мусить лягти в 0.35–0.59 (С73), ⛔ρ ≈0, ⛔ШУМ ≈0.\n'
          '🚨 ранг `k` тут ОРАКУЛЬНИЙ (базис і тензор із вибірки) ⟹ це СТЕЛЯ '
          'схеми,\n   а не її ціна: реальне джерело `G` мало б іти із '
          'ЗАМИКАННЯ (інакше антагонізм M46 §F).')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
