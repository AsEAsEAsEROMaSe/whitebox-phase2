"""ПАРНІ СТАТИСТИКИ Й ПОЕЛЕМЕНТНЕ `r` — виправлення двох моїх вад одразу.

ВАДА 1 (юзер): я зміряв частку активних АГРЕГОВАНО по 256 нейронах, отримав нуль
і оголосив «закон» про антикореляцію знаності та корисності. Усереднення по 256
нейронах — властивість МОГО СПОСОБУ брати статистику, не задачі. Тут `r` міряється
**ПОЕЛЕМЕНТНО**, на всіх пробних шарах.

ВАДА 2 (юзер): усі мої КЗ — функції ОДНОГО нейрона, а ~70% похибки живе в
КОРЕЛЯЦІЯХ. Парна статистика `a_i·a_j` має ОБЧИСЛЮВАНЕ гаусове середнє (формула
Прайса = ядро арккосинуса; при ненульових середніх беремо двовимірну квадратуру
Гаусса-Ерміта — це так само без семплів), і несе рівно ту інформацію, якої в
покомпонентних немає ЗА ПОБУДОВОЮ.

🔑 ГОЛОВНЕ ЧИСЛО — розподіл `r = |замикання − істина| / std(статистики)` по
СІМ'ЯХ. Якщо для пар `r` ближче до `E[a²]` (0.5–1.1), ніж до `E[a]` (7–9), гілка
жива і адресує задокументовані 70%. Якщо такий самий поганий — тоді в нас буде
СПРАВЖНІЙ закон замість проголошеного.

🔑 І УСАДКА БЕЗ ФІТУ: `β_{ℓ,i} = 1/(1+r²_{ℓ,i})` виводиться поелементно
([[finding-s2-beta-weights-derived-not-fitted]]), тобто нульовий ризик
оракульності — на відміну від померéжевого `β` у `patcv.py`.

КОНТРОЛІ
  * ✅ПЛОМБА КВАДРАТУРИ: на діагоналі `i=j` двовимірна квадратура мусить дати
    рівно `E[a²] = (m²+s²)Φ(t)+m·s·φ(t)` — замкнуту форму, яку ми вже маємо;
  * ✅ПЛОМБА ІСТИНИ: `E[a_i·a_i]` з великої вибірки проти `E[a_i²]` з тієї самої;
  * ⛔НЕГ: пари, взяті ВИПАДКОВО, проти пар, обраних за вагою наступного шару —
    якщо різниці нема, вибір пар нічого не дає.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, run_side1, LAM_S, W_, _phi, _Phi  # noqa: E402
from tilt_wide import truth_last                                       # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 96
K31 = 31
PROBE = [1, 4, 8, 16, 24, 30]
NPAIR_SEL = 256
NGH = 24
LAM31 = tuple(LAM_S) + (0.0,) * (K31 - len(LAM_S))


def closure_full(wc, layers, dev):
    """Рекурсія замикання з ПОВНОЮ коваріацією; повертає на пробних шарах
    (m_доакт, s_доакт, C_доакт) та покомпонентні замикальні середні."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    out = {}
    for i in range(max(layers) + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(d - C.diagonal())
        if i in layers:
            out[i] = {'mp': mp, 's': s, 'cp': cp, 'a': m.clone(),
                      'ind': Ph.clone(), 'a2': d + m * m}
    return out


def price_pairs(mp, s, cp, ii, jj, ngh=NGH):
    """`E[relu(u)relu(v)]` для гаусової пари — формула Прайса при ненульових
    середніх.

    🚨 ПЕРША ВЕРСІЯ БУЛА ХИБНА і пломба це спіймала: двовимірний Гаусс-Ерміт дав
    16% похибки на діагоналі. Причина та сама, що й уся наша стіна — **злам ReLU**:
    для негладкого підінтегрального ГЕ збігається повільно.

    Лікування: внутрішній вимір береться ТОЧНО,
        `E[relu(v)|z] = μ_v Φ(μ_v/σ_v) + σ_v φ(μ_v/σ_v)`,  μ_v = m₂+s₂ρz,
    а зовнішній має РІВНО ОДИН злам (при `z = −m₁/s₁`), тож інтегруємо
    Гауссом-Лежандром по `[злам, 9]`, де підінтегральна вже гладка.
    """
    dev = mp.device
    x, w = np.polynomial.legendre.leggauss(96)
    X = torch.tensor(x, dtype=torch.float64, device=dev)
    Wt = torch.tensor(w, dtype=torch.float64, device=dev)
    m1, m2 = mp[ii], mp[jj]
    s1, s2 = s[ii], s[jj]
    rho = (cp[ii, jj] / (s1 * s2)).clamp(-0.999999, 0.999999)
    sv = (s2 * (1.0 - rho * rho).clamp_min(1e-12).sqrt()).clamp_min(1e-30)
    lo = torch.maximum(-m1 / s1, torch.full_like(m1, -9.0))
    hi = torch.full_like(lo, 9.0)
    half = (hi - lo) * 0.5
    z = (half.view(-1, 1) * X.view(1, -1) + ((hi + lo) * 0.5).view(-1, 1))
    u = torch.relu(m1.view(-1, 1) + s1.view(-1, 1) * z)
    muv = m2.view(-1, 1) + (s2 * rho).view(-1, 1) * z
    tv = muv / sv.view(-1, 1)
    ev = muv * _Phi(tv) + sv.view(-1, 1) * _phi(tv)
    integ = u * ev * _phi(z)
    return (integ * Wt.view(1, -1)).sum(1) * half * (half > 0)


def truth_all(wc, layers, pairs, dev, seed, npair=262_144, chunk=32768):
    ns = [wc[i].shape[1] for i in range(len(wc))]
    acc = {i: {'a': torch.zeros(ns[i], dtype=torch.float64, device=dev),
               'ind': torch.zeros(ns[i], dtype=torch.float64, device=dev),
               'a2': torch.zeros(ns[i], dtype=torch.float64, device=dev),
               'pr': torch.zeros(pairs[i][0].numel(), dtype=torch.float64,
                                 device=dev)} for i in layers}
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = x @ w
            if i in acc:
                acc[i]['ind'] += (p > 0).double().sum(0)
            x = torch.relu(p)
            if i in acc:
                xd = x.double()
                acc[i]['a'] += xd.sum(0)
                acc[i]['a2'] += (xd * xd).sum(0)
                ii, jj = pairs[i]
                acc[i]['pr'] += (xd[:, ii] * xd[:, jj]).sum(0)
        done += k
    n = 2.0 * done
    return {i: {k: v / n for k, v in acc[i].items()} for i in acc}


def pick_pairs(wc, lay, dev, npick=NPAIR_SEL, rand=False):
    """Пари, найважчі за вагою НАСТУПНОГО шару: |W W^T| поза діагоналлю."""
    n = wc[lay].shape[1]
    if rand:
        g = torch.Generator(device=dev).manual_seed(555 + lay)
        ii = torch.randint(0, n, (npick,), device=dev, generator=g)
        jj = torch.randint(0, n, (npick,), device=dev, generator=g)
        return ii, jj
    Wn = wc[lay + 1].double() if lay + 1 < len(wc) else wc[lay].double().T
    G = (Wn @ Wn.T).abs()
    G.fill_diagonal_(0.0)
    flat = torch.topk(G.reshape(-1), npick * 2).indices[::2]
    return flat // n, flat % n


def main(nets=12, dev=DEV):
    torch.manual_seed(1234)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    FAM = ('E[a]', 'індик.', 'E[a²]', 'пара', '⛔пара вип.')
    rr = {f: {lay: [] for lay in PROBE} for f in FAM}
    seal = {'квадратура': [], 'істина': []}

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            lays = [l - 1 for l in PROBE]          # індекси шарів у `wc`
            pr_sel = {l: pick_pairs(wc, l, dev) for l in lays}
            pr_rnd = {l: pick_pairs(wc, l, dev, rand=True) for l in lays}
            allp = {l: (torch.cat([pr_sel[l][0], pr_rnd[l][0]]),
                        torch.cat([pr_sel[l][1], pr_rnd[l][1]])) for l in lays}
            CL = closure_full(wc, lays, dev)
            TR = truth_all(wc, lays, allp, dev, 830_000 + 7919 * n)
            for l in lays:
                ii, jj = allp[l]
                CL[l]['pr'] = price_pairs(CL[l]['mp'], CL[l]['s'], CL[l]['cp'],
                                          ii, jj)
                # ✅ПЛОМБА: на діагоналі квадратура = замкнута форма E[a²]
                d = torch.arange(min(64, wc[l].shape[1]), device=dev)
                q = price_pairs(CL[l]['mp'], CL[l]['s'], CL[l]['cp'], d, d)
                seal['квадратура'].append(
                    float(((q - CL[l]['a2'][d]).abs()
                           / CL[l]['a2'][d].abs().clamp_min(1e-30)).max()))

            S = {l: {k: [] for k in ('a', 'ind', 'a2', 'pr')} for l in lays}
            for dd in range(NDRAW):
                sd = 99_000_000 + 733 * n + dd
                g = torch.Generator(device=dev).manual_seed(int(sd))
                z = torch.randn(npair, W_, device=dev, generator=g)
                x = torch.cat([z, -z], 0)
                for i, w in enumerate(wc):
                    p = x @ w
                    if i in S:
                        S[i]['ind'].append((p > 0).double().mean(0))
                    x = torch.relu(p)
                    if i in S:
                        xd = x.double()
                        S[i]['a'].append(xd.mean(0))
                        S[i]['a2'].append((xd * xd).mean(0))
                        ii, jj = allp[i]
                        S[i]['pr'].append((xd[:, ii] * xd[:, jj]).mean(0))
            for k, l in enumerate(lays):
                lay = PROBE[k]
                for fam, key in (('E[a]', 'a'), ('індик.', 'ind'),
                                 ('E[a²]', 'a2')):
                    M = torch.stack(S[l][key])
                    sd_ = M.std(0).clamp_min(1e-30)
                    r = (CL[l][key] - TR[l][key]).abs() / sd_
                    rr[fam][lay].append(r.cpu().numpy())
                M = torch.stack(S[l]['pr'])
                sd_ = M.std(0).clamp_min(1e-30)
                r = (CL[l]['pr'] - TR[l]['pr']).abs() / sd_
                rr['пара'][lay].append(r[:NPAIR_SEL].cpu().numpy())
                rr['⛔пара вип.'][lay].append(r[NPAIR_SEL:].cpu().numpy())
                # ✅ПЛОМБА істини: E[a_i·a_i] проти E[a_i²] на діагональних парах
                same = (allp[l][0] == allp[l][1])
                if bool(same.any()):
                    i0 = allp[l][0][same]
                    seal['істина'].append(float(
                        ((TR[l]['pr'][same] - TR[l]['a2'][i0]).abs()
                         / TR[l]['a2'][i0].abs().clamp_min(1e-30)).max()))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж × {NDRAW} жеребів\n')
    print(f'✅ПЛОМБА квадратури (діагональ проти замкнутої форми): '
          f'макс. відн. похибка {max(seal["квадратура"]):.2e}')
    if seal['істина']:
        print(f'✅ПЛОМБА істини: {max(seal["істина"]):.2e}')
    print('\n🔑 МЕДІАНА `r = |замикання − істина| / std` ПОЕЛЕМЕНТНО '
          '(усадка = 1/(1+r²))')
    print(f'{"сім\'я":>12s} ' + ' '.join(f'{"шар " + str(l):>9s}' for l in PROBE))
    for fam in FAM:
        print(f'{fam:>12s} ' + ' '.join(
            f'{np.median(np.concatenate(rr[fam][l])):9.2f}' for l in PROBE))
    print(f'\nЧАСТКА ЕЛЕМЕНТІВ ІЗ `r < 1` (усадка > 0.5 ⟹ КЗ реально працює)')
    print(f'{"сім\'я":>12s} ' + ' '.join(f'{"шар " + str(l):>9s}' for l in PROBE))
    for fam in FAM:
        print(f'{fam:>12s} ' + ' '.join(
            f'{float((np.concatenate(rr[fam][l]) < 1.0).mean()):9.3f}'
            for l in PROBE))
    print('\n🔑 читати так: якщо в «пара» `r` близьке до `E[a²]`, а не до `E[a]`, '
          'то парні статистики ДОСТУПНІ — і вони єдині, хто адресує '
          'задокументовані ~70% похибки в кореляціях.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 12)
