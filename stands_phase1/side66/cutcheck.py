"""ЧОМУ ДВА ЧИСТИХ СТЕНДИ СУПЕРЕЧАТЬ ОДИН ОДНОМУ (С84)

## СУПЕРЕЧНІСТЬ
`selfcon.py` (повні `W`, істина 2¹⁹): Еджворт робить `m` ТОЧНІШОЮ — ×1.88 на
шарі 2, ×1.51 на шарі 8.
`edgeksig.py` (обрізані `wc`, банк): від цього MSE ГІРШАЄ ×0.875, монотонно
з ростом `λ`, при чистих пломбах (γ=0 ≡ база РІВНО, обернений знак теж гірший).
🚨 Обидва не можуть бути правильні водночас. Різниця між ними РІВНО одна:
повні `W` проти ОБРІЗАНИХ `wc` (`cut_budget91` викидає нейрони, у яких
`max h < 0` на пілоті зі 128 пар).

## ГІПОТЕЗА
Обрізання лишає нейрони з `max h ≥ 0`, тобто **умовно відбирає праві хвости**.
Це не безневинна оптимізація: маргінал `h_i` обрізаної мережі — уже НЕ той
розподіл, для якого рахувались `γ`. Якщо так, поправка на `wc` прикладається
до зміщеного об'єкта, і знак виграшу може перевернутись.

## ЩО МІРЯЄМО (той самий вимір, що в `selfcon`, але на `wc`)
Похибка `m` замикання проти ІСТИННОГО `E[a]`, пошарово, для трьох ланцюгів:
БАЗА · +κ₃ · +κ₃+κ₄. Істина — велика вибірка КРІЗЬ `wc`, з тим самим входом,
що в сабміті (вибілений + антитетичні пари ±z), а не просто `randn`.
Плюс контрольний прогін на ПОВНИХ `W` тих самих мереж — щоб побачити обидва
числа поруч, одним приладом.

⛔ ПЛОМБИ: (а) ланцюг ≡ `closure91`; (б) шар 1 — поправка ≈ нуль (там `h`
   точно гаусів); (в) шум істини половинками; (г) частка обрізаних нейронів.

📣 ПЕРЕДБАЧЕНО: якщо на `wc` виграш зникає або перевертається — суперечність
   пояснена обрізанням, і це САМО ПО СОБІ важливо: усі виміри «якості
   замикання» на повних `W` не переносяться на сабміт. Якщо ж на `wc` виграш
   ТОЙ САМИЙ (×1.5–1.9) — то суперечність не тут, і винен спосіб, у який
   ознака входить у прогноз, а не точність `m`.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/cutcheck.py [cuda:N] [NETS] [log2N]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, PILOT_PAIRS)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LG = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS, BS, L_ = 1 << LG, 1 << 13, 32
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def moments(ws, dev, ns, bs, whiten=False):
    """m_a, κ-моменти `h` пошарово. `whiten` — вхід як у сабміті (±z, вибілений)."""
    d = len(ws)
    Z = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in ws]
    S1 = [z.clone() for z in Z]; S2 = [z.clone() for z in Z]
    S3 = [z.clone() for z in Z]; S4 = [z.clone() for z in Z]
    AM = [z.clone() for z in Z]; AH = [z.clone() for z in Z]
    gen = torch.Generator(device=dev); gen.manual_seed(4242)
    done = 0
    while done < ns:
        b = min(bs, ns - done)
        z = torch.randn(b, W_, device=dev, generator=gen)
        if whiten:
            cov = (z.T @ z) / float(b)
            ev, V = torch.linalg.eigh(cov.double())
            z = z @ ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
            a = torch.cat([z, -z], 0)
        else:
            a = z
        nb = a.shape[0]
        for i in range(d):
            h = a @ ws[i]
            hd = h.double()
            S1[i] += hd.sum(0); S2[i] += (hd * hd).sum(0)
            S3[i] += (hd ** 3).sum(0); S4[i] += (hd ** 4).sum(0)
            a = torch.relu(h)
            AM[i] += a.double().sum(0)
            if done < ns // 2:
                AH[i] += a.double().sum(0)
        done += b
        tot = done * (2 if whiten else 1)
    g1, g2, ma, mah = [], [], [], []
    for i in range(d):
        mh = S1[i] / tot
        v = (S2[i] / tot - mh * mh).clamp_min(1e-30)
        k3 = S3[i] / tot - 3 * mh * (S2[i] / tot) + 2 * mh ** 3
        k4 = (S4[i] / tot - 4 * mh * (S3[i] / tot) + 6 * mh ** 2 * (S2[i] / tot)
              - 3 * mh ** 4 - 3 * v * v)
        g1.append(k3 / v ** 1.5); g2.append(k4 / v ** 2)
        ma.append(AM[i] / tot); mah.append(AH[i] / (tot // 2))
    return g1, g2, ma, mah


def chain(ws, dev, g1=None, g2=None):
    """Ланцюг замикання; повертає `m` на КОЖНОМУ шарі."""
    n0 = ws[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    out = []
    for i in range(len(ws)):
        w = ws[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        d2 = (mp * mp + s * s) * Ph + mp * s * ph
        if g1 is not None:
            m = m - (g1[i] / 6.0) * mp * ph
            d2 = d2 + (g1[i] / 3.0) * s * s * ph
        if g2 is not None:
            m = m + (g2[i] / 24.0) * s * (t * t - 1.0) * ph
            d2 = d2 - (g2[i] / 12.0) * s * s * t * ph
        out.append(m)
        if i < len(ws) - 1:
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            C = C + torch.diag((d2 - m * m) - C.diagonal())
    return out


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж × {NS} семплів, {DEV}\n', flush=True)

# [мережа, шар, режим(0=повні W,1=обрізані wc), варіант(0=база,1=κ₃,2=обидва)]
R = np.zeros((len(FS), L_, 2, 3))
RS = np.zeros((len(FS), L_, 2))
KEEP = np.zeros((len(FS), L_))
SEAL = 0.0

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        for i in range(L_):
            KEEP[n, i] = wc[i].shape[1] / W[i].shape[1]
        for mode, (ws, wh) in enumerate(((W, False), (wc, True))):
            g1, g2, ma, mah = moments(ws, DEV, NS, BS, whiten=wh)
            ch = [chain(ws, DEV), chain(ws, DEV, g1, None),
                  chain(ws, DEV, g1, g2)]
            if mode == 0:
                _, _, _, _, md = closure91(W, L_ - 1, DEV, kdeep=None)
                SEAL = max(SEAL, float((ch[0][L_-1] - md).abs().max()))
            for i in range(len(ws)):
                rn = ma[i].norm()
                for v in range(3):
                    R[n, i, mode, v] = float((ch[v][i] - ma[i]).norm() / rn)
                RS[n, i, mode] = float((mah[i] - ma[i]).norm() / rn)
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/cutcheck.npz',
         R=R, RS=RS, KEEP=KEEP)
print(f'\n⛔ПЛОМБА (а) ланцюг ≡ closure91: {SEAL:.2e} '
      f'{"✅" if SEAL < 1e-12 else "🚨"}')
print(f'⛔ПЛОМБА (г) частка ЗБЕРЕЖЕНИХ нейронів: шар 1 {KEEP[:,0].mean():.3f} · '
      f'шар 8 {KEEP[:,7].mean():.3f} · шар 31 {KEEP[:,30].mean():.3f}')
print(f'⛔ПЛОМБА (в) шум істини: повні {np.median(RS[:,:,0]):.2e} · '
      f'обрізані {np.median(RS[:,:,1]):.2e}\n')

print(f'{"шар":>4s} │ {"ПОВНІ W: база":>13s} {"+κ₃+κ₄":>11s} {"×":>7s} │ '
      f'{"ОБРІЗАНІ wc: база":>17s} {"+κ₃+κ₄":>11s} {"×":>7s}')
print('─' * 88)
for i in list(range(0, 8)) + list(range(8, L_, 4)) + [L_ - 1]:
    a0, a2 = R[:, i, 0, 0].mean(), R[:, i, 0, 2].mean()
    b0, b2 = R[:, i, 1, 0].mean(), R[:, i, 1, 2].mean()
    print(f'{i+1:4d} │ {a0:13.3e} {a2:11.3e} {a0/max(a2,1e-30):7.3f} │ '
          f'{b0:17.3e} {b2:11.3e} {b0/max(b2,1e-30):7.3f}')

print('\n🎯 ВИРОК')
for i, nm in ((1, 'шар 2'), (7, 'шар 8'), (L_-1, 'шар 32')):
    gf = R[:, i, 0, 0].mean() / max(R[:, i, 0, 2].mean(), 1e-30)
    gc = R[:, i, 1, 0].mean() / max(R[:, i, 1, 2].mean(), 1e-30)
    print(f'   {nm:7s}: повні ×{gf:.3f}   обрізані ×{gc:.3f}   '
          f'{"🚨 ОБРІЗАННЯ ЗНИЩУЄ ВИГРАШ" if gc < gf * 0.7 else "✅ те саме"}')
print('\n📊 БАЛ 1.954e-07 · діагностика, у сабміт нічого не йде')
