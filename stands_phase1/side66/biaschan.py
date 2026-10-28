"""ЗМІЩЕННЯ ЗАМИКАННЯ ЯК ОКРЕМА ОЗНАКА — БЕЗ ІСТИНИ (С84)

## ЗВІДКИ ЦЕ НАРОДИЛОСЬ (з ПРОВАЛУ, а не з успіху)
`side66/edgeksig.py`: поправка Еджворта робить `m_cl` точнішою — і канал від
цього ГІРШАЄ (×0.875 на оптимумах, ×0.779 при c=1). Контролі чисті: при c=0
усі варіанти збігаються ТОЧНО, обернений знак теж гірший (×0.927).
🔑 МЕХАНІЗМ: ознака `δ = x̄ − m_cl` = ШУМ ВИБІРКИ + ЗМІЩЕННЯ ЗАМИКАННЯ, і
похибка ВИХОДУ містить те саме накопичене зміщення ⟹ `δ` корелює з ціллю
через СПІЛЬНУ СИСТЕМАТИКУ, а не лише через шум. Прибравши зміщення, я забрав
у каналу половину роботи.
✅ Це пояснює аномалію С75: ⛔НЕГ (ознаки з СУСІДНЬОГО жеребу) давав 1.031× —
БІЛЬШЕ за справжні ([[finding-only-the-mask-family-stays-clean-at-depth]]).

## 🎯 ПРОТИЛЕЖНИЙ ХІД
С75 записав: «~1.03× може лежати в суто зміщенній частині, але вона НЕ
доступна в рантаймі — потрібне істинне значення». **Тепер доступна:**
    `D_i = m_едж(i) − m_база(i)`
— аналітична оцінка похибки замикання на шарі `i`, БЕЗ істини й БЕЗ вибірки.
Детермінована (не додає шуму), а опори на всіх 31 шарах у сабміті вже
рахуються ([[reference-sub91-torch-port-is-the-real-base]]).
⟹ не ЧИСТИТИ ознаку, а зробити зміщення ОКРЕМИМ каналом.

## ЩО МІРЯЄМО
`pr = базовий_сабміт − Σ_i μ · D_i↦вихід`, де `↦` — той самий ланцюг воріт
`v ← (v@W)·gate`, що переносить усі інші ознаки. `μ` — ОДИН скаляр на зріз
шарів, підбирається **LOO по мережах** (фіт на всіх, крім поточної).

⛔ ПЛОМБИ І КОНТРОЛІ:
 (а) `μ=0` мусить дати РІВНО базовий MSE;
 (б) ⛔КОНТРОЛЬ «ЧУЖА МЕРЕЖА»: `D` від сусідньої мережі. 🚨 НЕГ по ЖЕРЕБУ тут
     НЕ Є нулем (С75) — `D` взагалі не залежить від жеребу, тож єдиний чесний
     контроль руйнує МЕРЕЖЕВУ складову;
 (в) ⛔КОНТРОЛЬ «ШУМ»: `D` замінена на випадковий вектор тієї ж норми;
 (г) істина — БАНК, крос-ціль `(p−T1)(p−T2)`.

📣 ПЕРЕДБАЧЕНО: 🚨 попереднє передбачення (1.01–1.05× для чистішої `m_cl`)
   ПРОВАЛИЛОСЬ у протилежний бік, тож тут кажу обережніше й з механізмом.
   `D` — це оцінка зміщення, а не саме зміщення: Еджворт бере лише перші два
   кумулянти, і на шарі 2 він знімає ~47% похибки (2.58e-3 → 1.375e-3), на
   шарі 8 ~34%, на шарі 21 ~10%. Якщо канал справді живе зі зміщення, `μ`
   вийде ДОДАТНИМ і виграш буде в межах **1.005–1.03×** — тобто на рівні
   шумової підлоги LOO (≈1.03× за С75), і головним буде не число, а те, чи
   переживе воно контроль «чужа мережа».
   Якщо `μ` вийде ≈0 або контроль дасть стільки ж — механізм не той.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/biaschan.py [cuda:N] [NETS] [log2N]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS,
                        ALPHA, W_BLEND)

def true_cumulants(wc, dev, ns, bs):
    """ІСТИННІ γ₁,γ₂ маргіналів `h` на КОЖНОМУ шарі, крізь ОБРІЗАНІ ваги.

    🚨 Копія з `edgeksig.py`, а не імпорт: там код на верхньому рівні, і
    `import` запустив би весь той стенд.
    """
    d = len(wc)
    S1 = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
    S2 = [torch.zeros_like(s) for s in S1]
    S3 = [torch.zeros_like(s) for s in S1]
    S4 = [torch.zeros_like(s) for s in S1]
    gen = torch.Generator(device=dev); gen.manual_seed(4242)
    done = 0
    while done < ns:
        b = min(bs, ns - done)
        a = torch.randn(b, W_, device=dev, generator=gen)
        for i in range(d):
            h = a @ wc[i]
            hd = h.double()
            S1[i] += hd.sum(0); S2[i] += (hd * hd).sum(0)
            S3[i] += (hd ** 3).sum(0); S4[i] += (hd ** 4).sum(0)
            a = torch.relu(h)
        done += b
    g1, g2 = [], []
    for i in range(d):
        mh = S1[i] / ns
        v = (S2[i] / ns - mh * mh).clamp_min(1e-30)
        k3 = S3[i] / ns - 3 * mh * (S2[i] / ns) + 2 * mh ** 3
        k4 = (S4[i] / ns - 4 * mh * (S3[i] / ns) + 6 * mh ** 2 * (S2[i] / ns)
              - 3 * mh ** 4 - 3 * v * v)
        g1.append(k3 / v ** 1.5); g2.append(k4 / v ** 2)
    return g1, g2



DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LG = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NS, BS, L_, NDRAW = 1 << LG, 1 << 13, 32, 8
SLICES = [(1, 9), (1, 17), (1, 32), (9, 32), (17, 32)]   # зрізи шарів для D
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж × {NDRAW} жеребів, {DEV}\n', flush=True)


def closure_both(wc, dev, g1, g2):
    """База і Еджворт ОДНИМ проходом ⟹ `D_i = m_едж − m_база` на КОЖНОМУ шарі.

    🔑 Обидва ланцюги ведуться ПАРАЛЕЛЬНО і незалежно; `D` береться як різниця
    `m` на однакових шарах. Опори для каналів — з БАЗИ (у сабміті так і є).
    """
    n0 = wc[0].shape[0]
    mb = torch.zeros(n0, dtype=torch.float64, device=dev)
    Cb = torch.eye(n0, dtype=torch.float64, device=dev)
    me = mb.clone(); Ce = Cb.clone()
    s_o, m_o, q_o, p_o, D = [], [], [], [], []
    for i in range(len(wc)):
        w = wc[i].double()
        for k in range(2):
            m, C = (mb, Cb) if k == 0 else (me, Ce)
            cp = w.T @ (C @ w)
            s = cp.diagonal().clamp_min(1e-20).sqrt()
            mp = m @ w
            t = mp / s
            ph, Ph = _phi(t), _Phi(t)
            mo = mp * Ph + s * ph
            d2 = (mp * mp + s * s) * Ph + mp * s * ph
            if k == 1:
                mo = (mo - (g1[i] / 6.0) * mp * ph
                      + (g2[i] / 24.0) * s * (t * t - 1.0) * ph)
                d2 = (d2 + (g1[i] / 3.0) * s * s * ph
                      - (g2[i] / 12.0) * s * s * t * ph)
            if k == 0 and 1 <= i <= KSIG:
                s_o.append(s); p_o.append(Ph)
            if i < len(wc) - 1:
                Cn = (cp * torch.outer(Ph, Ph)
                      + (0.5 * cp * cp) * torch.outer(ph / s, ph / s))
                Cn = Cn + torch.diag((d2 - mo * mo) - Cn.diagonal())
            else:
                Cn = C
            if k == 0:
                mb, Cb = mo, Cn
                if 1 <= i <= KSIG:
                    m_o.append(mo); q_o.append(Cn.diagonal() + mo * mo)
            else:
                me, Ce = mo, Cn
        D.append(me - mb)
    return (s_o, m_o, q_o, p_o, mb), D


def run(wc, npair, dev, seed, cl, Dl):
    """Прогін сабміту + перенесення матриці `Dl` (шар×256) до виходу."""
    depth = len(wc)
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                     device=dev)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    y = z @ (wh @ wc[0])
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    den = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / den - sg0 * INV_SQRT_2PI
    s_cl, m_cl, q_cl, P_cl, m_deep = cl
    B = torch.zeros(len(Dl), wc[0].shape[1], dtype=torch.float64, device=dev)
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        B = (B @ wd) * gate.unsqueeze(0)
        for k, (i0, D) in enumerate(Dl):
            if i0 == i:
                B[k] = B[k] + D
        pd = p.double()
        mh = pd.sum(0) / den
        sh = (pd.pow(2).sum(0) / den - mh * mh).clamp_min(1e-30).sqrt()
        x = torch.relu(p)
        xm = x.double().sum(0) / den
        xq = x.double().pow(2).sum(0) / den
        if i - 1 < len(s_cl) and i <= KSIG:
            d = (_phi(mh / sh) * (sh - s_cl[i - 1]), xm - m_cl[i - 1],
                 xq - q_cl[i - 1], gate - P_cl[i - 1])
            v = (v + SP.LAM_S[i - 1] * d[0] + SP.LAM_A[i - 1] * d[1]
                 + SP.LAM_Q[i - 1] * d[2] + SP.LAM_G[i - 1] * d[3])
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    pr = ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)
    return pr, B


NV = 3                       # 0: справжня D · 1: ⛔чужа мережа · 2: ⛔шум
# для кожної мережі/жеребу/зрізу/варіанта зберігаємо скалярні добутки,
# щоб потім підбирати μ БЕЗ повторних прогонів:
#   MSE(μ) = Σ(pr − μB − T1)(pr − μB − T2) = P0 − μ·P1 + μ²·P2
P0 = np.zeros(len(FS)); P1 = np.zeros((len(FS), len(SLICES), NV))
P2 = np.zeros((len(FS), len(SLICES), NV))
SEAL0 = 0.0
DBANK = []

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        g1, g2 = true_cumulants(wc, DEV, NS, BS)
        cl, D = closure_both(wc, DEV, g1, g2)
        # ⛔ПЛОМБА (а): база тут ≡ closure91
        cb = closure91(wc, KSIG, DEV, kdeep=len(wc))
        SEAL0 = max(SEAL0, float((cl[4] - cb[4]).abs().max()))
        DBANK.append([d.clone() for d in D])
        gen = torch.Generator(device=DEV); gen.manual_seed(777 + n)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        # три набори D: справжня, чужа (сусідня мережа), шум
        prev = DBANK[n - 1] if n > 0 else DBANK[0]
        sets = []
        for vi in range(NV):
            lst = []
            for si, (i0, i1) in enumerate(SLICES):
                for i in range(i0, i1):
                    if vi == 0:
                        d = D[i]
                    elif vi == 1:
                        d = prev[i] if D[i].shape == prev[i].shape else D[i] * 0
                    else:
                        r = torch.randn(D[i].shape, dtype=torch.float64,
                                        device=DEV, generator=gen)
                        d = r * (D[i].norm() / r.norm())
                    lst.append((i, d, si))
            sets.append(lst)
        for d in range(NDRAW):
            sd = 93_000_000 + 733 * n + d
            for vi in range(NV):
                Dl = [(i, dd) for i, dd, _ in sets[vi]]
                idx = [si for _, _, si in sets[vi]]
                pr, B = run(wc, npair, DEV, sd, cl, Dl)
                if vi == 0:
                    P0[n] += float(((pr - T1) * (pr - T2)).sum()) / NDRAW
                for si in range(len(SLICES)):
                    bs = B[[k for k, s in enumerate(idx) if s == si]].sum(0)
                    P1[n, si, vi] += float((bs * ((pr - T1) + (pr - T2))).sum()) / NDRAW
                    P2[n, si, vi] += float((bs * bs).sum()) / NDRAW
    del W, wc
    if n > 1:
        DBANK[n - 2] = None
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/biaschan.npz',
         P0=P0, P1=P1, P2=P2, SL=np.array(SLICES))

print(f'\n⛔ПЛОМБА (а) база стенду ≡ closure91: {SEAL0:.2e} '
      f'{"✅" if SEAL0 < 1e-12 else "🚨"}')
base = P0.mean()
print(f'⛔ПЛОМБА (б) базовий MSE: {base:.5e}\n')
NM = ['🎯 справжня D', '⛔ чужа мережа', '⛔ шум тієї ж норми']
print(f'{"зріз":>10s} {"варіант":>22s} {"μ* (LOO)":>10s} {"MSE":>13s} {"×":>8s}')
print('─' * 70)
best = (1.0, None)
for si, (i0, i1) in enumerate(SLICES):
    for vi in range(NV):
        acc = 0.0
        for n in range(len(FS)):
            o = [k for k in range(len(FS)) if k != n]
            mu = P1[o, si, vi].sum() / max(2.0 * P2[o, si, vi].sum(), 1e-300)
            acc += (P0[n] - mu * P1[n, si, vi] + mu * mu * P2[n, si, vi])
        acc /= len(FS)
        mu_all = P1[:, si, vi].sum() / max(2.0 * P2[:, si, vi].sum(), 1e-300)
        print(f'{f"{i0}..{i1-1}":>10s} {NM[vi]:>22s} {mu_all:10.4f} '
              f'{acc:13.5e} {base/acc:8.4f}')
        if vi == 0 and base / acc > best[0]:
            best = (base / acc, (i0, i1))
    print()
print('🎯 ВИРОК')
print(f'   найкращий зріз справжньої D: {best[1]} ⟹ ×{best[0]:.4f}')
print(f'   🚨 засвідчується ЛИШЕ якщо перевищує ОБИДВА контролі того ж зрізу')
print(f'\n📊 БАЛ 1.954e-07 → {1.954e-07/max(best[0],1e-9):.4e} (якщо переживе контролі)')
