"""ОРТОГОНАЛІЗОВАНА ПОПРАВКА ДО σ — ЧИ ОЖИВЕ `L6`? (С87)

## ЗВІДКИ — МЕХАНІЗМ, ЗНАЙДЕНИЙ СЬОГОДНІ
`gcut87` довів: поправка, що конкурує з `_ALPHA` за МАСШТАБ, програє їй.
Розв'язок — проєкція в ортогональне доповнення базового прогнозу:
`γ` прямо ×0.523 → ортогоналізована ×1.022 (механізм дає ×1.95).

🔑 ЦЕ МАЄ СТОСУВАТИСЬ НЕ ЛИШЕ `γ`. Перевіряю на `σ`:
```
δm_a/δσ = φ(t)   ⟹  поправка до σ дає ВЕКТОР `φ(t)·δσ`, у якого Є ФОРМА
```
`L6` (оракульна σ у ланцюг) давала ×0.959 — «нейтральна». Але вона теж НЕ
ортогоналізована ⟹ могла втрачати свою форму в конкуренції з α.
⚠️Контраст: шарові `k_σ` — ЧИСТО масштабна поправка, її ортогональна складова
РІВНО НУЛЬ, тому там ортогоналізація не допоможе за побудовою (×0.119 і
лишиться). Це передбачення механізму, і воно перевіряється тут же.

## 🎯 СХОДИНКИ
```
база                          наше замикання
S1  σ оракульна ПРЯМО         має відтворити L6 ×0.959
S2  σ оракульна ⊥ОРТО         🎯 головне число
S3  σ вибіркова (N=6000) ⊥    реальний режим
S4  k_σ шаровий ⊥             ⛔мусить лишитись ×0.1 (передбачення механізму)
⛔S5 σ ЧУЖОЇ мережі ⊥          контроль
```

## ⛔ПЛОМБИ
 P1 S1 мусить відтворити ×0.96 із `layerk87` (інакше не той об'єкт).
 P2 S4 мусить лишитись ×0.1 — це ПЕРЕДБАЧЕННЯ механізму, не підгонка.
 P3 ⛔чужа σ мусить бути помітно гіршою.
 P4 α-калібровка LOO по мережах.

📣 ПЕРЕДБАЧЕННЯ:
 (а) S2 (оракульна ⊥) дасть ≥×1.2. Шанс 40%.
 (б) S4 лишиться ×0.1-0.3 (масштабна поправка не має ортогональної частини).
     Шанс **85%** — це майже тавтологія, і саме тому добра пломба механізму.
 (в) S3 (вибіркова) дасть ≥×1.05. Шанс 25%.
"""
import glob, math, sys
import numpy as np, torch
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NSAMP = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
NTRUE = 1 << (int(sys.argv[4]) if len(sys.argv) > 4 else 19)
BATCH = 1 << 13; L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4')
print(__doc__.split('📣')[1], flush=True)
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)


def chain(W, SO=None, ortho=False, layer_scalar=False):
    """`SO[i]` — цільова σ шару i. Поправка `δm = φ(t)·(σ_ціль − σ_своя)`.

    `ortho=True` — проєкція поправки в ортогональне доповнення `m_b`.
    `layer_scalar=True` — з `SO` беремо ЛИШЕ масштаб (перевірка передбачення,
    що чисто масштабна поправка ортогональної складової не має)."""
    C = W[0].T @ W[0]
    m = torch.zeros(W_, dtype=torch.float64, device=W[0].device)
    for i in range(1, L_ + 1):
        s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        t = m / s; P, f = Phi(t), phi(t)
        m_b = m * P + s * f
        corr = torch.zeros_like(m_b)
        if SO is not None:
            so = SO[i - 1]
            if layer_scalar:
                k = float((s * so).sum() / torch.clamp((s * s).sum(), min=1e-30))
                so = s * k
            corr = f * (so - s)
            if ortho:
                den = float((m_b * m_b).sum())
                if den > 1e-30:
                    corr = corr - m_b * (float((corr * m_b).sum()) / den)
        m_a = m_b + corr
        if i == L_:
            return m_a
        q2 = (m * m + s * s) * P + m * s * f
        Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30) - torch.diagonal(Ca))
        m = m_a @ W[i]; C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · N={NSAMP} · {DEV}\n', flush=True)
TAGS = ('база', 'S1 σ оракул ПРЯМО', 'S2 σ оракул ⊥ОРТО', 'S3 σ вибірка ⊥',
        'S4 k_σ шаровий ⊥', '⛔S5 σ чужої ⊥')
P = {t: np.zeros((len(FS), W_)) for t in TAGS}
TRUE_ = np.zeros((len(FS), W_)); SALL = []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV) for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        S1_ = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        S2_ = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(6600 + n); d = 0
        while d < NTRUE:
            b = min(BATCH, NTRUE - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]; hd = h.double()
                S1_[q] += hd.sum(0); S2_[q] += (hd * hd).sum(0)
                a = torch.relu(h)
            acc += a.sum(0, dtype=torch.float64); d += b
        TRUE_[n] = (acc / NTRUE).cpu().numpy()
        mo = S1_ / NTRUE
        SO = [torch.sqrt(torch.clamp(S2_[q] / NTRUE - mo[q] ** 2, min=1e-30)) for q in range(L_)]
        gw = torch.Generator(device=DEV); gw.manual_seed(7700 + n)
        z = torch.randn(NSAMP // 2, W_, device=DEV, generator=gw)
        a = torch.cat([z, -z], 0)
        s1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        s2 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        for q in range(L_):
            h = a @ W32[q]; hd = h.double()
            s1[q] += hd.sum(0); s2[q] += (hd * hd).sum(0)
            a = torch.relu(h)
        ms = s1 / NSAMP
        SH = [torch.sqrt(torch.clamp(s2[q] / NSAMP - ms[q] ** 2, min=1e-30)) for q in range(L_)]
        SALL.append([x.cpu() for x in SH])
        P['база'][n] = chain(W).cpu().numpy()
        P['S1 σ оракул ПРЯМО'][n] = chain(W, SO).cpu().numpy()
        P['S2 σ оракул ⊥ОРТО'][n] = chain(W, SO, ortho=True).cpu().numpy()
        P['S3 σ вибірка ⊥'][n] = chain(W, SH, ortho=True).cpu().numpy()
        P['S4 k_σ шаровий ⊥'][n] = chain(W, SO, ortho=True, layer_scalar=True).cpu().numpy()
    print(f'  net{n}', flush=True)
for n in range(len(FS)):
    sd0 = int(np.load(FS[n])['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float64), device=DEV) for q in range(L_)]
    SX = [x.to(DEV) for x in SALL[(n + 1) % len(FS)]]
    with torch.no_grad():
        P['⛔S5 σ чужої ⊥'][n] = chain(W, SX, ortho=True).cpu().numpy()


def am(t):
    pt = (P[t]*TRUE_).sum(1); pp = (P[t]*P[t]).sum(1); tt = (TRUE_*TRUE_).sum(1)
    a_n = pt/np.maximum(pp,1e-30); out=np.zeros(len(FS))
    for i in range(len(FS)):
        msk = np.arange(len(FS))!=i
        a = a_n[msk].mean() if len(FS)>1 else a_n[i]
        out[i] = (a*a*pp[i]-2*a*pt[i]+tt[i])/W_
    return out.mean()


b = am('база')
print(f'\n{"="*66}\nПОПРАВКА ДО σ, ПІСЛЯ α')
print(f'{"варіант":20s} {"ПІСЛЯ α":>11s} {"vs база":>9s}')
for t in TAGS:
    print(f'{t:20s} {am(t):11.3e} {"x%.3f" % (b/am(t)):>9s}')
print(f'\nПЛОМБА P2 (передбачення механiзму): k_σ шаровий ⊥ = x{b/am("S4 k_σ шаровий ⊥"):.3f}')
print('   мусить лишитись ~x0.1-0.3, бо чисто масштабна поправка')
print('   ортогональної складової НЕ МАЄ за побудовою')
print(f'\nВИРОК: S2 (оракул ⊥) x{b/am("S2 σ оракул ⊥ОРТО"):.3f} · '
      f'S3 (вибірка ⊥) x{b/am("S3 σ вибірка ⊥"):.3f} · '
      f'⛔чужа x{b/am("⛔S5 σ чужої ⊥"):.3f}')
