"""γ-ПОПРАВКА З МАСКОЮ ПО |t| — ВИПРАВЛЯТИ ЛИШЕ ТАМ, ДЕ РЯД ВАЛІДНИЙ. (С87)

## ЗВІДКИ — ДВА ФАКТИ, ЩО ДОСІ ЛЕЖАЛИ ОКРЕМО
1. `deadneu87` (сьогодні): похибка замикання концентрується в ПЕРЕХІДНІЙ ЗОНІ
   `|t| < 2` — **96.3% похибки на 31.6% нейронів** (концентрація ×3.05).
   При `t > 2` похибка `E[ReLU]` = 0.017% (майже точно), при `t < −2` відносна
   похибка 152%, але ВНЕСОК 2% (бо `E[ReLU]` там крихітне).
2. Ряд Еджворта РОЗБІГАЄТЬСЯ на великому `|t|`: параметр розкладу
   `|γ₃·He₃(−t)/6|` = **1.50 при t=3** (>1 ⟹ розбіжність), але **0.10 при t=2**.

⟹ 🔑 МИ ЗАВЖДИ ЗАСТОСОВУВАЛИ `γ` ДО ВСІХ НЕЙРОНІВ — включно з тими, де ряд
розбігається і де похибки все одно немає. Саме розбіжні члени й псують
результат (`gamalpha87`: γ на всіх нейронах = ×0.521 після α).

## 🎯 СПОСІБ, ЯКИЙ ЦЕ ПРОПОНУЄ
Застосовувати `γ₃,γ₄` ЛИШЕ там, де `|t| < t_cut`; інакше — чистий гаус.
Це виправляє там, де (а) формула валідна і (б) сидить похибка; і не чіпає там,
де ряд розбігається.
💰ЦІНА: одна маска на шар = `n` порівнянь ⟹ ~0.001% бюджету. Гейт пройдено.

## ⛔ПЛОМБИ
 P1 `t_cut = ∞` мусить відтворити `gamalpha87` G2 (×0.521 після α).
 P2 `t_cut = 0` мусить дати РІВНО базу (γ вимкнена всюди).
 P3 ⛔ІНВЕРТОВАНА МАСКА (`|t| > t_cut`) — мусить бути ГІРШОЮ. Якщо краща,
    механізм зрозумілий навпаки і гіпотеза хибна.
 P4 α-калібровка LOO по мережах на кожній сходинці.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) крива по `t_cut` матиме МАКСИМУМ усередині (не на кінцях). Шанс 55%.
 (б) найкращий `t_cut` вийде 1.5–2.5. Шанс 50%.
 (в) виграш ≥×1.05 проти бази ПІСЛЯ α. Шанс **35%**.
 (г) виграш ≥×1.3. Шанс 12%.
 ⚠️Ризик: `γ` оракульна. Якщо навіть вона не дає — маска не врятує.
"""
import glob, math, sys
import numpy as np, torch
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N; BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
CUTS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 99.0)
assert DEV not in ('cuda:0', 'cuda:4')
print(__doc__.split('📣')[1], flush=True)
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
SUB97 = 1.873e-06


def chain(W, G, cut=None, invert=False, orc_ms=None, ortho=False):
    """γ застосовується ЛИШЕ де `|t| < cut` (або `>` при invert).

    🔑 `ortho=True`: поправка ОРТОГОНАЛІЗУЄТЬСЯ до базового прогнозу на кожному
    кроці. Мотив: `gcut87` показав, що γ дає ПРАВИЛЬНИЙ МАСШТАБ, але хибну
    форму, і конкурує з `_ALPHA` за ту саму масштабну складову (сира похибка
    падає ×1.48, після α росте). Забравши в поправки складову ВЗДОВЖ базового
    вектора, лишаємо чисту ФОРМУ — те, чого α дати не може."""
    n = W[0].shape[1]
    C = W[0].T @ W[0]
    m = torch.zeros(n, dtype=torch.float64, device=W[0].device)
    for i in range(1, len(W) + 1):
        s_own = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        m_u, s = (orc_ms[i - 1] if orc_ms is not None else (m, s_own))
        t = m_u / s
        P, f = Phi(t), phi(t)
        if G is not None and cut is not None:
            g3, g4 = G[i - 1]
            msk = (torch.abs(t) > cut) if invert else (torch.abs(t) < cut)
            g3 = torch.where(msk, g3, torch.zeros_like(g3))
            g4 = torch.where(msk, g4, torch.zeros_like(g4))
        else:
            g3 = g4 = torch.zeros_like(s)
        G3_, G4_ = -t * f, (t * t - 1.0) * f
        G6_ = (t ** 4 - 6.0 * t * t + 3.0) * f
        m_b = s * (t * P + f)                      # базовий (гаусів) прогноз
        corr = s * ((g3 / 6.0) * G3_ + (g4 / 24.0) * G4_
                    + (g3 * g3 / 72.0) * G6_)
        if ortho:
            den = float((m_b * m_b).sum())
            if den > 1e-30:
                corr = corr - m_b * (float((corr * m_b).sum()) / den)
        m_a = m_b + corr
        if i == len(W):
            return m_a
        s2 = s * s
        q2 = ((m_u * m_u + s2) * P + m_u * s * f + (g3 / 3.0) * s2 * f
              - (g4 / 12.0) * s2 * t * f
              + (g3 * g3 / 36.0) * s2 * (3.0 * t - t ** 3) * f)
        Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30)
                             - torch.diagonal(Ca))
        m = m_a @ W[i]
        C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} · {DEV}\n', flush=True)
TAGS = (['база'] + [f'cut={c:g}' for c in CUTS]
        + [f'⛔ІНВ cut={c:g}' for c in (2.0, 3.0)]
        + [f'⊥ORTHO cut={c:g}' for c in (1.5, 2.0, 3.0, 99.0)]
        + ['⛔⊥ЧУЖА γ cut=2', '⛔⊥ЗНАК −γ cut=2'])
MS = {t: np.zeros(len(FS)) for t in TAGS}
PT = {t: np.zeros(len(FS)) for t in TAGS}
PP = {t: np.zeros(len(FS)) for t in TAGS}
TT = np.zeros(len(FS)); NRM2 = np.zeros(len(FS))
GPREV = None; GX = []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV) for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(1500 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]; hd = h.double(); p = hd
                for kk in range(4):
                    MOM[q, kk] += p.sum(0); p = p * hd
                a = torch.relu(h)
            acc += a.sum(0, dtype=torch.float64); d += b
        tru = acc / NS; MOM /= NS
        NRM2[n] = float((tru ** 2).mean()); TT[n] = float((tru * tru).sum())
        G = []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q, 0], MOM[q, 1], MOM[q, 2], MOM[q, 3]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = m4 - 4.0*m1*m3 + 6.0*m1*m1*m2 - 3.0*m1**4 - 3.0*c2*c2
            s = torch.sqrt(c2); G.append((c3 / s ** 3, c4 / s ** 4))
        runs = {'база': chain(W, None)}
        for c in CUTS:
            runs[f'cut={c:g}'] = chain(W, G, cut=c)
        for c in (2.0, 3.0):
            runs[f'⛔ІНВ cut={c:g}'] = chain(W, G, cut=c, invert=True)
        for c in (1.5, 2.0, 3.0, 99.0):
            runs[f'⊥ORTHO cut={c:g}'] = chain(W, G, cut=c, ortho=True)
        # ⛔ВИРІШАЛЬНИЙ КОНТРОЛЬ: та сама ортогоналізація, але γ ЧУЖОЇ мережі.
        #    Якщо він теж дає ×1.6 — працює сама ортогоналізація (прибирання
        #    масштабного шуму), а не інформація в γ, і виграш фальшивий.
        GX.append(None)
        runs['⛔⊥ЧУЖА γ cut=2'] = chain(W, GPREV if GPREV is not None else G,
                                        cut=2.0, ortho=True)
        runs['⛔⊥ЗНАК −γ cut=2'] = chain(
            W, [(-a, -b) for a, b in G], cut=2.0, ortho=True)
        GPREV = G
        for t, pr in runs.items():
            MS[t][n] = float(((pr - tru) ** 2).mean())
            PT[t][n] = float((pr * tru).sum()); PP[t][n] = float((pr * pr).sum())
    print(f'  net{n} · база {math.sqrt(MS["база"][n]/NRM2[n]):.4%} · '
          f'cut2 {math.sqrt(MS["cut=2"][n]/NRM2[n]):.4%}', flush=True)


def am(t):
    out = np.zeros(len(FS)); a_n = PT[t] / np.maximum(PP[t], 1e-30)
    for i in range(len(FS)):
        msk = np.arange(len(FS)) != i
        a = a_n[msk].mean() if len(FS) > 1 else a_n[i]
        out[i] = (a*a*PP[t][i] - 2.0*a*PT[t][i] + TT[i]) / W_
    return out.mean()


b = am('база')
print(f'\n{"="*70}\n⛔ПЛОМБИ')
print(f' P2 cut=0 ≡ база: {abs(am("cut=0")-b):.3e}  '
      f'{"OK" if abs(am("cut=0")-b) < 1e-14 else "ПРОВАЛ"}')
print(f' P1 cut=99 (γ всюди) = ×{b/am("cut=99"):.3f}  (gamalpha87 давав ×0.521)')
print(f'\n{"="*70}\nγ З МАСКОЮ |t| < cut — ПІСЛЯ α')
print(f'{"варіант":16s} {"віднос.":>9s} {"ПІСЛЯ α":>11s} {"vs база":>9s} {"vs sub97":>9s}')
for t in TAGS:
    a_ = am(t)
    print(f'{t:16s} {math.sqrt(MS[t].mean()/NRM2.mean()):8.4%} {a_:11.3e} '
          f'{"x%.3f" % (b/a_):>9s} {"x%.3f" % (SUB97/a_):>9s}')
bc = max((b/am(f'cut={c:g}'), c) for c in CUTS)
print(f'\nВИРОК cut-маска: найкращий cut={bc[1]:g} дає x{bc[0]:.4f}')
bo = max((b / am(f'⊥ORTHO cut={c:g}'), c) for c in (1.5, 2.0, 3.0, 99.0))
xg = b / am('⛔⊥ЧУЖА γ cut=2'); xs = b / am('⛔⊥ЗНАК −γ cut=2')
print(f'\n🔑 ОРТОГОНАЛІЗАЦІЯ: найкраще cut={bo[1]:g} -> x{bo[0]:.4f}')
print(f'   ⛔чужа γ x{xg:.4f} · ⛔знак −γ x{xs:.4f}')
if bo[0] >= 1.15 and bo[0] > 1.3 * max(xg, xs):
    print('   ЖИВЕ: виграш саме вiд IНФОРМАЦIЇ в γ, не вiд самої процедури')
else:
    print('   ПРОВАЛ КОНТРОЛЮ: працює процедура, а не γ')
print(f'   inv cut=2 x{b/am("ІНВ cut=2".replace("ІНВ","⛔ІНВ")):.4f} — мусить бути ГІРШИМ')
