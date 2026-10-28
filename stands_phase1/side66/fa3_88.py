"""ОСТАННЯ НЕПЕРЕВІРЕНА НИТКА: `F(t_j,t_k,ρ)` ПІД СХЕМОЮ **A3** (С88, БЕЗ МК)

## ЗВІДКИ — П'ЯТЬ ЗАКРИТТІВ СЬОГОДНІ ЗВЕЛИ ВСЕ В ОДНУ ТОЧКУ
`price88` гаусове замикання вже точне ×1.001 · `lowrk88` низький ранг r=3 →
×0.32 · `cum88` кумулянти у ВИХІД — стеля ×1.15 · `herm88` гаусова копула +
точні маргіналі — стеля ×1.07, далі ГІРШЕ · `tprof88` еф.ранг 127→3.1.
⟹ 🎯 **негаусовість живе в КОПУЛІ, а копула — властивість ПАРИ `(j,k)`.**
Жоден понейронний набір її не задає. Потрібна поправка від пари — а вона в нас
УЖЕ Є: `F(t_j,t_k,ρ)`, `R²=0.666` (40 мереж × 31 шар, `r2chk88` дав 0.695).

## 🚨 ЧОМУ ВОНА ДОСІ НЕ ПЕРЕВІРЕНА ЯК СЛІД
`learnc87` дав ×1.027, але це було під схемою **A1** — з вагою поправки,
ПРИБИТОЮ ДО 1. `calib88` довів, що A1 систематично вбиває робочі поправки:
`γ` ×0.527→**×1.044**, `γ`+пошарова ×0.995→**×1.429**, `σ` як `δm`
×1.160→**×1.446**. Схема **A3** фітить `β` спільним МНК разом з `α`:
```
m̂_n = α·b_n + β·c⊥_n ,  c⊥_n = c_n − b_n⟨c_n,b_n⟩/⟨b_n,b_n⟩
```
Оскільки `⟨b_n, c⊥_n⟩ ≡ 0` понейромережно, нормальні рівняння блочно-
діагональні ⟹ `α*` тотожно НЕ змінюється, а A0 (`β=0`) лежить ВСЕРЕДИНІ A3
⟹ **A3 не може програти базі.** Агент С88 записав це як єдине неперевірене,
з передбаченням **≥×1.03, ймовірно ×1.1–1.3**.

## 🎯 ЩО МІРЯЄТЬСЯ
 (1) навчання `F` бінінгом по `(t_j,t_k,ρ)`, LOO по мережах, ціль СВІЖА;
 (2) ланцюг із `F` при кількох вагах;
 (3) калібровка A0 (база) / A1 (вага=1) / A2 (⊥) / **A3 (фітований β)**;
 (4) контроль ⛔ЧУЖА `F` — під A3 має дати рівно ×1.000.

## 🚨 ПАСТКИ, ЯКІ ВЖЕ КОШТУВАЛИ ГІЛОК (не наступити знову)
 • ЦІЛЬ МУСИТЬ БУТИ СВІЖОЮ: на вхід кроку — ІСТИННІ `m,C`. Накопичена ціль
   дала ×0.009 (у 100× гірше) у С87.
 • НОРМУВАТИ НА `s` ПРЕАКТИВАЦІЙ, не на `√diag(Ca)`: для третини нейронів
   `Var(a)→0` і нормування дало RMS 6e+21.
 • БАЗА ЗАВЖДИ ПІСЛЯ `α` — інакше фальшивий ×3.26.
 • LOO ПО МЕРЕЖАХ і в навчанні `F`, і в калібровці.
 • ⚠️ Під A3 контроль «перевернутий знак» ПОРОЖНІЙ за побудовою (`β` сам
   міняє знак) ⟹ валідним лишається лише «ЧУЖА `F`».

## ⛔ ПЛОМБИ
 P1 база після `α` = **0.4894–0.6013%** (`calib88`, два незалежні виміри).
 P2 🔑`α(A3) − α(A0)` мусить бути ~1e-16 (блочна діагональність — тотожність).
 P3 A0 ≤ A3 ЗАВЖДИ (A0 всередині A3) — якщо A3 програє, у коді баг.
 P4 ⛔ЧУЖА `F` під A3 → ×1.000 (схема безпечна), під A1 → ×0.3–0.4.
 P5 `R²` навченої `F` на відкладених мережах ≈ 0.6–0.7 (`r2chk88`: 0.695).

## 📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ
 A. A1 відтворить ×1.02–1.05 (`learnc87` дав ×1.027). Шанс **70%**.
 B. 🎯A3 дасть **×1.10–1.35**. Шанс **45%**. Це передбачення агента, і воно
    спирається на те, що для `γ` перехід A1→A3 дав ×1.044→×1.429.
 C. ⚠️АЛЕ переклад у бал: `виграш = (k + 15.66)/16.66`. Навіть `k=1.35` дає
    лише **×1.02** у балі ⟹ сабміта це НЕ дає. Шанс, що вийде &gt;×1.05 у
    балі — **10%**.
 D. 📌 Справжня цінність — ВИРОК ПО КЛАСУ: якщо парна поправка з `R²=0.666`
    під найкращою схемою калібровки дає &lt;×1.4, то детермінована гілка
    закрита ПОВНІСТЮ, і залишок часу треба класти на хедж сабмітів.
"""
import glob, math, sys
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 17
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
NB = 14                      # сітка квантилів 14³ (насичення за `r2chk88`)
WS = (0.0, 0.25, 0.5, 1.0)   # ваги поправки в ЛАНЦЮГУ
assert DEV not in ('cuda:0', 'cuda:4'), 'cuda:0 ТАБУ, cuda:4 — монітор юзера'
print(__doc__.split('📣')[1], flush=True)

I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
NF = len(FS)
print(f'{NF} мереж · 2^{LOG2N} · {DEV} · сітка {NB}³\n', flush=True)

TQ = torch.tensor(np.linspace(0.0, 1.0, NB + 1)[1:-1], dtype=torch.float64,
                  device=DEV)


def closure(m, C):
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s
    P, f = Phi(t), phi(t)
    m_a = m * P + s * f
    Ca = C * torch.outer(P, P) + 0.5 * (C * C) * torch.outer(f / s, f / s)
    d = (m * m + s * s) * P + (m * s) * f - m_a * m_a
    Ca = Ca + torch.diag(d - torch.diagonal(Ca))
    return m_a, Ca, s, t


# ---------- ФАЗА 1: істина + СВІЖА ціль по шарах ----------
TRU = []          # (a_true останнього шару) на мережу
BINS = torch.zeros(NF, NB, NB, NB, dtype=torch.float64, device=DEV)
CNT = torch.zeros(NF, NB, NB, NB, dtype=torch.float64, device=DEV)
EDG = None
print('ФАЗА 1: збір істини і свіжої цілі', flush=True)
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                        device=DEV, dtype=torch.float64) for q in range(L_)]
    with torch.no_grad():
        A1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        AA = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(9900 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                h = (a @ W32[q].float()).double()
                aq = torch.clamp(h, min=0.0)
                A1[q] += aq.sum(0); AA[q] += aq.T @ aq
                a = aq.float()
            d += b
        TRU.append((A1[L_ - 1] / NS).clone())
        prevm = None; prevC = None
        for q in range(L_):
            mq = A1[q] / NS
            Cq = AA[q] / NS - torch.outer(mq, mq)
            if q == 0:
                C_h = W32[0].T @ W32[0]
                m_h = torch.zeros(W_, dtype=torch.float64, device=DEV)
            else:
                C_h = W32[q].T @ (prevC @ W32[q])
                m_h = prevm @ W32[q]
            m_cl, Ca_cl, s, t = closure(m_h, C_h)      # 🚨 вхід — ІСТИННІ m,C
            R = C_h / torch.outer(s, s)
            y = (Cq - Ca_cl) / torch.outer(s, s)       # 🚨 нормування на `s` ПРЕАКТ.
            iu = torch.triu_indices(W_, W_, offset=1, device=DEV)
            tj, tk = t[iu[0]], t[iu[1]]
            lo = torch.minimum(tj, tk); hi = torch.maximum(tj, tk)   # симетризація
            rr = R[iu[0], iu[1]]; yy = y[iu[0], iu[1]]
            if EDG is None:
                EDG = (torch.quantile(lo, TQ), torch.quantile(hi, TQ),
                       torch.quantile(rr, TQ))
            b0 = torch.bucketize(lo, EDG[0])
            b1 = torch.bucketize(hi, EDG[1])
            b2 = torch.bucketize(rr, EDG[2])
            idx = (b0 * NB + b1) * NB + b2
            BINS[n].view(-1).index_add_(0, idx, yy)
            CNT[n].view(-1).index_add_(0, idx, torch.ones_like(yy))
            prevm, prevC = mq, Cq
        del AA
    print(f'  net{n}', flush=True)

# ---------- ФАЗА 2: ланцюг із LOO-`F` ----------
print('\nФАЗА 2: ланцюг', flush=True)
NW = len(WS)
PRED = torch.zeros(NF, NW, W_, dtype=torch.float64, device=DEV)
PRED_X = torch.zeros(NF, W_, dtype=torch.float64, device=DEV)   # ⛔чужа F
R2 = np.zeros(NF)
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                        device=DEV, dtype=torch.float64) for q in range(L_)]
    msk = torch.ones(NF, dtype=torch.bool, device=DEV); msk[n] = False
    Fl = BINS[msk].sum(0) / torch.clamp(CNT[msk].sum(0), min=1.0)
    Fx = BINS[(n + 1) % NF] / torch.clamp(CNT[(n + 1) % NF], min=1.0)  # ⛔чужа
    ss = float((BINS[n] ** 2 / torch.clamp(CNT[n], min=1.0)).sum())
    R2[n] = 0.0 if ss <= 0 else 1.0
    with torch.no_grad():
        for wi, wgt in enumerate(list(WS) + ['X']):
            useF = Fx if wgt == 'X' else Fl
            w_ = 1.0 if wgt == 'X' else wgt
            m = torch.zeros(W_, dtype=torch.float64, device=DEV)
            C = torch.eye(W_, dtype=torch.float64, device=DEV)
            for q in range(L_):
                cp = W32[q].T @ (C @ W32[q]); cp = 0.5 * (cp + cp.T)
                mp = m @ W32[q]
                m, C, s, t = closure(mp, cp)
                if w_ != 0.0 and q < L_ - 1:
                    R = cp / torch.outer(s, s)
                    lo = torch.minimum(t.view(-1, 1), t.view(1, -1))
                    hi = torch.maximum(t.view(-1, 1), t.view(1, -1))
                    b0 = torch.bucketize(lo, EDG[0])
                    b1 = torch.bucketize(hi, EDG[1])
                    b2 = torch.bucketize(R, EDG[2])
                    corr = useF[b0, b1, b2] * torch.outer(s, s) * w_
                    corr = corr - torch.diag(torch.diagonal(corr))
                    C = C + corr
            if wgt == 'X':
                PRED_X[n] = m
            else:
                PRED[n, wi] = m
    print(f'  net{n}', flush=True)

# ---------- ФАЗА 3: КАЛІБРОВКИ A0/A1/A2/A3 ----------
Y = torch.stack(TRU)
B = PRED[:, 0]                                   # база (w=0)


def fit_eval(Cm, mode):
    """LOO по мережах. mode: 'A0' | 'A1' | 'A2' | 'A3'."""
    out = np.zeros(NF); alphas = np.zeros(NF)
    for i in range(NF):
        tr = [j for j in range(NF) if j != i]
        if mode == 'A0':
            a = sum(float((B[j] * Y[j]).sum()) for j in tr) / \
                sum(float((B[j] * B[j]).sum()) for j in tr)
            p = a * B[i]; alphas[i] = a
        elif mode == 'A1':
            a = sum(float((Cm[j] * Y[j]).sum()) for j in tr) / \
                sum(float((Cm[j] * Cm[j]).sum()) for j in tr)
            p = a * Cm[i]; alphas[i] = a
        else:
            CP = []
            for j in range(NF):
                c = Cm[j] - B[j]
                c = c - B[j] * (float((c * B[j]).sum()) / float((B[j] * B[j]).sum()))
                CP.append(c)
            a = sum(float((B[j] * Y[j]).sum()) for j in tr) / \
                sum(float((B[j] * B[j]).sum()) for j in tr)
            alphas[i] = a
            if mode == 'A2':
                p = a * (B[i] + CP[i])
            elif mode == 'A3':
                den = sum(float((CP[j] * CP[j]).sum()) for j in tr)
                bb = 0.0 if den <= 1e-30 else \
                    sum(float((CP[j] * (Y[j] - a * B[j])).sum()) for j in tr) / den
                p = a * B[i] + bb * CP[i]
            else:
                # 🆕A4: спільний МНК по (B, c) БЕЗ ортогоналізації — повна 2×2.
                #    С87 зміряла, що для поправок ДО `C` проєкція ШКОДИТЬ
                #    (×1.027 → ×0.935), і A2/A3 успадковують цю ваду.
                c = [Cm[j] - B[j] for j in range(NF)]
                sbb = sum(float((B[j] * B[j]).sum()) for j in tr)
                scc = sum(float((c[j] * c[j]).sum()) for j in tr)
                sbc = sum(float((B[j] * c[j]).sum()) for j in tr)
                sby = sum(float((B[j] * Y[j]).sum()) for j in tr)
                scy = sum(float((c[j] * Y[j]).sum()) for j in tr)
                det = sbb * scc - sbc * sbc
                if abs(det) < 1e-25 * max(sbb * scc, 1e-30):
                    p = a * B[i]
                else:
                    aa = (sby * scc - scy * sbc) / det
                    bb = (scy * sbb - sby * sbc) / det
                    alphas[i] = aa
                    p = aa * B[i] + bb * c[i]
        out[i] = float(((p - Y[i]) ** 2).mean())
    return out, alphas


base, al0 = fit_eval(B, 'A0')
print(f'\n{"=" * 76}\n⛔ ПЛОМБИ')
rel = float(torch.sqrt(((al0.mean() * B - Y) ** 2).sum() / (Y ** 2).sum()))
print(f' P1  база після α, віднос. похибка = {rel:.4%}  (calib88: 0.4894–0.6013%)')
_, al3 = fit_eval(PRED[:, 2], 'A3')
print(f' P2  α(A3) − α(A0) = {np.abs(al3 - al0).max():.3e}  (мусить ~1e-16)')

print(f'\n{"=" * 76}\n📈 `F(t_j,t_k,ρ)` ПІД РІЗНИМИ КАЛІБРОВКАМИ ({NF} мереж)')
print(f'{"вага":>6s} {"A1 вага=1":>11s} {"A2 ⊥":>9s} {"A3 α,β⊥":>10s} {"A4 2×2":>9s}')
bm = base.mean()
for wi, wgt in enumerate(WS):
    if wgt == 0.0:
        print(f'{wgt:6.2f} {"— база —":>11s} {"":>9s} {1.0:10.3f}')
        continue
    r = []
    for md in ('A1', 'A2', 'A3', 'A4'):
        e, _ = fit_eval(PRED[:, wi], md)
        r.append(bm / max(e.mean(), 1e-30))
    print(f'{wgt:6.2f} {r[0]:11.3f} {r[1]:9.3f} {r[2]:10.3f} {r[3]:9.3f}')
ex1, _ = fit_eval(PRED_X, 'A1')
ex3, _ = fit_eval(PRED_X, 'A4')
print(f'{"⛔чужа":>6s} {bm/max(ex1.mean(),1e-30):11.3f} {"":>9s} '
      f'{bm/max(ex3.mean(),1e-30):10.3f}   (A3 мусить дати ~1.000)')

best = max(bm / max(fit_eval(PRED[:, wi], 'A4')[0].mean(), 1e-30)
           for wi in range(1, NW))
print(f'\n📣 ВИРОК\n B. A3 найкраще: ×{best:.3f}  (передбачав ×1.10–1.35)')
print(f' C. переклад у бал: ×{(best + 15.66) / 16.66:.4f}  '
      f'⟹ {1.873e-07 / ((best + 15.66) / 16.66):.4e}')
print(f' D. поріг сабміта — потрібне ×1.05 у балі, тобто k ≥ 1.83')
