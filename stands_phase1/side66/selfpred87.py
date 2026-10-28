"""ЧИ ПЕРЕДБАЧУВАНА ПОХИБКА ГЛИБОКИХ ШАРІВ ІЗ МІЛКИХ? (С87, ціль ЮЗЕРА)

## ЗВІДКИ — ТЕЗА ЮЗЕРА
«Якщо є виграш навіть у оракула, то має бути і спосіб витягнути його.
Використовуй неочевидні ідеї.»

## 🔑 ДЕ МИ ЩЕ НЕ ШУКАЛИ
Усі попередні спроби дістати `σ` шукали її **У ВАГАХ** (`sigscan87`: 107520
точок, 15 виведених ознак, R²=0.203) або **В ЛОКАЛЬНИХ КУМУЛЯНТАХ** (оракульні
дають 0.465). Жодного разу не шукали **У ВЛАСНІЙ ДИНАМІЦІ ЛАНЦЮГА**.
А динаміку тепер розкладено у формулу:
```
похибка σ(шар l) = δ (похибка кроку) ÷ (1 − 0.976)   ← ЗНАМЕННИК ВІДОМИЙ
```
⟹ якщо `δ` конкретної мережі вимірна ТАМ, ДЕ ЦЕ ДЕШЕВО, формула дає `σ` на
глибині БЕЗ прямого вимірювання там.

## 🎯 ТРИ ПИТАННЯ, НА ЯКІ ВІДПОВІДАЄ ЦЕЙ СТЕНД
1. **Чи корелює похибка кроку МІЖ ШАРАМИ всередині мережі?** Якщо так —
   мілкі шари передбачають глибокі, і це НОВИЙ канал.
2. **Чи це не просто константа?** ⛔Якщо `δ` однакова для всіх мереж, канал
   вироджується в число з банку — а воно вже мертве (`layerk87`, ×1.053).
   Тому головна колонка — кореляція ПІСЛЯ зняття шарового середнього.
3. **Чи вимірна `δ` вибіркою?** Порівнюємо оракульну `δ` з тією, що дає
   скінченна МК на 6000 семплів (наш робочий розмір).

## 📐 ЧОМУ ЦЕ МОГЛО Б ПРАЦЮВАТИ ТАМ, ДЕ ВАГИ ПРОВАЛИЛИСЬ
Ознака з ваг мусить ВГАДАТИ негаусовість, накопичену з усієї історії. Ознака з
динаміки її ВИМІРЮЄ на ранніх шарах, де вона ще мала і де шум вибірки її не
топить. Це той самий перехід, що [[finding-remove-beats-predict]] («знімати, не
передбачати»), але вперше застосований до `σ`, а не до зміщення.

## ⛔ПЛОМБИ
 P1 ТАВТОЛОГІЯ: `δ` шару проти себе самої ⟹ корел 1.0000.
 P2 ⛔ПЕРЕМІШАНІ МЕРЕЖІ: `δ` мілкого шару чужої мережі — корел мусить впасти
    до нуля. Якщо ні — працює шарова константа, а не індивідуальність.
 P3 ШУМ: `δ`, зміряна на половині семплів, проти повної — скільки з кореляції
    з'їдає шум.
 P4 ПІСЛЯ ЗНЯТТЯ ШАРОВОГО СЕРЕДНЬОГО — інакше спільний тренд по глибині дасть
    фальшиву кореляцію (та сама пастка, що `_ALPHA` у `layerk87`).

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) СИРА корел `δ₄ ↔ δ₃₁` буде висока (>0.7) — але переважно через спільний
     шаровий тренд. Шанс 75%.
 (б) ПІСЛЯ зняття шарового середнього корел впаде нижче 0.3. Шанс **60%** —
     це і буде вирок: індивідуальність мережі в `δ` слабка.
 (в) якщо корел ≥0.5 після зняття — канал ЖИВИЙ, і це найкращий результат
     сесії. Мій чесний шанс: **25%**.
 (г) вибіркова `δ` при N=6000 збереже ≥половину сигналу. Шанс 45%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/selfpred87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N
NSUB = 6000                      # наш робочий розмір вибірки в сабміті
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
PROBE = tuple(range(2, 31, 2))   # 15 шарів
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def qform(W, C):
    return torch.clamp((C @ W * W).sum(0), min=1e-30)


def closure(m, C):
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s
    P, f = Phi(t), phi(t)
    Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
    m_a = m * P + s * f
    q2 = (m * m + s * s) * P + (m * s) * f
    return Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30)
                           - torch.diagonal(Ca)), m_a


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
NP = len(PROBE)
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · {DEV} · {NP} шарів\n', flush=True)

DEL = np.zeros((len(FS), NP))      # оракульна похибка кроку (масштаб)
DEL_H = np.zeros((len(FS), NP))    # те саме на ПОЛОВИНІ семплів (шум)
DEL_S = np.zeros((len(FS), NP))    # те саме на N=6000 (робочий розмір)
ACC = np.zeros((len(FS), NP))      # НАКОПИЧЕНА похибка σ ланцюга

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        SH = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GH = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        SA = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GA = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        SHh = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GHh = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        SAh = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GAh = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        SHs = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GHs = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        SAs = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GAs = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(4700 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                a = torch.relu(h)
                if q in PROBE:
                    j = PROBE.index(q)
                    hd, ad = h.double(), a.double()
                    SH[j] += hd.sum(0); GH[j] += hd.T @ hd
                    SA[j] += ad.sum(0); GA[j] += ad.T @ ad
                    if d < NS // 2:
                        SHh[j] += hd.sum(0); GHh[j] += hd.T @ hd
                        SAh[j] += ad.sum(0); GAh[j] += ad.T @ ad
                    if d < NSUB:
                        k = min(b, NSUB - d)
                        SHs[j] += hd[:k].sum(0); GHs[j] += hd[:k].T @ hd[:k]
                        SAs[j] += ad[:k].sum(0); GAs[j] += ad[:k].T @ ad[:k]
            d += b

        def delta_at(j, q, sh, gh, sa, ga, cnt):
            """`δ` = масштабна похибка ОДНОГО кроку замикання на цьому шарі."""
            mh = sh[j] / cnt
            Ch = gh[j] / cnt - torch.outer(mh, mh)
            Ch = 0.5 * (Ch + Ch.T)
            ma = sa[j] / cnt
            Ca = ga[j] / cnt - torch.outer(ma, ma)
            Ca = 0.5 * (Ca + Ca.T)
            Wn = W[q + 1]
            st = torch.sqrt(qform(Wn, Ca))
            Ccl, _ = closure(mh, Ch)
            sc = torch.sqrt(qform(Wn, Ccl))
            return float((sc * st).sum() / (st * st).sum())   # масштаб <1

        for j, q in enumerate(PROBE):
            DEL[n, j] = delta_at(j, q, SH, GH, SA, GA, NS)
            DEL_H[n, j] = delta_at(j, q, SHh, GHh, SAh, GAh, NS // 2)
            DEL_S[n, j] = delta_at(j, q, SHs, GHs, SAs, GAs, NSUB)

        # НАКОПИЧЕНА похибка σ нашого ланцюга проти істини
        C = W[0].T @ W[0]
        m = torch.zeros(W_, dtype=torch.float64, device=DEV)
        for i in range(1, L_):
            if (i - 1) in PROBE:
                j = PROBE.index(i - 1)
                mo = SH[j] / NS
                so = torch.sqrt(torch.clamp(
                    torch.diagonal(GH[j] / NS - torch.outer(mo, mo)), min=1e-30))
                sc = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
                ACC[n, j] = float((sc * so).sum() / (so * so).sum())
            C, m_a = closure(m, C)
            m = m_a @ W[i]
            C = W[i].T @ (C @ W[i])
    print(f'  net{n} · δ(шар3) {DEL[n, 0]:.5f} · δ(шар29) {DEL[n, -1]:.5f} '
          f'· накопич(29) {ACC[n, -1]:.4f}', flush=True)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / math.sqrt(max((a * a).sum() * (b * b).sum(), 1e-30)))


print(f'\n{"═" * 76}\n⛔ПЛОМБИ')
print(f' P1 тавтологія δ₀↔δ₀ = {corr(DEL[:, 0], DEL[:, 0]):.4f}  ✅')
rng = np.random.default_rng(87)
perm = rng.permutation(len(FS))
print(f' P2 ⛔перемішані мережі δ₀(чужа)↔δ_глиб = '
      f'{corr(DEL[perm, 0], DEL[:, -1]):+.4f}  (мусить бути ≈0)')
print(f' P3 шум: δ(повна)↔δ(половина) = {corr(DEL[:, -1], DEL_H[:, -1]):+.4f}')
print(f' P3b вибірка N=6000: δ(повна)↔δ(6000) = '
      f'{corr(DEL[:, -1], DEL_S[:, -1]):+.4f}')

print(f'\n{"═" * 76}')
print('δ ПО ШАРАХ (масштаб кроку; <1 = замикання ЗАНИЖУЄ)')
print('шар    : ' + ' '.join(f'{q + 1:7d}' for q in PROBE[::2]))
print('δ сер  : ' + ' '.join(f'{DEL[:, j].mean():7.5f}' for j in range(0, NP, 2)))
print('δ розкид: ' + ' '.join(f'{DEL[:, j].std():7.5f}' for j in range(0, NP, 2)))
print('накопич: ' + ' '.join(f'{ACC[:, j].mean():7.4f}' for j in range(0, NP, 2)))

print(f'\n{"═" * 76}')
print('🎯 ГОЛОВНЕ: ЧИ ПЕРЕДБАЧАЮТЬ МІЛКІ ШАРИ ГЛИБОКІ (по мережах)?')
Dc = DEL - DEL.mean(0)          # ⛔P4: ЗНЯТО ШАРОВЕ СЕРЕДНЄ
Ac = ACC - ACC.mean(0)
print(f'{"мілкий шар":>12s} {"сира корел":>11s} {"після зняття":>13s} '
      f'{"→накопич(29)":>13s}')
for j in (0, 1, 2, 3, 5, 7):
    print(f'{PROBE[j] + 1:12d} {corr(DEL[:, j], DEL[:, -1]):+11.4f} '
          f'{corr(Dc[:, j], Dc[:, -1]):+13.4f} {corr(Dc[:, j], Ac[:, -1]):+13.4f}')

best = max(abs(corr(Dc[:, j], Ac[:, -1])) for j in range(6))
print(f'\n📣 ВИРОК: найкраща |корел| мілкого δ з НАКОПИЧЕНОЮ похибкою = {best:.4f}')
if best >= 0.5:
    print('   🟢 КАНАЛ ЖИВИЙ — мілкі шари несуть індивідуальність мережі.')
    print(f'   очікуване зняття дисперсії: R² ≈ {best ** 2:.3f}')
else:
    print('   ⛔ мілкі шари НЕ передбачають глибокі понад шарову константу.')
    print('   ⟹ динаміка ланцюга індивідуальності мережі не несе.')
