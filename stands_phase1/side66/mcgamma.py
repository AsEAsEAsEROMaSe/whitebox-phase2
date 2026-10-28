"""`γ` З ВЛАСНИХ МК-СЕМПЛІВ: ЧИ Б'Є ФОРМУЛА ПРЯМЕ СЕРЕДНЄ. (С86)

## ЗВІДКИ
`perneuron.py` показав: понейронних 4 моментів вистачає (×12.88), але здобути
`γ₃`,`γ₄` детерміновано нема як (рекурсія 2.9%, скаляри ×1.0, оракульна
регресія на `t` ×1.58). Лишився ОДИН непробуваний шлях — узяти `γ` з наших
ВЛАСНИХ ~5к МК-семплів, за які ми вже платимо.

## 🎯 ЧЕСНЕ ПОРІВНЯННЯ ПРИ ОДНАКОВІЙ ЦІНІ
З ТИХ САМИХ `N` семплів будуються дві оцінки `E[a_ℓ]`:
   DIRECT : пряме середнє `mean(relu(h))`           ← те, що робить сабміт зараз
   FORMULA: `G_0+(γ₃/6)G_3+(γ₄/24)G_4` з моментів   ← Rao-Blackwell останнього кроку
Обидві коштують однаково (той самий прогін). Питання лише: хто ТОЧНІШИЙ.
⟹ якщо FORMULA < DIRECT — напрямок живий; якщо ні — закритий.

## ⚠️ СУМНІВ, ЗАПИСАНИЙ ДО ВИМІРУ
Виграш Rao-Blackwell ≈ `Var(ReLU) / [Φ(t)²σ² + φ(t)²σ²/2]`. При `t=0` це ~3%,
при `t=3` — **0.13%**, бо ReLU там майже лінійний і інтегрувати нема чого.
`|t|` росте 0.55→2.95 ⟹ ОЧІКУЮ майже нуль на глибині і дещо на шарах 2–6.
📣 Шанс, що на ФІНАЛЬНОМУ шарі виграш ≥5%: **15%**.
📣 Шанс, що на шарах 2–6 виграш ≥20%: **45%**.

## ⛔ПЛОМБИ
 (1) 🔑ІСТИНА — З НЕЗАЛЕЖНОЇ ВИБІРКИ (інший seed). Якби істина й оцінка йшли з
     тих самих семплів, шум скоротився б і FORMULA виглядала б краще, ніж є.
 (2) шар 1: `h₁` точно гаусів ⟹ `γ`-члени мусять НЕ допомагати (там і так точно),
     а FORMULA мусить бити DIRECT рівно на Rao-Blackwell-виграші.
 (3) N→великі: обидві оцінки мусять сходитись до істини як `1/√N`.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/mcgamma.py [cuda:N] [NETS]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА.
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TRUTH_LOG2 = 22
CUTS = [2048, 4096, 8192, 16384, 32768, 65536]
BATCH = 1 << 11
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[0].split('## ⚠️')[1][:400], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж · істина 2^{TRUTH_LOG2} · зрізи {CUTS} · {DEV}\n', flush=True)

SQ2PI = math.sqrt(2.0 * math.pi)


def moments(W, dev, ns, batch, seed, cuts=None):
    """Сирі суми 1..4 для `h` і сума `relu(h)`, зі зрізами на межах `cuts`.

    Повертає {n: (Sh[шар][4], Sa[шар])} для кожного n у cuts (+ фінальне ns).
    """
    Sh = [torch.zeros(4, W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    Sa = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    marks = sorted(set((cuts or []) + [ns]))
    out = {}
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    d = 0
    while d < ns:
        b = min(batch, ns - d)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            h = a @ W[i]
            hd = h.double()
            p = hd
            for k in range(4):
                Sh[i][k] += p.sum(0)
                p = p * hd
            a = torch.relu(h)
            Sa[i] += a.double().sum(0)
        d += b
        if d in marks:
            out[d] = ([s.clone() for s in Sh], [s.clone() for s in Sa])
    return out


def cumulants(S, n):
    m1, m2, m3, m4 = S[0] / n, S[1] / n, S[2] / n, S[3] / n
    c2 = m2 - m1 * m1
    c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
    return m1, c2, c3, mu4 - 3.0 * c2 * c2


def formula(m, c2, c3, c4, use_g):
    s = torch.sqrt(torch.clamp(c2, min=1e-30))
    t = m / s
    ph = torch.exp(-0.5 * t * t) / SQ2PI
    Ph = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
    out = t * Ph + ph
    if use_g:
        g3, g4 = c3 / s ** 3, c4 / s ** 4
        out = out + (g3 / 6.0) * (-t * ph) + (g4 / 24.0) * ((t * t - 1.0) * ph)
    return s * out


E_DIR = np.zeros((len(FS), len(CUTS), L_))
E_GAU = np.zeros((len(FS), len(CUTS), L_))
E_FRM = np.zeros((len(FS), len(CUTS), L_))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        # ⛔ПЛОМБА(1): істина — НЕЗАЛЕЖНИЙ seed (7700+n), оцінка — (5500+n)
        tr = moments(W, DEV, 1 << TRUTH_LOG2, 1 << 13, 7700 + n)[1 << TRUTH_LOG2]
        truth = [tr[1][i] / (1 << TRUTH_LOG2) for i in range(L_)]
        est = moments(W, DEV, max(CUTS), BATCH, 5500 + n, CUTS)
        for ci, N in enumerate(CUTS):
            Sh, Sa = est[N]
            for i in range(L_):
                tv = truth[i]
                base = float(torch.sqrt((tv * tv).mean()))
                m, c2, c3, c4 = cumulants(Sh[i], N)
                d = Sa[i] / N
                for arr, val in ((E_DIR, d),
                                 (E_GAU, formula(m, c2, c3, c4, False)),
                                 (E_FRM, formula(m, c2, c3, c4, True))):
                    arr[n, ci, i] = float(torch.sqrt(((val - tv) ** 2).mean())) / base
    print(f'  … {n+1}/{len(FS)} seed {sd0} · N=4096 шар32: '
          f'DIRECT {E_DIR[n,1,31]*100:.4f}%  FORMULA {E_FRM[n,1,31]*100:.4f}%',
          flush=True)
    del W
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/mcgamma.npz',
         dir=E_DIR, gau=E_GAU, frm=E_FRM, cuts=np.array(CUTS))

print('\n' + '=' * 84)
print('📊 ПОХИБКА ОЦІНКИ `E[a_ℓ]` З `N` СЕМПЛІВ, % (істина — незалежні 2²²)')
for i in (1, 3, 7, 15, 31):
    print(f'\n── ШАР {i+1} ' + '─' * 60)
    print(f'{"N":>7s} {"DIRECT":>10s} {"ГАУС":>10s} {"ФОРМУЛА":>10s} '
          f'{"ФОРМ/DIR":>10s} {"виграш":>9s}')
    for ci, N in enumerate(CUTS):
        dd, gg, ff = (E_DIR[:, ci, i].mean(), E_GAU[:, ci, i].mean(),
                      E_FRM[:, ci, i].mean())
        print(f'{N:7d} {dd*100:10.4f} {gg*100:10.4f} {ff*100:10.4f} '
              f'{ff/dd:10.4f} {"×%.3f" % (dd/ff):>9s}')
print('=' * 84)
print(f'\n🎯 ПРИ N=4096 (наш робочий режим), середнє по шарах 5..32:')
d5 = E_DIR[:, 1, 4:].mean(); f5 = E_FRM[:, 1, 4:].mean()
print(f'   DIRECT {d5*100:.4f}%  ФОРМУЛА {f5*100:.4f}%  ⟹ ×{d5/f5:.4f}')
print(f'   ФІНАЛЬНИЙ ШАР 32: DIRECT {E_DIR[:,1,31].mean()*100:.4f}%  '
      f'ФОРМУЛА {E_FRM[:,1,31].mean()*100:.4f}%  ⟹ ×{E_DIR[:,1,31].mean()/E_FRM[:,1,31].mean():.4f}')
print('\n⛔ПЛОМБА(3) збіжність 1/√N (DIRECT, шар 32): '
      + ' → '.join('%.4f' % (E_DIR[:, ci, 31].mean() * 100) for ci in range(len(CUTS))))
