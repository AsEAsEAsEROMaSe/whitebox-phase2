"""ВИРІВНЮВАННЯ СУСІДНІХ ШАРІВ ЯК ОЗНАКА МЕРЕЖІ. (С85)

## ЗВІДКИ — АНАЛОГІЯ ЮЗЕРА ПРО ПЛОЩИНИ Й ПОЛЯРИЗАТОРИ
«Модель повнозв'язна схожа на послідовність площин із відкритими щілинами
(нейрони) і поляризаторами (relu).»
✅ ReLU = поляризатор ТОЧНО: обидва — необоротні проєктори, обидва пропускають
рівно ½ енергії (`E[a²] = σ²/2` — прямий аналог закону Малюса, усередненого
по кутах), і цей коефіцієнт ½ УЖЕ вбудований у Мелерову рекурсію.
⛔ Щілини — ні: немає локальності, відстані, фази ⟹ деструктивної
інтерференції не існує (після ReLU всі `a ≥ 0`, скасування можливе лише через
знаки ваг у `h`, а це вже в `C`).

🔑 ЩО В АНАЛОГІЇ ЛИШИЛОСЬ НЕВИКОРИСТАНИМ — СЛОВО «ПОСЛІДОВНІСТЬ».
Закон Малюса дає `cos²θ`, і `½` виходить ЛИШЕ ПІСЛЯ усереднення по θ. Отже
`½` — уже усереднена величина, а конкретний КУТ між сусідніми шарами не
усереднений: у кожної мережі він свій. В оптиці це принцип поляризаційного
ротатора — ланцюг поляризаторів із МАЛИМИ кутами між сусідніми проводить
світло майже без втрат, з великими — гасить. Втрати визначає не окремий
елемент, а **вирівнювання сусідніх**.
У нас це `W_i · W_{i+1}` — і це ГЛОБАЛЬНА величина, а не локальна. Закриття
С84 («8 ознак із ваг дали `R²` +0.0028») стосувалось ознак ОКРЕМОГО НЕЙРОНА;
міжшарове вирівнювання туди не входило, і воно рівно того класу, який
[[finding-function-is-worse-than-uniformly-high-dimensional]] лишає живим.

## ЧОМУ САМЕ МАСШТАБ, А НЕ `b_q`
У `b_q` патернів немає (`R²`=0.26). А в МАСШТАБІ мережева складова Є:
`ĉ` корелює −0.60 між мережами ([[finding-holographic-reference-beam-dies-on-draw-noise]]).
sub95 помер не через відсутність сигналу, а тому, що `ĉ` **пожеребна**
(розкид по жеребах ×1.71 більший за мережевий). Вирівнювання шарів
пожеребного шуму НЕ МАЄ ЗОВСІМ: воно рахується з `W` і в кожному жеребі те саме.

## ЩО МІРЯЄМО (48 мереж × NDRAW жеребів, банк, T1/T2)
ЦІЛЬ  `c*_net = ⟨p, T1+T2⟩ / (2‖p‖²)` — оракульний масштаб, усереднений по
      жеребах (жеребна складова гаситься як 1/√NDRAW).
ОЗНАКИ з ваг (обрізаних `wc` — саме вони йдуть у прохід):
      `a_i = ‖W_i W_{i+1}‖_F² / (‖W_i‖_F² ‖W_{i+1}‖_F² / n_i)` — нормоване
      перекриття сусідів; для незалежних матриць ⟨a_i⟩ ≈ 1.
      F1 сер. по всіх · F2 сер. по 1..8 (де стоять λ) · F3 сер. по глибоких
      · F4 **Σ log a_i** — логарифм повного «пропускання» ланцюга, тобто
      рівно те, що аналогія й називає добутком по послідовності.
⛔ КОНТРОЛІ (три ХВОРОБИ, [[feedback-controls-have-three-distinct-diseases]]):
   N1 «лише норми» — чи ознака не є переодягненим масштабом ваг;
   N2 «лише ширини» — чи не є вона переодягненим обрізанням;
   SHUF «перемішані шари» — руйнує СУСІДСТВО, зберігає норми й ширини;
   ЧУЖА мережа — руйнує відповідність ознака↔ціль.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: ~20%, тобто скоріше мертво. Арифметика проти:
   для незалежних He-матриць 256×256 слід `‖W_iW_{i+1}‖_F²` має відносну
   флуктуацію ≈ `1/√(n²) = 0.4%`. Якщо розкид `c*` по мережах — одиниці
   відсотків, ознака з розкидом 0.4% може його пояснити лише з коефіцієнтом
   ~10, а це підсилення власного шуму ознаки в ті ж 10 разів. Той самий закон
   великих чисел, що вже вбив ознаки з ваг. ⟹ гейт: |корел| < 0.3 ⟹ мертво.
🎁 ЯКЩО ЖИВЕ — воно БЕЗКОШТОВНЕ: добутки сусідніх шарів уже рахуються
   всередині замикання (`Cov = WᵀCW`) на шарах 1..8.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/layalign.py [cuda:N] [NETS] [NDRAW]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 24
L_ = 32
# 🚨 КОНСТАНТИ ЯВНО — урок [[feedback-new-stand-inherits-constants-not-just-truth]]
W_BLEND, ALPHA = 0.0883, -0.00751
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
print(f'⛔КОНСТАНТИ СТЕНДУ: w={W_BLEND}, α={ALPHA} (sub92/sub93, НЕ порт!)\n',
      flush=True)

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


def aux_parts(wc, dev):
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    AA, BB = [], []
    for i in range(len(wc)):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        if 1 <= i <= KSIG:
            s2ph = (s * s) * ph
            AA.append(s2ph); BB.append(s2ph * t)
        m = mp * Ph + s * ph
        if i < len(wc) - 1:
            d2 = (mp * mp + s * s) * Ph + mp * s * ph
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            C = C + torch.diag((d2 - m * m) - C.diagonal())
    return AA, BB


def align_feats(wc, gen):
    """🎯 ОЗНАКИ ВИРІВНЮВАННЯ + ⛔контрольні ознаки. Усе — ЛИШЕ з ваг."""
    nf = [float(w.double().pow(2).sum()) for w in wc]          # ‖W_i‖_F²
    wid = [int(w.shape[1]) for w in wc]                        # ширини
    a, ash = [], []
    perm = torch.randperm(len(wc) - 1, generator=gen).tolist()
    for i in range(len(wc) - 1):
        P = wc[i].double() @ wc[i + 1].double()
        num = float(P.pow(2).sum())
        a.append(num / (nf[i] * nf[i + 1] / wc[i].shape[1]))
        # ⛔SHUF: той самий i-й лівий шар, але ПРАВИЙ — з іншого місця
        j = perm[i]
        if wc[i].shape[1] == wc[j + 1].shape[0]:
            Q = wc[i].double() @ wc[j + 1].double()
            ash.append(float(Q.pow(2).sum())
                       / (nf[i] * nf[j + 1] / wc[i].shape[1]))
        else:
            ash.append(a[-1])          # форми не збіглись — беремо як є
    a = np.array(a); ash = np.array(ash)
    return {
        '🎯F1 сер. усі сусіди': a.mean(),
        '🎯F2 сер. шари 1..8': a[:KSIG].mean(),
        '🎯F3 сер. глибокі 24+': a[24:].mean(),
        '🎯F4 Σ log a (ланцюг)': float(np.log(a).sum()),
        '⛔N1 лише норми': float(np.mean(nf)),
        '⛔N2 лише ширини': float(np.mean(wid)),
        '⛔SHUF перемішані': ash.mean(),
    }


def run93(wc, npair, dev, seed, cl, aux):
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
    AA, BB = aux
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        pd = p.double(); p2 = pd * pd
        mh = pd.sum(0) / den
        e2 = p2.sum(0) / den
        sh2 = (e2 - mh * mh).clamp_min(1e-30)
        sh = sh2.sqrt()
        x = torch.relu(p)
        if i <= KSIG:
            e3 = (p2 * pd).sum(0) / den
            e4 = (p2 * p2).sum(0) / den
            mh2 = mh * mh
            k3 = e3 - 3.0 * mh * e2 + 2.0 * mh2 * mh
            k4 = (e4 - 4.0*mh*e3 + 6.0*mh2*e2 - 3.0*mh2*mh2 - 3.0*sh2*sh2)
            dq = ((k3 / (sh2 * sh)) * (AA[i-1] / 3.0)
                  - (k4 / (sh2 * sh2)) * (BB[i-1] / 12.0))
            xm = x.double().sum(0) / den
            xq = x.double().pow(2).sum(0) / den
            v = (v + (SP.LAM_S[i-1] * _phi(mh/sh)) * (sh - s_cl[i-1])
                 + SP.LAM_A[i-1] * (xm - m_cl[i-1])
                 + SP.LAM_Q[i-1] * ((xq - q_cl[i-1]) - dq)
                 + SP.LAM_G[i-1] * (gate - P_cl[i-1]))
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    return ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)


N = len(FS)
PP = np.zeros(N); PT = np.zeros(N); TT = np.zeros(N)
CSD = np.zeros(N)                       # розкид c* ПО ЖЕРЕБАХ усередині мережі
FEAT = None

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
        aux = aux_parts(wc, DEV)
        gen = torch.Generator(); gen.manual_seed(555 + n)
        fd = align_feats(wc, gen)
        if FEAT is None:
            FEAT = {k: np.zeros(N) for k in fd}
        for k, v in fd.items():
            FEAT[k][n] = v
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        cds = []
        for d in range(NDRAW):
            pr = run93(wc, npair, DEV, 107_000_000 + 733*n + d, cl, aux)
            pp = float((pr * pr).sum()); pt = float((pr * (T1 + T2)).sum())
            PP[n] += pp / NDRAW
            PT[n] += pt / NDRAW
            TT[n] += float((T1 * T2).sum()) / NDRAW
            cds.append(pt / (2.0 * pp))
        CSD[n] = float(np.std(cds))
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{N}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/layalign.npz',
         PP=PP, PT=PT, TT=TT, CSD=CSD, **{f'F{i}': v
                                          for i, v in enumerate(FEAT.values())})
cstar = PT / (2.0 * PP)


def mse(c):
    return float((c**2 * PP - c * PT + TT).mean())


base = mse(np.ones(N))
top = mse(cstar)
print(f'\n⛔ПЛОМБА (а) c=1 ⟹ база sub93 = {base:.6e}  '
      f'(інші стенди С84 давали ≈6.5e-04)')
print(f'⛔ПЛОМБА (б) c=c* ОРАКУЛ ⟹ СТЕЛЯ = {top:.6e} ⟹ ×{base/top:.4f}')
print(f'🔑 c* по МЕРЕЖАХ: сер {cstar.mean():.6f} розкид {cstar.std():.6f} '
      f'({cstar.std()/cstar.mean()*100:.3f}%)')
print(f'   залишковий ЖЕРЕБНИЙ шум у c*_net: '
      f'{CSD.mean()/np.sqrt(NDRAW):.6f} '
      f'({CSD.mean()/np.sqrt(NDRAW)/cstar.std()*100:.1f}% від мережевого)\n')


def loo_pred(x):
    o = np.zeros(N)
    for k in range(N):
        idx = np.array([j for j in range(N) if j != k])
        o[k] = np.polyval(np.polyfit(x[idx], cstar[idx], 1), x[k])
    return o


print(f'{"ознака":>26s} {"розкид,%":>9s} {"корел":>8s} {"R²":>7s} '
      f'{"LOO ×":>8s} {"⛔ЧУЖА ×":>9s}')
print('─' * 74)
for k, v in FEAT.items():
    rel = v.std() / abs(v.mean()) * 100 if v.mean() != 0 else float('nan')
    r = np.corrcoef(v, cstar)[0, 1]
    p = loo_pred(v)
    print(f'{k:>26s} {rel:9.3f} {r:+8.4f} {r*r:7.4f} '
          f'{base/mse(p):8.4f} {base/mse(np.roll(p, 1)):9.4f}')
print('─' * 74)
print(f'{"⛔ одна константа (LOO)":>26s} {"":9s} {"":8s} {"":7s} '
      f'{base/mse(np.array([cstar[[j for j in range(N) if j != k]].mean() for k in range(N)])):8.4f}')
print(f'{"🟢 c* ОРАКУЛ (стеля)":>26s} {"":9s} {"":8s} {"":7s} '
      f'{base/top:8.4f}')
print('\n🎯 ГЕЙТ: |корел| < 0.3 у всіх 🎯-ознак ⟹ гілка МЕРТВА, аналогію '
      'закрито числом.\n   ⛔SHUF мусить бути СЛАБШИМ за 🎯F1, інакше ознака '
      'ловить норми, а не сусідство.')
