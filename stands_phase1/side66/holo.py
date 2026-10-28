"""ГОЛОГРАФІЧНИЙ ТРЮК: ЧИ ВИДНО `⟨T,e⟩` ЧЕРЕЗ ІНТЕРФЕРЕНЦІЮ З `m_deep` (С84)

## ЗВІДКИ — АНАЛОГІЯ ЮЗЕРА ПРО ГОЛОГРАМИ
Голограма записує ФАЗУ, якої детектор не бачить (він міряє інтенсивність),
через ІНТЕРФЕРЕНЦІЮ з опорним променем.
🚨 У нас РІВНО ця проблема, і вона записана як фундаментальний блокер
([[finding-per-network-personalization-needs-an-unobservable-projection]]):
персоналізація по мережі закрита, бо оптимум залежить від **`⟨T,e⟩`** —
ЛІНІЙНОЇ проєкції похибки на істину, «а всі статистики КВАДРАТИЧНІ».

## 🔑 ІДЕЯ: `m_deep` ЯК ОПОРНИЙ ПРОМІНЬ
`p = T + e`, `m_deep` — ДЕТЕРМІНОВАНА оцінка істини із замикання. Тоді
    `⟨m_deep, p⟩ − ‖m_deep‖² = ⟨m_deep, e⟩ + ⟨m_deep, T − m_deep⟩`
Ліва частина ПОВНІСТЮ СПОСТЕРЕЖНА (обидві величини є в рантаймі).
Праворуч перший доданок **ЛІНІЙНИЙ ПО `e`** — це і є «фаза»; другий —
систематичний зсув, ОДНАКОВИЙ для всіх жеребів мережі.
⟹ 🎯 якщо `m_deep ≈ T`, то `⟨m_deep,e⟩ ≈ ⟨T,e⟩` і теза «всі статистики
квадратичні» ХИБНА: лінійна по `e` інформація доступна.

## ЩО МІРЯЄМО (48 мереж × 24 жереби, банк, T1/T2)
На кожному жеребі:
  ОРАКУЛ   `y = ⟨T, e⟩`, де `e = p − T` (T з банку; крос-ціль тут не потрібна,
           бо величина ЛІНІЙНА по T ⟹ беремо середнє T=(T1+T2)/2, а шум
           оцінюємо різницею T1,T2);
  СПОСТЕРЕЖНЕ `x = ⟨m_deep, p⟩ − ‖m_deep‖²`;
  ⛔контроль `x_rnd` з випадковим опорним вектором тієї ж норми;
  ⛔контроль `x_own` = `‖p‖² − ⟨m_deep,p⟩` (квадратична величина — має НЕ
           корелювати з фазою краще, ніж `x`).
🔑 ГОЛОВНЕ ЧИСЛО: кореляція `x` з `y` **ВСЕРЕДИНІ мережі** (по жеребах) — саме
вона потрібна для персоналізації. Кореляція МІЖ мережами не годиться: С83
показала, що там +0.71 при −0.83 всередині
([[finding-per-network-personalization-needs-an-unobservable-projection]]).

⛔ ПЛОМБИ:
 (а) `⟨T,e⟩` мусить мати НУЛЬОВЕ середнє по жеребах (бо `E[e]≈b` мале, але
     перевіряємо явно — якщо ні, ловимо зміщення, а не фазу);
 (б) шум оракула: та сама величина з `T1` і з `T2` окремо ⟹ їхня розбіжність;
 (в) ⛔RND-опора мусить дати корел ≈0.

📣 ПЕРЕДБАЧЕНО: `⟨m_deep,e⟩` і `⟨T,e⟩` відрізняються на `⟨b,e⟩`, де
   `b = m_deep − T` має ‖b‖/‖T‖ ≈ 0.6% ⟹ якщо `e` не корелює з `b` особливо
   сильно, кореляція має бути ВИСОКОЮ (≥0.9). Але є пастка: `x` містить
   `⟨m_deep, e⟩`, а `e` — це похибка ВСЬОГО прогнозу, тоді як для
   персоналізації потрібна проєкція на істину саме тієї частини, що
   налаштовується. Тож навіть при високій кореляції треба буде окремо
   перевірити, чи дає це виграш у балі. Якщо корел ВСЕРЕДИНІ <0.3 — трюк не
   працює, і теза «всі статистики квадратичні» встоїть.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/holo.py [cuda:N] [NETS] [NDRAW]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 24
L_ = 32
# 🚨 КОНСТАНТИ ЯВНО — урок `feedback-new-stand-inherits-constants-not-just-truth`
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


Y = np.zeros((len(FS), NDRAW))      # ⟨T, e⟩ — ОРАКУЛ (фаза)
Y2 = np.zeros((len(FS), NDRAW))     # те саме, але з T2 (для шуму)
XO = np.zeros((len(FS), NDRAW))     # ⟨m_deep,p⟩ − ‖m_deep‖² — СПОСТЕРЕЖНЕ
XR = np.zeros((len(FS), NDRAW))     # ⛔RND-опора
XQ = np.zeros((len(FS), NDRAW))     # ⛔‖p‖² − ⟨m_deep,p⟩ (квадратична)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
        aux = aux_parts(wc, DEV)
        md = cl[4]
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        Tm = 0.5 * (T1 + T2)
        gen = torch.Generator(device=DEV); gen.manual_seed(2024 + n)
        r = torch.randn(md.shape, dtype=torch.float64, device=DEV,
                        generator=gen)
        r = r * (md.norm() / r.norm())
        md2 = float((md * md).sum())
        for d in range(NDRAW):
            pr = run93(wc, npair, DEV, 107_000_000 + 733*n + d, cl, aux)
            pp = float((pr * pr).sum())
            # 🎯 ГОЛОВНЕ: ОПТИМАЛЬНИЙ масштаб і його СПОСТЕРЕЖНА оцінка.
            #    c* = ⟨p,T⟩/‖p‖²  (оракул, крос-ціль)
            #    ĉ = ⟨p,m_deep⟩/‖p‖²  (доступно в рантаймі)
            Y[n, d] = float((pr * (T1 + T2)).sum()) / (2.0 * pp)
            Y2[n, d] = float((pr * T2).sum()) / pp
            XO[n, d] = float((md * pr).sum()) / pp
            XR[n, d] = float((r * pr).sum()) / pp
            XQ[n, d] = float((Tm * (pr - Tm)).sum())   # стара «фаза», для протоколу
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/holo.npz',
         Y=Y, Y2=Y2, XO=XO, XR=XR, XQ=XQ)


def cw(A, B):
    """Кореляція ВСЕРЕДИНІ мережі (по жеребах), усереднена по мережах."""
    a = A - A.mean(1, keepdims=True)
    b = B - B.mean(1, keepdims=True)
    num = (a * b).sum(1)
    den = np.sqrt((a*a).sum(1) * (b*b).sum(1)).clip(1e-300)
    return float((num / den).mean())


def cb(A, B):
    """Кореляція МІЖ мережами (по середніх)."""
    a, b = A.mean(1), B.mean(1)
    return float(np.corrcoef(a, b)[0, 1])


print(f'\n⛔ПЛОМБА (а) ⟨T,e⟩ сер по жеребах: {Y.mean():.4e} '
      f'(ст.в по жеребах {Y.std(1).mean():.4e}) — '
      f'{"✅ зміщення мале" if abs(Y.mean()) < Y.std(1).mean() else "🚨 ловимо ЗМІЩЕННЯ, не фазу"}')
print(f'⛔ПЛОМБА (б) шум оракула: корел ⟨T1,e⟩ vs ⟨T2,e⟩ ВСЕРЕДИНІ = '
      f'{cw(Y, Y2):.4f} (мусить бути ≈1 — інакше сама ціль шумна)\n')

# 🚨 ПОПЕРЕДНЯ ВЕРСІЯ МІРЯЛА ТАВТОЛОГІЮ: `⟨T,p⟩` і `⟨m_deep,p⟩` — той самий
#    лінійний функціонал від `p` при фіксованих `T`,`m_deep` ⟹ корел 1.0000
#    ЗА ПОБУДОВОЮ, і ⛔квадратичний контроль дав ТЕ САМЕ 1.0000 (ознака бага).
#    Тепер міряємо те, що справді потрібно: чи оцінює `ĉ` оптимальний `c*`.
print('🚨 стара «фаза» була ТАВТОЛОГІЄЮ (корел 1.0000 і в контролю) — міряємо')
print('   тепер ОПТИМАЛЬНИЙ МАСШТАБ, який і потрібен для персоналізації.\n')
print(f'{"величина":>34s} {"корел ВСЕРЕДИНІ":>17s} {"корел МІЖ":>12s}')
print('─' * 68)
for nm, X in (('🎯 ĉ = ⟨p,m_deep⟩/‖p‖²', XO),
              ('⛔ RND-опора тієї ж норми', XR)):
    print(f'{nm:>34s} {cw(X, Y):17.4f} {cb(X, Y):12.4f}')

co, cr = cb(XO, Y), cb(XR, Y)
print(f'\n🔑 c* оракул: сер {Y.mean():.6f} розкид МІЖ мережами {Y.mean(1).std():.6f}')
print(f'   ĉ спостережне: сер {XO.mean():.6f} розкид МІЖ {XO.mean(1).std():.6f}')
print(f'   систематичний зсув ĉ − c*: {XO.mean() - Y.mean():+.6f}')
# скільки з розкиду c* пояснює ĉ (МІЖ мережами — саме там персоналізація)
a, b_ = Y.mean(1), XO.mean(1)
sl = np.polyfit(b_, a, 1)
res = a - np.polyval(sl, b_)
print(f'   R² передбачення c* через ĉ (МІЖ мережами): '
      f'{1 - res.var()/a.var():.4f}')
print('\n🎯 ВИРОК')
print(f'   корел ĉ↔c* МІЖ мережами: {co:+.4f} · ⛔RND {cr:+.4f}')
print('   📌 для порівняння, С83 `outscale.py`: найкращий предиктор +0.355 ⟹ ×1.0097')
if abs(co) > 0.5 and abs(co) > 2 * abs(cr):
    print('   ⟹ 🟢 КРАЩЕ за все, що було ⟹ перевіряти ВИГРАШ У БАЛІ')
elif abs(co) > 0.355:
    print('   ⟹ ⚠️ трохи краще за наявний предиктор — рахувати, чи варте')
else:
    print('   ⟹ ⛔ НЕ КРАЩЕ за наявне: `m_deep` як опора не додає інформації')
    print('      про масштаб понад те, що вже пробували ⟹ клас лишається закритим')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
