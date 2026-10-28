"""ВІДПАЛ: ЗНЯТИ ЗАМОРОЖЕНІ НАПРУЖЕННЯ КОНСТАНТ `sub93` (С84, теза ЮЗЕРА)

## ЗВІДКИ — АНАЛОГІЯ ЮЗЕРА ПРО ВІДПАЛ МЕТАЛУ
⛔ Відпал як МЕТОД ПОШУКУ (simulated annealing) нам не поможе: обмеження не в
пошуку, а в даних (закон `11.76%·k/N`, при N=48 доступно ~3.6 параметра —
[[finding-capacity-law-and-why-q-has-no-pattern]]).
🎯 АЛЕ відпал — це насамперед ЗНЯТТЯ ВНУТРІШНІХ НАПРУЖЕНЬ, і ось це б'є точно.
Естиматор зібраний із констант, підігнаних У РІЗНИЙ ЧАС ПРИ ЗАМОРОЖЕНИХ ІНШИХ:
`_LAM` (32 числа) — С66 під тодішню базу · `_ALPHA=−0.00751` — окремо, як
компенсація зсуву замикання · `_W_BLEND=0.0883` — С83 під БАНКОВУ істину ·
`_KSIG`,`_GATE_N`,`_TARGET_UTIL` — інші сесії.
🚨 І `sub93` додав НОВУ напругу: він змінив САМУ ОЗНАКУ `δ_Q` (чистка), але
`_ALPHA` і `_W_BLEND` лишились від sub92, тобто налаштовані під БРУДНУ ознаку.

## 🔑 ЧОМУ ЦЕ ДЕШЕВО І ЧЕСНО
`pr = ours + w·((1+α)·m_deep − ours) = (1−w)·ours + w(1+α)·m_deep`
⟹ ДВА лінійні коефіцієнти `A = 1−w` при `ours` і `B = w(1+α)` при `m_deep`.
Оптимум аналітичний (2×2 система), ЛОО по мережах. Два параметри при
дозволених 3.6 — рівно в межах закону ємності.
📌 Переклад назад: `w = 1−A`, `α = B/(1−A) − 1`.

⛔ ПЛОМБИ: (а) `(A,B)` нинішнього sub93 мусять відтворити його MSE РІВНО;
 (б) LOO по мережах; (в) ⛔контроль: та сама процедура на sub92 — якщо виграш
 такий самий, то він НЕ від чистки `q`, а просто від того, що константи були
 не в оптимумі взагалі; (г) банк, крос-ціль.

📣 ПЕРЕДБАЧЕНО: `sub92` уже перепідбирався по `w` у С83 на банку, тож для нього
   приріст має бути малим (≈×1.00). Для `sub93` — більшим, бо ознака змінилась
   після того підбору. Чекаю ×1.005–1.02 для sub93 і різницю між ними як
   ЧИСТУ ціну успадкованого налаштування. Якщо ж обидва дадуть однаково —
   значить справа не в чистці `q`, а в тому, що `α` ніколи не підбирався
   СПІЛЬНО з `w` (він фітувався окремо ще до бленду).

## СТАРА ШАПКА
ЗМІЩЕННЯ ОБРІЗАННЯ — АНАЛІТИЧНО, ДВОМА ЗАМИКАННЯМИ (С84)

## ЗВІДКИ (метафора ЮЗЕРА: «мікробугорок на початку зміщує всю траєкторію»)
У прямому прочитанні метафора СПРОСТОВАНА: коефіцієнт передачі збурення через
шар `G = 0.22` (`side66/selfcon.py`) — крок ГАСИТЬ бугорок у 4.5×, а якір із
шару 0 приходить до виходу у 8.1× меншим (С75).
🔑 АЛЕ вона працює буквально для РІШЕННЯ, а не для величини: `cut_budget91`
дивиться 128 пілотних пар і НАЗАВЖДИ викидає нейрони з `max h < 0`. Це не
доданок до траєкторії — це ЗМІНА САМОЇ МЕРЕЖІ, крізь яку йде решта вибірки,
і воно не гасне.

## 📊 РОЗМІР УЖЕ ЗМІРЯНИЙ, ПОПРАВКА — НІ
С82 ([[reference-bank-truth-is-the-right-target]]): зсув «обрізана проти повної»
= **1.5e-04…6.1e-04** при RMS робочого 1.4e-03 ⟹ **до 20% MSE**, і він
БІЛЬШИЙ за обидва шуми, тобто систематичний.
С81: «нейрон із `p_on≈1%` на 256 пілотних семплах виглядає мертвим, пропуск
7.6%» — і там же дослівно: **«як внесок У БАЛ НЕ МІРЯНО»**.
📌 Обрізання САМЕ ПО СОБІ — чистий плюс (без нього −30.25% MSE), бо дає більше
пар за той самий бюджет. Питання не «чи обрізати», а «чи компенсувати зсув».

## 🔑 ІДЕЯ: ЗСУВ ОБЧИСЛЮВАНИЙ БЕЗ ВИБІРКИ
Замикання вже рахується на ОБРІЗАНИХ вагах (`m_deep`). Порахувати його ще раз
на ПОВНИХ — і різниця
    `D = m_deep(ПОВНІ W) − m_deep(ОБРІЗАНІ wc)`
є АНАЛІТИЧНОЮ оцінкою зсуву обрізання: детермінована, без краплі шуму вибірки.
💰 ЦІНА: 32 матмули 256×256 = 6.7e7 флоп ≈ **0.8% бюджету** (вибірка — 7400×32
таких), і йде ПОВЗ `n_pairs`, бо `per_sample_cut` не змінюється.
🚨 Вихід останнього шару НЕ обрізається по стовпцях (`if i < depth-1`), тож
обидва `m_deep` — 256-вимірні й віднімаються прямо.

## ЩО МІРЯЄМО
`pr' = pr + β·D`, `β` — один скаляр, ОПТИМУМ АНАЛІТИЧНИЙ (прогноз лінійний):
    `MSE(β) = ⟨e₁e₂⟩ + β⟨D(e₁+e₂)⟩ + β²⟨D²⟩`
⛔ ПЛОМБИ І КОНТРОЛІ:
 (а) `β=0` ⟹ РІВНО базовий MSE;
 (б) ⛔ЧУЖА МЕРЕЖА: `D` від сусідньої мережі (зсув обрізання — ВЛАСТИВІСТЬ
     МЕРЕЖІ, тож жеребний контроль тут не є нулем — урок С75);
 (в) ⛔ШУМ тієї ж норми;
 (г) ЧЕСНИЙ LOO по мережах — `β` фітується без поточної мережі;
 (д) істина — БАНК (ПОВНА мережа!), крос-ціль.
📌 Міряємо на sub93 як базі (найкращий готовий кандидат).

📣 ПЕРЕДБАЧЕНО: знак `β` має вийти ДОДАТНИМ (обрізання ВІДКИДАЄ внесок ⟹ наш
   прогноз занижений, `D>0` його повертає) і СТАБІЛЬНИМ по мережах — це
   головна ознака, що ловимо систематику, а не шум. Розмір: зсув ~20% MSE, але
   замикання знає його неточно (власна похибка форми 0.19% для `m`), тож чекаю
   ×1.01–1.04. Якщо `β` вийде близьким до 1.0 — різниця замикань відтворює
   зсув майже точно, і це сильний результат. Якщо LOO ≈ ⛔чужа мережа —
   різниця несе лише спільну корпусну складову, і треба буде дивитись, чи
   виживає вона як ОДНА КОНСТАНТА для всіх мереж.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/trimbias.py [cuda:N] [NETS]
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

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
L_, NDRAW = 32, 8
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


def run93(wc, npair, dev, seed, cl, aux, clean=True):
    """Повертає (`ours`, `m_deep`) — БЕЗ бленду, щоб (A,B) фітувались окремо.

    `clean=False` ⟹ sub92 (без чистки `δ_Q`) для ⛔контролю.
    """
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
        pd = p.double()
        p2 = pd * pd
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
            k4 = (e4 - 4.0 * mh * e3 + 6.0 * mh2 * e2
                  - 3.0 * mh2 * mh2 - 3.0 * sh2 * sh2)
            dq = (((k3 / (sh2 * sh)) * (AA[i - 1] / 3.0)
                   - (k4 / (sh2 * sh2)) * (BB[i - 1] / 12.0))
                  if clean else 0.0)
            xm = x.double().sum(0) / den
            xq = x.double().pow(2).sum(0) / den
            v = (v + (SP.LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + SP.LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + SP.LAM_Q[i - 1] * ((xq - q_cl[i - 1]) - dq)
                 + SP.LAM_G[i - 1] * (gate - P_cl[i - 1]))
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    return ours, m_deep          # 🔑 БЕЗ бленду: (A,B) фітуються окремо


# скаляри на мережу для MSE(A,B) = ⟨e₁e₂⟩ + A·⟨o·es⟩ + B·⟨d·es⟩
#   + A²⟨o²⟩ + B²⟨d²⟩ + 2AB⟨od⟩,  де e = −T (при A=B=0), es = e₁+e₂ = −(T1+T2)
NV = 2                      # 0: sub93 (з чисткою) · 1: ⛔sub92 (без)
ST = np.zeros((len(FS), NV, 6))
SEALP = 0.0
A0, B0 = 1.0 - W_BLEND, W_BLEND * (1.0 + ALPHA)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
        aux = aux_parts(wc, DEV)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        for d in range(NDRAW):
            sd = 105_000_000 + 733 * n + d
            for v in range(NV):
                o, dd = run93(wc, npair, DEV, sd, cl, aux, clean=(v == 0))
                if v == 0 and d == 0:
                    # ⛔ПЛОМБА (а): (A₀,B₀) відтворюють нинішній sub93
                    pr = A0 * o + B0 * dd
                    ref = o + W_BLEND * ((1.0 + ALPHA) * dd - o)
                    SEALP = max(SEALP, float((pr - ref).abs().max()))
                es = -(T1 + T2)
                ST[n, v, 0] += float((T1 * T2).sum()) / NDRAW
                ST[n, v, 1] += float((o * es).sum()) / NDRAW
                ST[n, v, 2] += float((dd * es).sum()) / NDRAW
                ST[n, v, 3] += float((o * o).sum()) / NDRAW
                ST[n, v, 4] += float((dd * dd).sum()) / NDRAW
                ST[n, v, 5] += float((o * dd).sum()) / NDRAW
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/anneal.npz', ST=ST)

def mse(v, idx, A, B):
    z = ST[idx, v]
    return float((z[:, 0] + A*z[:, 1] + B*z[:, 2] + A*A*z[:, 3]
                  + B*B*z[:, 4] + 2*A*B*z[:, 5]).mean())


def opt(v, idx):
    z = ST[idx, v].sum(0)
    M = np.array([[2*z[3], 2*z[5]], [2*z[5], 2*z[4]]])
    try:
        return np.linalg.solve(M, -np.array([z[1], z[2]]))
    except np.linalg.LinAlgError:
        return np.array([A0, B0])


print(f'\n⛔ПЛОМБА (а) (A₀,B₀) ≡ нинішній бленд: max|Δ| = {SEALP:.2e} '
      f'{"✅" if SEALP < 1e-15 else "🚨"}')
ALL = np.arange(len(FS))
print(f'⛔нинішні константи: A₀={A0:.5f} B₀={B0:.5f} '
      f'(w={W_BLEND:.4f}, α={ALPHA:+.5f})\n')
NM = ['🎯 sub93 (з чисткою q)', '⛔ sub92 (без чистки) — контроль']
print(f'{"варіант":>34s} {"MSE нині":>13s} {"A*":>8s} {"B*":>8s} '
      f'{"MSE опт":>13s} {"×":>8s} {"LOO":>13s} {"×LOO":>8s}')
print('─' * 108)
res = []
for v in range(NV):
    cur = mse(v, ALL, A0, B0)
    A, B = opt(v, ALL)
    va = mse(v, ALL, A, B)
    lo = 0.0
    for n in range(len(FS)):
        o = np.array([j2 for j2 in range(len(FS)) if j2 != n])
        a2, b2_ = opt(v, o)
        lo += mse(v, np.array([n]), a2, b2_)
    lo /= len(FS)
    res.append((cur, A, B, va, lo))
    print(f'{NM[v]:>34s} {cur:13.6e} {A:8.5f} {B:8.5f} {va:13.6e} '
          f'{cur/va:8.4f} {lo:13.6e} {cur/lo:8.4f}')

print('\n🎯 ВИРОК')
for v in range(NV):
    cur, A, B, va, lo = res[v]
    w = 1.0 - A
    al = B / max(1.0 - A, 1e-12) - 1.0
    print(f'   {NM[v]:>34s}: w={w:.4f} (було {W_BLEND:.4f}) · '
          f'α={al:+.5f} (було {ALPHA:+.5f}) · LOO ×{cur/lo:.4f}')
g93, g92 = res[0][0]/res[0][4], res[1][0]/res[1][4]
print(f'\n   ЦІНА УСПАДКОВАНОГО НАЛАШТУВАННЯ (sub93 понад sub92): '
      f'×{g93/max(g92,1e-12):.4f}')
if g93 > 1.003:
    print(f'   ⟹ 🟢 ВІДПАЛ ДАЄ ×{g93:.4f} ⟹ бал 1.928e-07 → '
          f'{1.928e-07/g93:.4e}')
    if g92 > 1.003:
        print('      🚨 але контроль sub92 теж росте ⟹ справа НЕ в чистці `q`,')
        print('      а в тому, що `α` ніколи не підбирався СПІЛЬНО з `w`')
else:
    print('   ⟹ ⛔ константи вже в спільному оптимумі, напружень нема')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
