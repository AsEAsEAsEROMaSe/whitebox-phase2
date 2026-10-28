"""ЧИ МАЄ `S_jk = κ₃(a_j,a_j,a_k)` НИЗЬКИЙ РАНГ. (С86)

## ЗВІДКИ — ВКАЗІВКА ЮЗЕРА
«Шукай спосіб витягнути це понейронне.»

## ЧОМУ САМЕ РАНГ, І ЧИМ ЦЕ НЕ ФІТ
`perneuron.py`: понейронних `κ₃`,`κ₄` вистачає на ×12.88, але здобути їх нема як
(4 способи мертві). Головний член (С86, 93.6% норми `κ₃`):
    `κ₃(h_i) ≈ 3·Σ_{j≠k} W_ji²·W_ki·S_jk`,   `S_jk = κ₃(a_j,a_j,a_k)`
Повна `S` коштує O(n³). 🔑АЛЕ якщо `S ≈ Σ_r u_r v_rᵀ` рангу `r`, то
    `κ₃(h_i) = 3·Σ_r ([(W²)ᵀu_r]_i·[Wᵀv_r]_i − Σ_j W_ji³ u_rj v_rj)`
— кілька матвеків, `~1.3e5·r` флопів на шар.
⚠️ЦЕ НЕ ФІТ ОЗНАКАМИ. `lowrank_rungs.log` уже показав, що фіт зміщення
провалюється в СТРУКТУРІ залишку (оракул 5.6×, реальний фіт 1.087×). Тут
інше: ТОЧНА факторизація точної величини. Питання лише, чи ранг малий.
📌 Непрямий доказ, що структура є: еф. ранг `C_ℓ` падає 165.6 → 4.8 з глибиною
(`arena62/crank.py`, шапка `side66/lowrank.py`).

## 🔑 ГІПОТЕЗА ПРО ФАКТОРИ (rank collapse)
`perneuron.py` зміряв corr(γ₃, sign t) = 0.88 ⟹ у глибині `a_j ≈ c_j·u(z)`.
Тоді `κ₃(a_j,a_j,a_k) ≈ c_j²·c_k·κ₃(u)` — РАНГ 1 з ВІДОМИМИ факторами:
`u_j ∝ σ_j²`, `v_k ∝ σ_k`, а `σ` замикання дає з масштабом 1.0008.
⟹ якщо це так, `κ₃` понейронно стає дешево обчислюваним БЕЗ SVD.

## 🎯 ЩО МІРЯЄМО
1. спектр `S` — частка норми в 1/2/3/8 головних компонентах;
2. `κ₃(h)` через `S` УРІЗАНУ до рангу 1,2,3 — проти повної `S` і проти ІСТИНИ;
3. 🔑МОДЕЛЬ `S = α·σ²σᵀ` — те, що ми РЕАЛЬНО можемо порахувати з ваг.

⛔ПЛОМБИ:
 (1) повна `S` мусить дати корел ≈0.93 з істинним `κ₃(h)` — це вже зміряно
     С86 (`side66/jjk.py`); розбіжність = баг ТУТ;
 (2) ⛔контроль: `S` із переставленими індексами — корел мусить впасти до ~0;
 (3) урізання НЕ може дати корел вищу за повну `S`.

📣 ПЕРЕДБАЧЕНО: спектр спадає швидко (r=1 ≥50% норми), АЛЕ модель `σ²σᵀ` дасть
   помітно менше за SVD-ранг-1, бо справжній вектор несе ще й знакову структуру.
   Шанс, що r=3 тримає корел ≥0.8 від повної: **55%**.
   Шанс, що модель `σ²σᵀ` дасть корел ≥0.5: **30%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/srank.py [cuda:N] [NETS] [LOG2N]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА.
"""
import glob
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
BATCH = 1 << 12
L_, W_ = 32, 256
LAYERS = [3, 7, 11, 15, 19, 23, 27, 30]
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} · шари {LAYERS} · {DEV}\n', flush=True)


def corr(x, y):
    u, v = x - x.mean(), y - y.mean()
    d = float(torch.sqrt((u * u).sum() * (v * v).sum()))
    return float((u * v).sum()) / d if d > 0 else 0.0


ROWS = []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        # ОДИН прохід: сирі E[a], E[a²], E[a_j a_k], E[a_j² a_k], моменти h
        P1 = {i: torch.zeros(2, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        P2 = {i: torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        P3 = {i: torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        HM = {i: torch.zeros(3, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        g = torch.Generator(device=DEV)
        g.manual_seed(6600 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                # 🚨 ІНДЕКСАЦІЯ (перша редакція мала ЗСУВ НА ШАР, зловила ПЛОМБА(1)):
                #    на вході ітерації `q` змінна `a` — активація шару `q`
                #    (q=0 ⟹ вхід z). Далі `h = a @ W[q]` — преактивація шару q+1.
                #    Для цільового шару L усе береться на ітерації q = L−1:
                #    S з `a` (шар L−1), ваги W[L−1], істина з `h` (шар L).
                L = q + 1
                if L in P1:
                    ad = a.double()
                    P1[L][0] += ad.sum(0)
                    P1[L][1] += (ad * ad).sum(0)
                    P2[L] += ad.T @ ad
                    P3[L] += (ad * ad).T @ ad
                h = a @ W[q]
                if L in HM:
                    hd = h.double()
                    p = hd
                    for k in range(3):
                        HM[L][k] += p.sum(0)
                        p = p * hd
                a = torch.relu(h)
            d += b

        for i in LAYERS:
            m1 = P1[i][0] / NS
            m2 = P1[i][1] / NS
            Ejk = P2[i] / NS                                  # E[a_j a_k]
            Ejjk = P3[i] / NS                                 # E[a_j² a_k]
            # κ₃(a_j,a_j,a_k) = E[ã_j²ã_k] через сирі моменти
            S = (Ejjk - 2.0 * m1[:, None] * Ejk
                 - m2[:, None] * m1[None, :] + 2.0 * (m1 * m1)[:, None] * m1[None, :])
            va = m2 - m1 * m1
            sg = torch.sqrt(torch.clamp(va, min=1e-30))
            hm1, hm2, hm3 = HM[i][0] / NS, HM[i][1] / NS, HM[i][2] / NS
            k3_true = hm3 - 3.0 * hm1 * hm2 + 2.0 * hm1 ** 3
            Wn = W[i - 1].double()      # 🚨 ваги, що ведуть з шару i−1 у шар i
            w2 = Wn * Wn

            def jjk(Sm):
                G = (w2 * (Sm @ Wn)).sum(0)
                D = (w2 * Wn * Sm.diagonal()[:, None]).sum(0)
                return 3.0 * (G - D)

            full = jjk(S)
            U, sv, Vh = torch.linalg.svd(S)
            tot = float((sv * sv).sum())
            sp = [float((sv[:r] ** 2).sum()) / tot for r in (1, 2, 3, 8)]
            ck = [corr(jjk((U[:, :r] * sv[:r]) @ Vh[:r]), k3_true) for r in (1, 2, 3)]
            mdl = torch.outer(sg * sg, sg)
            mdl = mdl * (float((S * mdl).sum()) / float((mdl * mdl).sum()))
            pm = torch.randperm(W_, device=DEV)
            ROWS.append([i, sp[0], sp[1], sp[2], sp[3], corr(full, k3_true),
                         ck[0], ck[1], ck[2], corr(jjk(mdl), k3_true),
                         corr(jjk(S[pm][:, pm]), k3_true),
                         corr(U[:, 0].abs(), sg * sg), corr(Vh[0].abs(), sg)])
    q = [r for r in ROWS if r[0] == 15][-1]
    print(f'  … {n+1}/{len(FS)} seed {sd0} · шар15: r=1 несе {q[1]*100:.1f}% · '
          f'корел повна {q[5]:.3f} → r=1 {q[6]:.3f} → модель σ²σᵀ {q[9]:.3f}',
          flush=True)
    del W, P1, P2, P3, HM
    torch.cuda.empty_cache()

M = np.array(ROWS)
np.save('/mnt/win_d/llm/White-Box/arena62/logs/srank.npy', M)
print('\n' + '=' * 98)
print('📊 СПЕКТР `S` І ЯКІСТЬ УРІЗАННЯ (корел `κ₃(h)` з ІСТИНОЮ)')
print(f'{"шар":>4s} {"r=1":>7s} {"r=2":>7s} {"r=3":>7s} {"r=8":>7s} | '
      f'{"ПОВНА":>7s} {"r=1":>7s} {"r=2":>7s} {"r=3":>7s} {"σ²σᵀ":>7s} {"⛔ctrl":>7s}')
print('─' * 98)
for i in LAYERS:
    q = M[M[:, 0] == i]
    print(f'{i:4d} {q[:,1].mean()*100:6.1f}% {q[:,2].mean()*100:6.1f}% '
          f'{q[:,3].mean()*100:6.1f}% {q[:,4].mean()*100:6.1f}% | '
          f'{q[:,5].mean():7.3f} {q[:,6].mean():7.3f} {q[:,7].mean():7.3f} '
          f'{q[:,8].mean():7.3f} {q[:,9].mean():7.3f} {q[:,10].mean():7.3f}')
print('─' * 98)
print(f'СЕРЕДНЄ: r=1 несе {M[:,1].mean()*100:.1f}% норми · r=3 {M[:,3].mean()*100:.1f}%')
print(f'   корел: ПОВНА {M[:,5].mean():.3f} → r=1 {M[:,6].mean():.3f} → '
      f'r=2 {M[:,7].mean():.3f} → r=3 {M[:,8].mean():.3f}')
print(f'   🔑МОДЕЛЬ σ²σᵀ (без SVD, з ваг): {M[:,9].mean():.3f}')
print(f'⛔ПЛОМБА(2) переставлена S: {M[:,10].mean():+.4f} (мусить ≈0)')
print(f'📌 головні вектори: лівий↔σ² {M[:,11].mean():.3f} · правий↔σ {M[:,12].mean():.3f}')
print('\n🎯 r=1..3 тримає корел ≈ повної ⟹ `κ₃` стає O(r·n²) замість O(n³).')
print('   Модель σ²σᵀ близька ⟹ фактори беруться з ваг і SVD не потрібен.')
