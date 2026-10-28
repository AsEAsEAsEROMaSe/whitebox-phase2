"""ЧИ НЕСЕ НИЗЬКИЙ РАНГ `C` ВЕЛИЧИНУ `σ`? — ЕТАП 1, ВИРІШУЄ ДОЛЮ КЛАСУ. (С87)

## ЗВІДКИ — ЗАВДАННЯ ЮЗЕРА
«Напрям 1 (самоузгоджена `σ` понейронно) — і всі можливі варіанти математики
й алгоритмів.»

## 📐 ВИВЕДЕНО (не вгадано)
Потрібна не матриця, а КВАДРАТИЧНА ФОРМА:
    `σ_i² = (Wᵀ C_a W)_ii = w_iᵀ C_a w_i`   — 256 чисел, не 65536.
Якщо `C_a ≈ D + UΛUᵀ`, то
    `σ_i² = Σ_j W_ji² D_j + Σ_r λ_r (u_rᵀ w_i)²`   ціна: k матвеків.
При `k=3` це 3.1 матвека/шар = РІВНО те, що дозволяє `F`=1.3e7 лідера.

🔑 І ФОРМАТ ЗАМИКАЄТЬСЯ під нашим замиканням Мелера:
  член 1  `Φ_jΦ_k C_jk`     → `diag(Φ²D) + (ΦU)Λ(ΦU)ᵀ`        ранг k
  член 2  `½C_jk²(φ/σ)(φ/σ)ᵀ` → `Σ_rs λ_rλ_s(u_r⊙u_s)(u_r⊙u_s)ᵀ` ранг k(k+1)/2
⟹ самоузгоджена низькорангова рекурсія ІСНУЄ. Але будувати її має сенс лише
якщо низький ранг взагалі несе `σ` — це і міряє цей стенд.

## ⚠️ ЗАПИСАНИЙ ЗАКОН, ЯКИЙ ЦЬОМУ СУПЕРЕЧИТЬ
«ЕФЕКТИВНИЙ РАНГ ОМАНЛИВИЙ: міряє ДЕ ДИСПЕРСІЯ; домінантні компоненти за ЦГТ
майже гаусові ⟹ пояснює ВСІ провали низького рангу.» Еф. ранг активацій падає
161.6 → 2.5, і це спокуса, на якій уже загинули гілки. Тому тут — СТЕЛЯ з
ІСТИННОЮ `C`, а не рекурсія: якщо стеля не дотягує, рекурсію не будуємо.

## 🎯 ЩО САМЕ МІРЯЄТЬСЯ
Істинна `C_a` (МК) шару `l` → обрізається до рангу `k` → квадратична форма
`w_iᵀ C̃ w_i` → відносна похибка `σ` проти точної `w_iᵀ C w_i`.
🎯 ПОРІГ: щоб зрівнятися з `sub97`, `σ` треба на **0.465%**; щоб побити
лідера — на **0.153%** ([[reference-direct-mse-instrument-and-its-seal]]).

## 🧮 ВАРІАНТИ (усі — різна математика, не різні гіперпараметри)
  V0  лише діагональ (k=0)                — база класу, відомо ~4-10%
  Vk  D + rank-k на ЗАЛИШКУ, діаг ТОЧНА   — основний
  Vc  rank-k на КОРЕЛЯЦІЙНІЙ матриці      — інша метрика, ті самі витрати
  Vm  mean-field: `C ≈ D + ρ·√D√Dᵀ`       — 1 матвек, ρ самоузгоджений скаляр
  Vs  rank-k у базисі ВАГ (`Wᵀ` замість власних) — базис відомий БЕЗ `C`
  ⛔R  ВИПАДКОВІ напрямки того ж рангу     — КОНТРОЛЬ

## ⛔ПЛОМБИ
 P1 `k = 256` (повний ранг) МУСИТЬ дати рівно 0 (машинна точність) — інакше
    квадратична форма зібрана неправильно.
 P2 `k = 0` мусить відтворити відоме 4-10% (`onefac.py`) — інакше не той об'єкт.
 P3 ⛔ВИПАДКОВІ напрямки того ж рангу мусять бути ПОМІТНО гіршими за власні.
    Якщо ні — «низький ранг» нічого не знає, працює сам факт корекції.
 P4 шум істини: піврозмір МК, окремою колонкою.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) `k=0` дасть 4–10%. Шанс **80%** (відомо з `onefac.py`).
 (б) `k=4` дасть <1%. Шанс **30%** — закон про оманливий еф. ранг проти.
 (в) `k ≤ 8` дасть ≤0.465% (рівень sub97). Шанс **20%**.
 (г) точність РОСТЕ з глибиною (еф. ранг падає 161.6→2.5). Шанс 65%.
 (д) ⛔випадкові напрямки будуть помітно гіршими. Шанс 85%. Якщо НІ — клас
     мертвий одразу, бо виграє не структура.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/lowrk87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
# 🚨 БАТЧ МУСИТЬ БУТИ МЕНШИЙ ЗА ПІВНАБОРУ, інакше «половинна» вибірка збігається
#    з повною, а дільник `NS//2` подвоює середнє ⟹ пломба P4 показує фальшиві
#    74–94% замість справжніх часток відсотка (спіймано на димовому прогоні).
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
KS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
PROBE = (3, 7, 11, 15, 19, 23, 27, 30)      # шари, де міряємо
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def qform(W, C):
    """`σ_i² = (Wᵀ C W)_ii` — БЕЗ побудови повної матриці.

    `(WᵀCW)_ii = Σ_jk W_ji W_ki C_jk = w_iᵀ C w_i` ⟹ поелементно по стовпцях:
    `sum((C @ W) * W, dim=0)`. Це `n²` на нейрон, `n³` на шар — тут ми його
    платимо чесно, бо міряємо ТОЧНІСТЬ, а не ціну."""
    return torch.clamp((C @ W * W).sum(0), min=1e-30)


def eig_desc(C):
    """Одна декомпозиція на матрицю — власні пари за спаданням.

    🚨 Раніше тут стояв `eigh` ВСЕРЕДИНІ циклу по `k` ⟹ 40 декомпозицій на шар
    замість 3, і прогін не збігався за розумний час. Декомпозиція не залежить
    від `k` — обрізання є просто зрізом стовпців."""
    ev, U = torch.linalg.eigh(C)
    idx = torch.argsort(ev, descending=True)
    return ev[idx], U[:, idx]


def lowrank_keep_diag(C, k, ev=None, U=None):
    """`C̃ = U_kΛ_kU_kᵀ` (у ЗАДАНОМУ базисі) + ТОЧНА діагональ.

    🔑 Діагональ лишається точною завжди: вона коштує `O(n)` і в рекурсії
    приходить із замикання БЕЗКОШТОВНО. Апроксимується РІВНО позадіагональ —
    те, що скан `sigscan87` назвав головним винуватцем (`off` corr −0.4111
    із залишком `σ`).
    Якщо `ev is None`, `U` — ортонормований базис (ваги / випадковий), і
    проєкція робиться як `U(UᵀCU)Uᵀ`; інакше — власна пара."""
    d = torch.diagonal(C).clone()
    if k <= 0:
        return torch.diag(d)
    if ev is None:
        Uk = U[:, :k]
        Ct = Uk @ (Uk.T @ C @ Uk) @ Uk.T
    else:
        Ct = (U[:, :k] * ev[:k]) @ U[:, :k].T
    return Ct + torch.diag(d - torch.diagonal(Ct))


def meanfield(C):
    """`C ≈ D + ρ·√D√Dᵀ` — ОДИН скаляр, самоузгоджений у сенсі найм. квадратів.

    Найдешевший можливий варіант: у рекурсії коштує 1 матвек. `ρ` береться
    оптимальним (оракульним) ⟹ це СТЕЛЯ mean-field, не реалізація."""
    d = torch.diagonal(C)
    r = torch.sqrt(torch.clamp(d, min=1e-30))
    R = torch.outer(r, r)
    off = C - torch.diag(d)
    Roff = R - torch.diag(d)
    den = float((Roff * Roff).sum())
    rho = float((off * Roff).sum()) / den if den > 1e-25 else 0.0
    return torch.diag(d) + rho * Roff, rho


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · {DEV}', flush=True)
print(f'ранги: {KS} · шари: {PROBE}\n', flush=True)

ERR = {k: np.zeros((len(FS), len(PROBE))) for k in KS}
ERR_C = {k: np.zeros((len(FS), len(PROBE))) for k in KS}      # кореляційна
ERR_R = {k: np.zeros((len(FS), len(PROBE))) for k in KS}      # ⛔випадкова
ERR_W = {k: np.zeros((len(FS), len(PROBE))) for k in KS}      # базис ваг
ERR_MF = np.zeros((len(FS), len(PROBE)))
RHO_MF = np.zeros((len(FS), len(PROBE)))
NOISE = np.zeros((len(FS), len(PROBE)))
ERANK = np.zeros((len(FS), len(PROBE)))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        SUM = torch.zeros(len(PROBE), W_, dtype=torch.float64, device=DEV)
        GRAM = torch.zeros(len(PROBE), W_, W_, dtype=torch.float64, device=DEV)
        SUMH = torch.zeros(len(PROBE), W_, dtype=torch.float64, device=DEV)
        GRAMH = torch.zeros(len(PROBE), W_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(9100 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                a = torch.relu(a @ W32[q])
                if q in PROBE:
                    j = PROBE.index(q)
                    ad = a.double()
                    SUM[j] += ad.sum(0)
                    GRAM[j] += ad.T @ ad
                    if d < NS // 2:
                        SUMH[j] += ad.sum(0)
                        GRAMH[j] += ad.T @ ad
            d += b

        for j, q in enumerate(PROBE):
            mu = SUM[j] / NS
            C = GRAM[j] / NS - torch.outer(mu, mu)
            C = 0.5 * (C + C.T)
            Wn = W[q + 1]
            s2 = qform(Wn, C)
            s_ex = torch.sqrt(s2)

            # ⛔P4 шум істини: та сама величина на половині семплів
            muh = SUMH[j] / (NS // 2)
            Ch = GRAMH[j] / (NS // 2) - torch.outer(muh, muh)
            NOISE[n, j] = float(torch.sqrt(
                (((torch.sqrt(qform(Wn, 0.5 * (Ch + Ch.T))) - s_ex) ** 2).mean()
                 / (s_ex * s_ex).mean())))

            ev = torch.linalg.eigvalsh(C)
            ev = torch.clamp(ev, min=0.0)
            ERANK[n, j] = float(ev.sum() ** 2 / (ev * ev).sum())

            dg = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
            DG = torch.outer(dg, dg)
            Corr = C / DG
            gr = torch.Generator(device=DEV)
            gr.manual_seed(4400 + 31 * n + q)
            # 🔑 ТРИ декомпозиції на шар, а не 40: базис від `k` не залежить.
            evC, UC = eig_desc(C)
            evR, UR = eig_desc(Corr)
            # базис ВАГ: головні ліві сингулярні напрямки `W_{q+1}` — відомі
            # БЕЗ жодної `C`, тобто в рекурсії безкоштовні
            Ub = torch.linalg.svd(Wn, full_matrices=False)[0]
            # ⛔КОНТРОЛЬ: випадковий ортонормований базис (вкладений по `k`)
            Ur = torch.linalg.qr(torch.randn(W_, W_, dtype=C.dtype,
                                             device=C.device, generator=gr))[0]

            def rel(Ct):
                return float(torch.sqrt(
                    (((torch.sqrt(qform(Wn, Ct)) - s_ex) ** 2).mean()
                     / (s_ex * s_ex).mean())))

            for k in KS:
                ERR[k][n, j] = rel(lowrank_keep_diag(C, k, evC, UC))
                ERR_C[k][n, j] = rel(
                    lowrank_keep_diag(Corr, k, evR, UR) * DG)
                ERR_R[k][n, j] = rel(lowrank_keep_diag(C, k, None, Ur))
                ERR_W[k][n, j] = rel(lowrank_keep_diag(C, k, None, Ub))
            Cmf, rho = meanfield(C)
            ERR_MF[n, j] = rel(Cmf)
            RHO_MF[n, j] = rho
    print(f'  net{n} · k=0 {ERR[0][n].mean():.3%} · k=4 {ERR[4][n].mean():.3%} '
          f'· k=32 {ERR[32][n].mean():.3%} · MF {ERR_MF[n].mean():.3%}',
          flush=True)

print(f'\n{"═" * 78}\n⛔ПЛОМБИ')
print(f' P1 k=256 (повний ранг) = {ERR[256].mean():.3e}   (мусить бути ~0)  '
      f'{"✅" if ERR[256].mean() < 1e-9 else "⛔"}')
print(f' P2 k=0 (лише діагональ) = {ERR[0].mean():.3%}   (очік. 4–10%)  '
      f'{"✅" if 0.03 < ERR[0].mean() < 0.13 else "⛔"}')
print(f' P4 шум істини = {NOISE.mean():.4%}  (мусить бути ≪ порогу 0.465%)')

print(f'\n{"═" * 78}')
print('ТОЧНІСТЬ `σ` ЗАЛЕЖНО ВІД РАНГУ (усереднено по шарах і мережах)')
print(f'{"k":>5s} {"власні":>9s} {"кореляц":>9s} {"базис W":>9s} '
      f'{"⛔випадк":>9s} {"чи ≤0.465%":>11s}')
for k in KS:
    ok = '✅ sub97' if ERR[k].mean() <= 0.00465 else ''
    if ERR[k].mean() <= 0.00153:
        ok = '🥇 ЛІДЕР'
    print(f'{k:5d} {ERR[k].mean():8.4%} {ERR_C[k].mean():8.4%} '
          f'{ERR_W[k].mean():8.4%} {ERR_R[k].mean():8.4%} {ok:>11s}')
print(f'{"MF":>5s} {ERR_MF.mean():8.4%} {"":>9s} {"":>9s} {"":>9s}'
      f'   (ρ̄={RHO_MF.mean():.4f})')
print(f'\n🎯 ПОРОГИ: 0.465% = зрівнятися з sub97 · 0.153% = побити лідера')

print(f'\n{"═" * 78}')
print('ПО ШАРАХ — чи росте точність із глибиною (еф. ранг падає)?')
print('шар      : ' + ' '.join(f'{q + 1:8d}' for q in PROBE))
print('еф.ранг  : ' + ' '.join(f'{ERANK[:, j].mean():8.1f}'
                               for j in range(len(PROBE))))
for k in (0, 2, 4, 8, 16, 32):
    print(f'k={k:<3d}     : ' + ' '.join(f'{ERR[k][:, j].mean():8.3%}'
                                         for j in range(len(PROBE))))
print('⛔вип k=32: ' + ' '.join(f'{ERR_R[32][:, j].mean():8.3%}'
                                for j in range(len(PROBE))))
print('шум      : ' + ' '.join(f'{NOISE[:, j].mean():8.4%}'
                               for j in range(len(PROBE))))

need = [k for k in KS if ERR[k].mean() <= 0.00465]
print(f'\n📣 ВИРОК: мінімальний ранг для рівня sub97 (0.465%) = '
      f'{need[0] if need else "НЕ ДОСЯГНУТО ЖОДНИМ"}')
if need:
    kk = need[0]
    fl = 2.0 * kk * W_ * W_ * (L_ - 1)
    print(f'   💰 ціна рекурсії при k={kk}: {fl:.2e} флопів = '
          f'{fl / 2.72e11:.3%} бюджету  (лідер платить F=1.30e7)')
