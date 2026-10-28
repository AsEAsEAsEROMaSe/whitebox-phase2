"""ГОЛОВНЕ ПИТАННЯ КЛАСУ: ЧИ ВИСТАЧИТЬ ТРЬОХ МОМЕНТІВ ЗАМІСТЬ ДВОХ?

════════ НАВІЩО (переосмислення С74) ════════
Наш семплінг має похибку **0.164%**, замикання — **0.87%** ⟹ аналітика ВП'ЯТЕРО
ГІРША за вибірку, тож може входити лише як поправка з крихітною вагою. Звідси
ОДНИМ механізмом пояснюються всі смерті: Мелер K=3 (1.003×), Еджворт (1.04×),
глибокий якір (1.043×), варіаційна ширина (1.68× у похибці → 1.02× у балі).
Щоб щось змінилось, замикання мусить стати ТОЧНІШИМ ЗА ВИБІРКУ:
    0.87% → 0.16% — паритет (у 5.3×) ·  0.87% → 0.04% — розрив до лідера (21×)
Ціна це дозволяє: замикання = **7.5%** від бюджету, який ми вже витрачаємо.
Отже обмеження не в цінах, а в тому, що крізь шари несемо ЛИШЕ ДВА МОМЕНТИ.

════════ ЩО МІРЯЄМО ════════
Три рекурсії по 31 кроку, похибка `m` проти істини на КОЖНОМУ шарі:
  1. `база`   — гаусові формули (те, що в сабміті);
  2. `едж`    — Еджворт з ІСТИННИМИ марджинальними `γ₁,γ₂` преактивацій
                (оракул на негаусовість, реалізм на моменти);
  3. `оракул` — істинні `E[relu]` і `E[relu²]` на кожному кроці; недіагональ
                `C` усе одно за формулою ⟹ це СТЕЛЯ марджинального класу, а
                залишок похибки — внесок СТРУКТУРИ КОРЕЛЯЦІЙ.
📐 Формули виведено тут (через `∫_{-t}^∞ He_n φ = He_{n−1}(−t)φ`), другий рядок
   ніде в записах не траплявся, а він і годує наступний шар:
     E[relu]  = s[tΦ+φ] − (γ₁/6)s·t·φ + (γ₂/24)s(t²−1)φ
     E[relu²] = s²[(t²+1)Φ+tφ] + (γ₁/3)s²φ − (γ₂/12)s²t·φ

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 `едж` дасть **1.5–3×** (С73 бачила 1.04× для поправки лише в `m`; тут ще й
   діагональ `C`, тож більше, але не порядок).
📣 `оракул` дасть **3–6×** — і РІЗНИЦЯ між ним і `едж` покаже, скільки коштує
   обрізати ряд на 4-му кумулянті, а РЕШТА до нуля — це структура кореляцій.
🟢 КЛАС ЖИВИЙ, якщо `едж` ≥5× (тобто 0.87% → ≤0.17%, паритет із вибіркою).
🛑 Якщо навіть `оракул` <5×, то марджинальний опис вичерпано ЦІЛКОМ, і жодне
   уточнення моментів не виведе аналітику на рівень вибірки.

🚨 ПЛОМБИ
  ✅ `γ₁=γ₂=0` мусить дати РІВНО базу (тотожність, до останньої цифри);
  ✅ `оракул` мусить бути НЕ ГІРШИМ за `едж` на кожному шарі;
  ⛔ КОНТРОЛЬ: `γ` з ЧУЖОЇ мережі — мусить бути ГІРШЕ за базу;
  🚨 ШУМ ІСТИНИ: два незалежні сіди істини; розкид мусить бути << ефекту
     (С73 обпеклась: «локальна похибка кроку» виявилась шумом істини).
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_, _phi, _Phi               # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
TR_PAIRS, CHUNK = 131_072, 16384
SHOW = (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 31)
# 🚨 ПЕРША ВЕРСІЯ МАЛА ТАВТОЛОГІЮ: режим «оракул» підставляв істинне `E[relu]`
# як `m` і з ним же порівнювався ⟹ похибка тотожно 0 (надрукувало 7.6e11×).
# Правильне питання ділиться на ДВА джерела, і кожне вимикається окремо:
#   `едж`   — негаусовість при обчисленні `E[relu]` з правильних (m,C);
#   `дисп`  — істинна ДИСПЕРСІЯ активацій у діагональ `C` (те, чим шар годує
#             наступний); `m` при цьому все одно рахуємо формулою.
# Якщо навіть `едж+дисп` близьке до 1×, винна НЕДІАГОНАЛЬ `C` = структура
# кореляцій, і марджинальні моменти тут ні до чого.
MODES = ('база', 'едж', 'дисп', 'едж+дисп', '⛔чужі γ')


def truth_moments(wc, dev, seed):
    """Марджинальні моменти ДОАКТИВАЦІЙ (μ, s, γ₁, γ₂) і істинні
    `E[relu]`, `E[relu²]` для кожного шару.

    🚨 Центральні моменти рахуємо ЗСУНУТО (віднімаємо оцінку μ з першого
    чанка): сирі моменти 3-го і 4-го порядку у fp64 втрачають значущі цифри
    на різниці, а нам потрібні саме дрібні відхилення від гауса.
    """
    d = len(wc)
    n = float(2 * TR_PAIRS)
    sh = [None] * d
    acc = [torch.zeros(4, wc[i].shape[1], dtype=torch.float64, device=dev)
           for i in range(d)]
    ra = [torch.zeros(2, wc[i].shape[1], dtype=torch.float64, device=dev)
          for i in range(d)]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(d):
            p = (a @ wc[i]).double()
            if sh[i] is None:
                sh[i] = p.mean(0)
            u = p - sh[i]
            acc[i][0] += u.sum(0)
            acc[i][1] += (u * u).sum(0)
            acc[i][2] += (u ** 3).sum(0)
            acc[i][3] += (u ** 4).sum(0)
            a = torch.relu(p).float()
            ad = a.double()
            ra[i][0] += ad.sum(0)
            ra[i][1] += (ad * ad).sum(0)
        done += k
    mus, sgs, g1s, g2s, r1s, r2s = [], [], [], [], [], []
    for i in range(d):
        m1 = acc[i][0] / n
        c2 = acc[i][1] / n - m1 * m1
        c3 = acc[i][2] / n - 3.0 * m1 * (acc[i][1] / n) + 2.0 * m1 ** 3
        c4 = (acc[i][3] / n - 4.0 * m1 * (acc[i][2] / n)
              + 6.0 * m1 * m1 * (acc[i][1] / n) - 3.0 * m1 ** 4)
        sg = c2.clamp_min(1e-30).sqrt()
        mus.append(m1 + sh[i])
        sgs.append(sg)
        g1s.append(c3 / sg ** 3)
        g2s.append(c4 / sg ** 4 - 3.0)
        r1s.append(ra[i][0] / n)
        r2s.append(ra[i][1] / n)
    return mus, sgs, g1s, g2s, r1s, r2s


def run(wc, mode, g1=None, g2=None, r1=None, r2=None):
    """Рекурсія замикання. mode ∈ {база, едж, оракул}. Повертає `m` по шарах.

    🔑 `μ` і `s` завжди з ПОТОЧНОГО (наближеного) стану — оракульними є лише
    негаусовість (`едж`) або самі перші два моменти активацій (`оракул`).
    """
    d = len(wc)
    dev = wc[0].device
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    out = {}
    for i in range(d):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m1 = mp * Ph + s * ph                       # E[relu] (гаус)
        q1 = (mp * mp + s * s) * Ph + mp * s * ph   # E[relu²] (гаус)
        if mode in ('едж', 'едж+дисп', '⛔чужі γ'):
            a1, a2 = g1[i], g2[i]
            m1 = m1 - (a1 / 6.0) * s * t * ph \
                + (a2 / 24.0) * s * (t * t - 1.0) * ph
            q1 = q1 + (a1 / 3.0) * s * s * ph \
                - (a2 / 12.0) * s * s * t * ph
        m = m1
        C = cp * torch.outer(Ph, Ph) \
            + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        # 🔑 діагональ = ДИСПЕРСІЯ активацій. У режимах `дисп` беремо її
        # істинною; `m` лишається обчисленим — інакше вийде тавтологія.
        if mode in ('дисп', 'едж+дисп'):
            dg = (r2[i] - r1[i] * r1[i]).clamp_min(0.0)
        else:
            dg = q1 - m * m
        C = C + torch.diag(dg - C.diagonal())
        out[i] = m.clone()
    return out


def main(nets=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: едж 1.5–3×, оракул 3–6×')
    print(f'🟢 КЛАС ЖИВИЙ при едж ≥5× (0.87%→0.17%, паритет із вибіркою)')
    print(f'🛑 якщо навіть ОРАКУЛ <5× — марджинальний опис вичерпано\n')

    NL = 32
    E = {k: np.zeros(NL) for k in MODES}
    E2 = np.zeros(NL)          # 🚨 база проти ДРУГОГО сіда істини — шум істини
    NRM = np.zeros(NL)
    cnt = np.zeros(NL)
    prevg = None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            d = len(wc)
            _, _, g1, g2, r1, r2 = truth_moments(wc, dev, 410_000 + 733 * n)
            _, _, _, _, r1b, _ = truth_moments(wc, dev, 950_000 + 733 * n)
            res = {'база': run(wc, 'база'),
                   'едж': run(wc, 'едж', g1, g2),
                   'дисп': run(wc, 'дисп', r1=r1, r2=r2),
                   'едж+дисп': run(wc, 'едж+дисп', g1, g2, r1, r2)}
            # ✅ПЛОМБА ТОТОЖНОСТІ: Еджворт із нульовими γ мусить дати РІВНО базу
            z0 = [torch.zeros_like(t) for t in g1]
            zr = run(wc, 'едж', z0, z0)
            seal = max(float((zr[i] - res['база'][i]).abs().max())
                       for i in range(d))
            if prevg is not None:
                # ⛔ ширина шару різна між мережами — доповнюємо нулями
                gg1, gg2 = [], []
                for i in range(d):
                    nn = wc[i].shape[1]
                    a = torch.zeros(nn, dtype=torch.float64, device=dev)
                    b = torch.zeros(nn, dtype=torch.float64, device=dev)
                    kk = min(prevg[0][i].shape[0], nn) if i < len(prevg[0]) \
                        else 0
                    if kk:
                        a[:kk] = prevg[0][i][:kk]
                        b[:kk] = prevg[1][i][:kk]
                    gg1.append(a); gg2.append(b)
                res['⛔чужі γ'] = run(wc, 'едж', gg1, gg2)
            for i in range(d):
                for k, v in res.items():
                    e = v[i] - r1[i]
                    E[k][i] += float((e * e).sum())
                # 🚨 ШУМ ІСТИНИ = розкид САМОЇ істини між двома сідами
                # (перша версія міряла тут похибку бази проти 2-го сіда —
                # це та сама похибка замикання, а не шум)
                eb = r1[i] - r1b[i]
                E2[i] += float((eb * eb).sum())
                NRM[i] += float((r1[i] * r1[i]).sum())
                cnt[i] += 1
            prevg = ([t.clone() for t in g1], [t.clone() for t in g2])
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    ok = cnt > 0
    NRM[ok] /= cnt[ok]
    E2[ok] /= cnt[ok]
    for k in MODES:
        E[k][ok] /= np.maximum(cnt[ok], 1)

    def pc(arr, i):
        return arr[i] ** 0.5 / max(NRM[i], 1e-300) ** 0.5 * 100

    print(f'\n{nets} мереж. RMS-похибка `m` у % від ‖істини‖ по глибині:')
    print(f'{"шар":>5s} {"база":>9s} {"едж":>9s} {"дисп":>9s} '
          f'{"едж+дисп":>9s} {"⛔чужі γ":>9s} {"кращий×":>8s}')
    print('-' * 70)
    for i in range(NL):
        if not ok[i] or i not in SHOW:
            continue
        b = pc(E['база'], i)
        row = f'{i:5d} {b:8.4f}%'
        for k in ('едж', 'дисп', 'едж+дисп', '⛔чужі γ'):
            row += f' {pc(E[k], i):8.4f}%' if E[k][i] > 0 else f' {"—":>9s}'
        row += f' {b / max(pc(E["едж+дисп"], i), 1e-12):7.2f}×'
        print(row)

    L = int(np.max(np.nonzero(ok)))
    bl, ed = pc(E['база'], L), pc(E['едж'], L)
    orc = pc(E['едж+дисп'], L)
    ge, go = bl / max(ed, 1e-12), bl / max(orc, 1e-12)
    print(f'\n✅ПЛОМБА ТОТОЖНОСТІ (Еджворт із γ=0 проти бази): max|Δ| = '
          f'{seal:.3e} {"✅" if seal < 1e-12 else "🛑 НЕ ТОТОЖНО"}')
    print(f'🚨ШУМ ІСТИНИ (розкид істини між сідами): {pc(E2, L):.4f}% проти '
          f'похибки {bl:.4f}% ⟹ {"✅" if pc(E2, L) < 0.3 * bl else "🛑 ЗАВЕЛИКИЙ"}')
    if E['⛔чужі γ'][L] > 0:
        print(f'⛔КОНТРОЛЬ (γ з чужої мережі): {pc(E["⛔чужі γ"], L):.4f}% '
              f'{"✅гірше за базу" if E["⛔чужі γ"][L] > E["база"][L] else "🛑 НЕ гірше"}')
    print(f'\n🔑 ШАР {L}: база {bl:.4f}% · едж {ed:.4f}% ({ge:.2f}×) · '
          f'едж+дисп {orc:.4f}% ({go:.2f}×)')
    print(f'🔑 ВИРОК: {"ЖИВИЙ" if ge >= 5.0 else "МЕРТВИЙ"} '
          f'(🟢гейт едж ≥5×; паритет із вибіркою = 0.164%)')
    if go < 5.0:
        print(f'🛑 НАВІТЬ ІДЕАЛЬНІ МАРДЖИНАЛЬНІ МОМЕНТИ {go:.2f}× ⟹ ВИЧЕРПАНО: '
              f'решта похибки — СТРУКТУРА КОРЕЛЯЦІЙ, не моменти')
    print(f'📊 БАЛ: 1.972e-07 (без змін — вимір стендовий)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
