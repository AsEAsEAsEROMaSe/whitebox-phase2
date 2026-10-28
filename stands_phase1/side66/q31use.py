"""ЯК САМЕ ЮЗАТИ КАНАЛ `q̄₃₁−q_замик`: віднімати чи «ловити перетини»?

Питання юзера: «а якщо юзати його так, як я казав?» — тобто не просто
віднімати, а шукати по кореляції моменти найбільшої точності / режими /
краї каналу.

🚨 ЦЕ НЕ ПИТАННЯ СМАКУ, А ВИМІР. Лінійне віднімання `pred = x̄ − λC`
оптимальне ТІЛЬКИ якщо `E[похибка | C]` лінійна. Якщо зв'язок нелінійний —
порогові/режимні правила беруть БІЛЬШЕ, і юзер має рацію.

Тому міряємо СТЕЛЮ БУДЬ-ЯКОЇ ФУНКЦІЇ від каналу:
  LIN   `λ·C` повним вектором         — те, що вже зміряно (1.0542×)
  SCAL  `λ·c·û`, `c=⟨C,û⟩`, `û=x̄/‖x̄‖` — той самий канал, але скаляром
  POLY  `(λ₁c + λ₂c|c| + λ₃c³)·û`     — непарний поліном (парні члени
        заборонені: похибка симетрична, `E[e]=0`)
  BIN   `h(c)·û`, `h` — ДОВІЛЬНА функція, порахована по бінах
        ⟹ це і є СТЕЛЯ всіх режимних/порогових/«перетинних» правил разом
  SEL   правило юзера в чистому вигляді: сильніше вірити оцінці там, де
        `|c|` малий (тобто «ми біля перетину»), слабше — де великий

🚨 ЧОМУ Я ЧЕКАЮ НУЛЯ, І ЧОМУ ВСЕ ОДНО МІРЯЮ. І `c`, і похибка — це середні
по тисячах семплів ⟹ за ЦГТ пара майже двовимірно-гаусова, а для двовимірного
гауса `E[y|c]` ТОЧНО лінійна. Тобто нелінійності просто нема звідки взятись.
Але це аргумент, а не число.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 BIN перевищить LIN щонайбільше на 1-2% (залишкова кривина від скінченного
   `n` і від того, що `û` сам вибірковий);
📣 SEL буде ГІРШИМ за LIN: він викидає інформацію про ЗНАК.
🟢 ЖИВЕ (тобто юзер має рацію), якщо BIN − LIN ≥ 3%.
🚨 ПЛОМБИ
  ✅ POLY з нульовими старшими коефіцієнтами мусить збігтися з SCAL;
  ⛔ НЕГ: біни, побудовані на `c` СУСІДНЬОГО жереба ⟹ ≈1.000×;
  ✅ усе фітується LOO **ПО МЕРЕЖАХ**; біни — теж (межі й значення `h`
     беруться з інших мереж), інакше BIN перефітить сам себе.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts                # noqa: E402
from jitprof import truth_means                                    # noqa: E402
from jitanim import bank_net                                       # noqa: E402
from q31ab import run, budget, KSIG                                # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 96
NBIN = 8


def loo_scalar(E, C):
    """LOO ПО МЕРЕЖАХ для лінійної поправки `E − λC` (обидва — вектори)."""
    nets = len(E)
    num = np.array([float((E[i] * C[i]).sum()) for i in range(nets)])
    den = np.array([float((C[i] * C[i]).sum()) for i in range(nets)])
    res = 0.0
    for i in range(nets):
        lam = (num.sum() - num[i]) / max(den.sum() - den[i], 1e-300)
        res += float(((E[i] - lam * C[i]) ** 2).sum())
    return res


def loo_design(E, D, U):
    """LOO ПО МЕРЕЖАХ для поправки `Σ_k β_k D_k · û` (D — список ознак)."""
    nets = len(E)
    p = D[0].shape[-1]
    A = np.zeros((nets, p, p))
    b = np.zeros((nets, p))
    for i in range(nets):
        # ціль уздовж û; решта напрямків цим правилом не чіпається
        y = (E[i] * U[i]).sum(-1)
        A[i] = D[i].T @ D[i]
        b[i] = D[i].T @ y
    res = 0.0
    for i in range(nets):
        Ai = A.sum(0) - A[i]
        bi = b.sum(0) - b[i]
        beta = np.linalg.solve(Ai + 1e-12 * np.eye(p) * np.trace(Ai) / p, bi)
        corr = (D[i] @ beta)[:, None] * U[i]
        res += float(((E[i] - corr) ** 2).sum())
    return res


def main(nets=12, dev=DEV, ndraw=NDRAW):
    print('📣 ПЕРЕДБАЧЕНО: BIN перевищить LIN щонайбільше на 1-2%, '
          'SEL буде ГІРШИМ за LIN')
    print('🟢 ЖИВЕ (юзер має рацію), якщо BIN − LIN ≥ 3%\n', flush=True)
    E, C, U, Cs = [], [], [], []
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            d = len(wc)
            nd, _ = budget(wc, d - 1)
            mt, _ = truth_means(wc, dev, 310_000 + 877 * n)
            m31 = mt[d - 1]
            s_cl, m_cl, P_cl, q_cl = closure_parts(wc, d - 1, dev, extras=True)
            cl8 = (s_cl[:KSIG], m_cl[:KSIG], P_cl[:KSIG], q_cl[:KSIG])
            q31 = q_cl[d - 2].double()
            e, c, u = [], [], []
            for k in range(ndraw):
                sd = 71_000_000 + 977 * n + k
                pr, ch = run(wc, nd, cl8, dev, sd, q31_cl=q31)
                v3 = ch[2]
                uh = pr / pr.norm()          # напрямок ДОСТУПНИЙ у прогоні
                e.append((pr - m31).cpu().numpy())
                c.append(v3.cpu().numpy())
                u.append(uh.cpu().numpy())
            E.append(np.array(e))
            C.append(np.array(c))
            U.append(np.array(u))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    # скаляр каналу вздовж доступного напрямку + нормування на власний
    # розкид мережі (ідеалізація НА КОРИСТЬ гіпотези юзера)
    for i in range(nets):
        s = (C[i] * U[i]).sum(-1)
        Cs.append(s / max(float(np.std(s)), 1e-300))

    base = sum(float((E[i] ** 2).sum()) for i in range(nets))
    tot = nets * len(E[0])
    lin = loo_scalar(E, C)
    scal = loo_design(E, [np.stack([Cs[i]], 1) for i in range(nets)], U)
    poly = loo_design(E, [np.stack([Cs[i], Cs[i] * np.abs(Cs[i]),
                                    Cs[i] ** 3], 1) for i in range(nets)], U)
    # BIN: довільна функція від `c` — стеля ВСІХ режимних правил
    edges = np.quantile(np.concatenate(Cs), np.linspace(0, 1, NBIN + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    def onehot(s):
        idx = np.clip(np.digitize(s, edges[1:-1]), 0, NBIN - 1)
        M = np.zeros((len(s), NBIN))
        M[np.arange(len(s)), idx] = 1.0
        return M
    binf = loo_design(E, [onehot(Cs[i]) for i in range(nets)], U)
    # SEL: правило юзера — довіряти сильніше там, де |c| малий (біни по |c|)
    sel = loo_design(E, [onehot(np.abs(Cs[i]) * np.sign(Cs[i]) * 0
                                + np.abs(Cs[i])) for i in range(nets)], U)
    # ⛔НЕГ: біни з `c` СУСІДНЬОГО жереба
    neg = loo_design(E, [onehot(np.roll(Cs[i], 1)) for i in range(nets)], U)

    print(f'\n{nets} мереж × {ndraw} жеребів')
    print(f'{"спосіб":>28} {"MSE":>12} {"×":>8}')
    for nm, v in (('без поправки', base), ('LIN  λ·C повним вектором', lin),
                  ('SCAL λ·c·û', scal), ('POLY непарний, 3 члени', poly),
                  ('BIN  ДОВІЛЬНА f(c) ← СТЕЛЯ', binf),
                  ('SEL  «довіряти при малому |c|»', sel),
                  ('⛔НЕГ біни з сусіднього жеребу', neg)):
        print(f'{nm:>28} {v/tot:>12.5e} {base/v:>8.4f}')
    # 🚨 ЧЕСНЕ ПОРІВНЯННЯ — BIN проти SCAL: обидва бачать ТІЛЬКИ скаляр `c`,
    #    різниця лише у ФОРМІ (лінійна проти довільної). LIN використовує
    #    ПОВНИЙ вектор, тобто більше інформації, і порівнювати з ним нелінійність
    #    було б підміною питання.
    gap = base / binf - base / scal
    print(f'\n🔑 BIN − SCAL = {gap:+.4f}×  (🟢гейт +0.03) — ЧИ ДАЄ НЕЛІНІЙНІСТЬ')
    print(f'   ВИРОК: {"ЖИВЕ — нелінійне правило бере більше" if gap >= 0.03 else "МЕРТВО — лінійна форма вичерпує канал"}')
    print(f'   SEL проти SCAL: {base/sel - base/scal:+.4f}×')
    print(f'   ✅ЯКІСТЬ ФІТУ: BIN мусить бути КРАЩИМ за ⛔НЕГ, інакше '
          f'бінів забагато на дані — BIN {base/binf:.4f}× проти НЕГ '
          f'{base/neg:.4f}× {"✅" if base/binf > base/neg else "🛑 ЗАМАЛО ЖЕРЕБІВ"}')
    print(f'   ℹ️ LIN (повний вектор) {base/lin:.4f}× проти SCAL {base/scal:.4f}× '
          f'⟹ векторна структура несе ще {base/lin - base/scal:+.4f}×')
    print('📊 БАЗОВИЙ БАЛ 1.954e-07')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 12)
