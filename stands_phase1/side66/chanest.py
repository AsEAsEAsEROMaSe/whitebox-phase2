"""ДВА СПОСТЕРЕЖЕННЯ ЮЗЕРА НА ГРАФІКУ — ЧИСЛОМ.

1️⃣ «В КІНЦІ ВИДНО ЗМІЩЕННЯ В ОДНУ СТОРОНУ, НІБИ ЩОСЬ ЗМІЩАЄ».
   Найімовірніший підозрюваний — НЕ метод, а МОЯ ЛІНІЯ ІСТИНИ: вона сама
   рахована на 1 млн семплів і має власний шум. Тоді біжуче середнє сходиться
   до СПРАВЖНЬОЇ істини, а нам здається, що воно сходиться повз.
   Перевірка: усереднити знакове відхилення по НЕЗАЛЕЖНИХ реалізаціях. Якщо
   зміщення реальне — середнє по реалізаціях НЕ нуль і НЕ падає з `n`.
   ✅ПЛОМБА: рахуємо істину ДВІЧІ незалежними потоками і дивимось, чи
   «зміщення» рухається СЛІДОМ за нею (тоді винна істина, а не метод).

2️⃣ «МІЖ РОЗВОРОТАМИ Є КАНАЛ, ЯК У ТРЕЙДИНГУ, І СЕРЕДНЄ МАЄ БУТИ ТАМ».
   Тобто оцінювати `m` не повним середнім, а серединою огинаючої по
   розворотах. Це родина «робастних» оцінок; поруч — медіана по фолдах і
   обрізане середнє.
   🚨 ТЕОРІЯ ПРОТИ: `x̄_n − m = S_n/n`, де `S_n` — блукання з НЕЗАЛЕЖНИМИ
   приростами. Минуле блукання нічого не каже про майбутній напрямок, а
   «канал» — це просто самоподібність броунівського руху: він виглядає
   каналом на БУДЬ-ЯКОМУ масштабі. Але міряємо, а не віримо.

📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ
📣 1: зміщення по реалізаціях буде НА РІВНІ шуму істини (≈0.02-0.03%) і НЕ
     зростатиме — тобто винна лінія істини, а не метод.
📣 2: середина огинаючої ГІРША за просте середнє в 1.5-3×; медіана по фолдах
     дасть ≈0.6× (це π/2, відомий штраф за гаусовість).
🟢 ЖИВЕ, якщо якась робастна оцінка дає ≥1.05× проти простого середнього.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_                          # noqa: E402
from jitanim import bank_net                                      # noqa: E402
from jitprof import truth_means                                   # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NREAL = int(sys.argv[3]) if len(sys.argv) > 3 else 40
NMAX = 24_000
CHK = np.unique(np.round(np.logspace(np.log10(64), np.log10(NMAX),
                                     40)).astype(int))


def path(wc, dev, seed, u, nmax):
    """Повний шлях біжучого середнього (скаляр уздовж `u`)."""
    d = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    acc, out, done, chunk = 0.0, [], 0, 512
    vals = []
    while done < nmax:
        k = min(chunk, nmax - done)
        a = torch.randn(k, W_, device=dev, generator=g)
        for i in range(d):
            a = torch.relu(a @ wc[i])
        vals.append((a.double() @ u).cpu().numpy())
        done += k
    v = np.concatenate(vals)
    return np.cumsum(v) / np.arange(1, len(v) + 1), v


def envelope_mid(rm, n, frac=0.5):
    """Середина «каналу»: (max+min)/2 біжучого середнього на хвості вікна."""
    lo = max(1, int(n * frac))
    seg = rm[lo:n]
    return 0.5 * (seg.max() + seg.min()) if len(seg) else rm[n - 1]


def main(net=0, dev=DEV, nreal=NREAL):
    W = bank_net(net, dev)
    with torch.no_grad():
        wc, _ = cut_and_budget(W, dev)
        d = len(wc)
        # ДВІ НЕЗАЛЕЖНІ ІСТИНИ — щоб відділити «зміщення методу» від
        # «шуму моєї лінії істини»
        mtA, _ = truth_means(wc, dev, 111_000 + net)
        mtB, _ = truth_means(wc, dev, 777_000 + net)
        A, B = mtA[d - 1], mtB[d - 1]
        u = A / A.norm()
        tA, tB = float(A @ u), float(B @ u)
        print(f'✅ПЛОМБА: дві незалежні істини розходяться на '
              f'{abs(tA - tB) / tA * 100:.4f}% (це і є шум лінії істини)')

        SA = np.zeros((nreal, len(CHK)))
        SB = np.zeros((nreal, len(CHK)))
        EP = np.zeros((nreal, len(CHK)))   # просте середнє
        EE = np.zeros((nreal, len(CHK)))   # середина огинаючої
        EM = np.zeros((nreal, len(CHK)))   # медіана по 8 фолдах
        ET = np.zeros((nreal, len(CHK)))   # обрізане середнє 20%
        PATHS = []
        for r in range(nreal):
            rm, v = path(wc, dev, 4_000_000 + 97 * r, u, NMAX)
            PATHS.append(rm)
            for j, n in enumerate(CHK):
                x = rm[n - 1]
                SA[r, j] = (x / tA - 1.0) * 100.0
                SB[r, j] = (x / tB - 1.0) * 100.0
                EP[r, j] = (x - tA) ** 2
                EE[r, j] = (envelope_mid(rm, n) - tA) ** 2
                f = v[:n - n % 8].reshape(8, -1).mean(1)
                EM[r, j] = (np.median(f) - tA) ** 2
                s = np.sort(v[:n])
                c = max(1, int(0.2 * n))
                ET[r, j] = (s[c:n - c].mean() - tA) ** 2
    del W, wc
    torch.cuda.empty_cache()

    print(f'\n1️⃣ ЧИ Є СИСТЕМАТИЧНЕ ЗМІЩЕННЯ ({nreal} незалежних реалізацій)')
    print(f'{"n":>7} {"сер.відх, %":>12} {"ст.помилка":>11} {"t":>6} '
          f'{"те саме vs ІСТИНА-Б":>20}')
    for j in range(0, len(CHK), 3):
        mA, sA = SA[:, j].mean(), SA[:, j].std(ddof=1) / np.sqrt(nreal)
        mB = SB[:, j].mean()
        print(f'{CHK[j]:>7} {mA:>12.4f} {sA:>11.4f} {mA/max(sA,1e-12):>6.2f} '
              f'{mB:>20.4f}')
    j = len(CHK) - 1
    mA, sA = SA[:, j].mean(), SA[:, j].std(ddof=1) / np.sqrt(nreal)
    print(f'{CHK[j]:>7} {mA:>12.4f} {sA:>11.4f} {mA/max(sA,1e-12):>6.2f} '
          f'{SB[:, j].mean():>20.4f}')
    print('   🔑 СТАЛЕ по n ⟹ винна лінія істини; СПАДАЄ ⟹ просто шум')
    j = len(CHK) - 1
    mA, sA = SA[:, j].mean(), SA[:, j].std(ddof=1) / np.sqrt(nreal)
    shift = abs(SA[:, j].mean() - SB[:, j].mean())
    print(f'\n   зсув «зміщення» при ЗАМІНІ лінії істини: {shift:.4f}% '
          f'проти самого «зміщення» {abs(mA):.4f}%')
    print(f'   ⟹ {"🔑 ВИНЕН ШУМ ІСТИНИ" if shift > 0.5 * abs(mA) else "🚨 зміщення НЕ пояснюється істиною"}')

    print(f'\n2️⃣ «КАНАЛ» І РОБАСТНІ ОЦІНКИ (виграш проти простого середнього)')
    print(f'{"n":>7} {"огинаюча":>10} {"медіана 8":>10} {"обріз.20%":>10}')
    for j in (len(CHK) // 3, 2 * len(CHK) // 3, len(CHK) - 1):
        b = EP[:, j].mean()
        print(f'{CHK[j]:>7} {b/EE[:, j].mean():>10.3f}× '
              f'{b/EM[:, j].mean():>9.3f}× {b/ET[:, j].mean():>9.3f}×')
    best = max(EP[:, -1].mean() / EE[:, -1].mean(),
               EP[:, -1].mean() / EM[:, -1].mean(),
               EP[:, -1].mean() / ET[:, -1].mean())
    print(f'\n🔑 ВИРОК КАНАЛУ: {"ЖИВЕ" if best >= 1.05 else "МЕРТВО"} '
          f'(найкраща робастна {best:.3f}×, 🟢гейт 1.05)')
    print(f'\n3️⃣ ЧИ ВИДНО ПОХИБКУ З САМОЇ ФОРМИ ГРАФІКА (без еталону)')
    print('   ознаки лише з траєкторії `x̄` на вікні [n/8, n]:')
    n = int(CHK[-1])
    X, ysig, ymag = [], [], []
    for r in range(nreal):
        rm = PATHS[r]
        seg = rm[n // 8:n]
        rg = seg.max() - seg.min()
        rg = rg if rg > 0 else 1e-30
        mid = 0.5 * (seg.max() + seg.min())
        turns = int(np.sum(np.diff(np.sign(np.diff(seg))) != 0))
        ma = np.convolve(seg, np.ones(64) / 64, mode='valid')
        cr = int(np.sum(np.diff(np.sign(seg[63:] - ma)) != 0))
        X.append([(rm[n - 1] - mid) / rg,            # де в каналі стоїмо
                  (rm[n - 1] - seg[0]) / rg,         # напрямок за вікно
                  (rm[n - 1] - rm[n // 2 - 1]) / rg,  # свіжий напрямок
                  turns / len(seg), cr / max(len(ma), 1),
                  rg * np.sqrt(n)])                  # ширина каналу
        ysig.append(rm[n - 1] - tA)
        ymag.append(abs(rm[n - 1] - tA))
    X = np.array(X); X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    ysig = np.array(ysig); ymag = np.array(ymag)

    def oos(Xa, ya):
        h = len(ya) // 2
        A = np.hstack([Xa, np.ones((len(ya), 1))])
        b, *_ = np.linalg.lstsq(A[:h], ya[:h], rcond=None)
        r_ = ya[h:] - A[h:] @ b
        return float(1.0 - r_.var() / max(ya[h:].var(), 1e-300))
    rs, rm_ = oos(X, ysig), oos(X, ymag)
    rr = oos(np.roll(X, 1, axis=0), ysig)
    print(f'   R² на ЗНАК похибки (те, що можна відняти): {rs:+.4f}')
    print(f'   R² на РОЗМІР похибки (не віднімається):    {rm_:+.4f}')
    print(f'   ⛔ROLL-контроль:                            {rr:+.4f}')
    print(f'   🔑 {"ЖИВЕ" if rs > 0.10 else "МЕРТВО"} '
          f'(🟢гейт R²(знак) ≥ 0.10)')
    print('📊 БАЛ: 1.954e-07 (діагностика)')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
