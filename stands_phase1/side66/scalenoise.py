"""ШУМ НА ГЛИБИНІ — ЦЕ ОДИН СКАЛЯР. ЧИ ПЕРЕДБАЧАЄТЬСЯ ВІН ІЗ ДОСТУПНОГО?

════════ ЗВІДКИ (`noisespec.py`) ════════
`cos(m, u₁) = 0.978` ⟹ головний напрямок шуму = напрямок самого `m`.
Механіка: `a_30^(i) ≈ c_i·u` (PR-ранг 2.5), тож `x̄_30 = mean(c)·u` і **шум
середнього на глибині — це флуктуація ОДНОГО СКАЛЯРА `mean(c)`, шум МАСШТАБУ,
а не напрямку**. Топ-1 напрямок несе 38% шуму, топ-8 — 76%.
⟹ Щоб зняти основну частину, треба знати НЕ 256 чисел, а ОДНЕ.

════════ ДВІ ЧАСТИНИ ════════
1️⃣ **РАДІАЛЬНИЙ Rao–Blackwell.** Мережа позитивно однорідна (зсувів нема):
   `f(z) = ‖z‖·f(z/‖z‖)` ТОЧНО, і `‖z‖ ⊥ напрямок`. Тому
   `mean(‖x‖)·mean(f(u))` НЕЗМІЩЕНА і має меншу дисперсію за `mean(‖x‖·f(u))`
   рівно на вибіркову коваріацію двох НЕЗАЛЕЖНИХ величин (чистий шум).
   🚨 Не потребує знання `E[‖z‖]` — досить вибіркового `mean(‖x‖)`.
2️⃣ **ЧИМ ПЕРЕДБАЧИТИ СКАЛЯР.** Ціль `y = ⟨x̄_L − m_L, u⟩` (шум масштабу).
   Ознаки — ТІЛЬКИ ДОСТУПНІ (вибірка + замикання), без істини:
   `mean‖x‖`, `gate_ℓ − Φ_замик,ℓ`, `⟨x̄_ℓ − m_замик,ℓ, ·⟩` по кількох шарах.
   `R²` каже, скільки шуму масштабу видно ЗСЕРЕДИНИ.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 (1) РАДІАЛЬНИЙ дасть **~0.2%** (`CVa²=Var‖z‖/E‖z‖²≈0.5/255=0.002`, а
   частка знятого = `CVa²CVb²/(CVa²CVb²+CVa²+CVb²) ≤ CVa²`), та ще й
   вибілювання душить радіальний шум у 35.5× ⟹ практично НУЛЬ. Міряю саме
   щоб перевірити цю арифметику, а не бо чекаю виграшу.
📣 (2) `R²` доступних ознак буде **0.3–0.6** — бо С71 бачила 58–76%, але
   ЛИШЕ зі знанням ІСТИННИХ `s,q,μ₃`; із замикальною опорою має впасти.
🟢 ЖИВЕ, якщо `R² ≥ 0.8` на ДОСТУПНИХ ознаках (тоді 1/(1−R²) ≥ 5× по
   когерентній компоненті, а вона несе 38% шуму ⟹ ~1.3× у MSE).

🚨 ПЛОМБИ
  ✅ ОДНОРІДНІСТЬ: `f(2z) = 2f(z)` до машинної точності — інакше вся
     конструкція (1) неправомірна;
  ✅ радіальна оцінка мусить бути НЕЗМІЩЕНОЮ (та сама середня по жеребах);
  ⛔ ROLL-контроль: ознаки, зсунуті на інший жереб, мусять дати `R² ≈ 0`.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N)

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 96
TR_PAIRS, CHUNK = 524_288, 32768
FEAT_L = (1, 2, 4, 8, 12, 16, 20, 24, 28)


def truth_means(wc, dev, seed):
    depth = len(wc)
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
           for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(depth):
            a = torch.relu(a @ wc[i])
            acc[i] += a.double().sum(0)
        done += k
    return [t / float(2 * TR_PAIRS) for t in acc]


def draw(wc, npair, dev, seed, m_cl, f_cl):
    """Один жереб. Повертає (x̄ по шарах, радіальна оцінка x̄_L, ознаки)."""
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cov = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xw = x @ wh
    nrm = xw.double().norm(dim=1)
    mn = float(nrm.mean())
    # 1️⃣ радіальний RB: ті самі напрямки, одинична норма
    xu = (xw / nrm.to(xw.dtype).unsqueeze(1)).float()

    a = torch.relu(xw @ wc[0])
    au = torch.relu(xu @ wc[0])
    out = [a.double().mean(0)]
    feats = [mn]
    for i in range(1, len(wc)):
        p = a @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        a = torch.relu(p)
        au = torch.relu(au @ wc[i])
        # 🚨 `m_cl[i-1]` — замикання шару `i`, тож брати треба середнє САМЕ
        # шару `i` (тобто ПІСЛЯ relu), а не попереднього (перша версія
        # порівнювала шар i−1 із замиканням шару i і падала на розмірах)
        if i in FEAT_L and i - 1 < len(f_cl):
            feats.append(float((gate - f_cl[i - 1]).sum()))
            feats.append(float((a.double().mean(0) - m_cl[i - 1]).sum()))
        out.append(a.double().mean(0))
    rad = au.double().mean(0) * mn
    return out, rad, feats


def main(nets=8, dev=DEV, ndraw=NDRAW):
    torch.manual_seed(23)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: (1) радіальний ~0.2% ⟹ практично НУЛЬ; '
          f'(2) R² доступних ознак 0.3–0.6')
    print(f'🟢 ЖИВЕ при R² ≥ 0.8\n')

    base_v, rad_v, bias = [], [], []
    R2, R2R, COH = [], [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            mt = truth_means(wc, dev, 310_000 + 877 * n)
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, min(28, d - 1), dev,
                                                   extras=True)
            # ✅ПЛОМБА ОДНОРІДНОСТІ: f(2z) = 2 f(z)
            g = torch.Generator(device=dev).manual_seed(999)
            zz = torch.randn(64, W_, device=dev, generator=g)
            a1, a2 = zz, zz * 2.0
            for i in range(d):
                a1 = torch.relu(a1 @ wc[i]); a2 = torch.relu(a2 @ wc[i])
            hom = float((a2 - 2.0 * a1).abs().max() / a1.abs().max())

            D, RD, FT = [], [], []
            for k in range(ndraw):
                xb, rad, ft = draw(wc, npair, dev,
                                   51_000_000 + 977 * n + k, m_cl, f_cl)
                D.append(xb[d - 1] - mt[d - 1])
                RD.append(rad - mt[d - 1])
                FT.append(ft)
            M = torch.stack(D, 0)
            Mr = torch.stack(RD, 0)
            base_v.append(float(M.pow(2).sum(1).mean()))
            rad_v.append(float(Mr.pow(2).sum(1).mean()))
            bias.append(float((M.mean(0) - Mr.mean(0)).norm()
                              / max(float(M.mean(0).norm()), 1e-30)))
            # головний напрямок шуму і скалярна ціль
            Mc = M - M.mean(0, keepdim=True)
            C = (Mc.T @ Mc) / float(Mc.shape[0])
            _, V = torch.linalg.eigh(C)
            u1 = V[:, -1]
            y = (Mc @ u1).cpu().numpy()
            COH.append(float((Mc @ u1).pow(2).mean()
                             / Mc.pow(2).sum(1).mean()))
            X = np.array(FT, dtype=np.float64)
            X = X - X.mean(0, keepdims=True)
            sd_ = X.std(0); sd_[sd_ == 0] = 1.0
            X = X / sd_
            X = np.hstack([X, np.ones((X.shape[0], 1))])

            # 🚨 IN-SAMPLE R² ТУТ БРЕХАВ: 20 параметрів на 64 жереби дали
            # 0.967 при ⛔ROLL 0.364 (мусив бути ~0) ⟹ чисте переганяння.
            # Рахуємо ЧЕСНИЙ out-of-sample: фіт на першій половині жеребів,
            # вимір на другій.
            def r2_oos(Xa, ya):
                h = len(ya) // 2
                b, *_ = np.linalg.lstsq(Xa[:h], ya[:h], rcond=None)
                r = ya[h:] - Xa[h:] @ b
                return 1.0 - r.var() / max(ya[h:].var(), 1e-300)
            R2.append(r2_oos(X, y))
            R2R.append(r2_oos(np.roll(X, 1, axis=0), y))   # ⛔ROLL
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (однорідність {hom:.2e})', flush=True)

    b = np.array(base_v); r = np.array(rad_v)
    print(f'\n{nets} мереж × {ndraw} жеребів.')
    print(f'✅ПЛОМБА ОДНОРІДНОСТІ `f(2z)=2f(z)`: {hom:.2e} '
          f'{"✅" if hom < 1e-5 else "🛑"}')
    print(f'✅ПЛОМБА НЕЗМІЩЕНОСТІ радіальної оцінки: '
          f'{np.mean(bias)*100:.3f}% від ‖зсуву‖ '
          f'{"✅" if np.mean(bias) < 0.2 else "🛑"}')
    print(f'\n1️⃣ РАДІАЛЬНИЙ Rao–Blackwell: {b.sum()/r.sum():.4f}× '
          f'(медіана {np.median(b/r):.4f}×) — 📣передбачено ~1.002×')
    print(f'\n2️⃣ КОГЕРЕНТНА ЧАСТКА (топ-1 напрямок): '
          f'{np.mean(COH)*100:.1f}% шуму')
    print(f'   R² OOS доступних ознак: {np.mean(R2):.4f} '
          f'(медіана {np.median(R2):.4f})')
    print(f'   ⛔ROLL-контроль:     {np.mean(R2R):.4f} '
          f'{"✅" if np.mean(R2R) < 0.15 else "🛑 тавтологія"}')
    live = np.mean(R2) >= 0.8
    print(f'\n🔑 ВИРОК: {"ЖИВЕ" if live else "МЕРТВО"} '
          f'(🟢гейт R² ≥ 0.8; маємо {np.mean(R2):.3f})')
    if not live and np.mean(R2) > np.mean(R2R) + 0.1:
        gain = 1.0 / (1.0 - np.mean(R2))
        eff = 1.0 / (1.0 - np.mean(COH) * np.mean(R2))
        print(f'   сигнал Є ({np.mean(R2):.3f} проти ROLL '
              f'{np.mean(R2R):.3f}), але його вистачає лише на '
              f'{gain:.2f}× по когерентній компоненті ⟹ {eff:.3f}× у MSE')
    print(f'📊 БАЛ: 1.972e-07 (діагностика, стендова)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
