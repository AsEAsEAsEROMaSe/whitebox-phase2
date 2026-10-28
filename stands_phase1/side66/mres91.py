"""mres91 — `R²` ПОНЕЙРОННИХ ОЗНАК НА ЗАЛИШКУ `m`. Число, якого нема ні в кого.

════════════════ ЗВІДКИ ПИТАННЯ ════════════════
АРХІВ (RAUND 9 §4) вивів гейт форми, і я його перевірив: замикання строго
позитивно-1-однорідне ⟹ `m = σ·F(t)` поглинається `_ALPHA` ТОТОЖНО для
будь-якої `F`. ⟹ **виживає лише те, що міняє відношення `α/β` у
`m = σ[αΦ(t) + βφ(t)]` НЕОДНАКОВО для різних нейронів ОДНОГО шару.**
Форма, що проходить гейт:
```
m_a = mp·Φ(t_a) + σ_a·φ(t_a)·(1 + g(x_a)),   x_a — понейронні ознаки, НЕ f(t)
```
Ознаки `off`, `a3`, `s4` рахуються з наявних `W` і `diag(C)`, ціна ~3n²/шар
= 0.0007% B. Їхній `R²` міряли на залишку **σ** (0.203). На залишку **m** —
НІКОЛИ. Гілка стоїть рівно на цьому числі.

════════════════ МОЯ ПОПРАВКА ДО ЙОГО СПИСКУ ОЗНАК ════════════════
🚨 Закон парності (С87): `Var(p)` парна за вагами ⟹ знак скорочується
(corr 0.0062); `κ₃` НЕПАРНА ⟹ знак вирішує (93.6%).
Залишок `m` веде АСИМЕТРІЯ маргіналі `h_a` (Еджворт: `−(γ₃/6)tφσ`), а `γ₃`
НЕПАРНА за `W`. ⟹ **`a3_модуль = Σ|W|³v^{3/2}` парна і НЕ МОЖЕ мати сигналу
за побудовою.** Треба ЗНАКОВА:
```
g3 = Σ_j W_ja³·κ₃(a_j) / σ_a³        ← провідний член κ₃(h_a) при незалежних a
```
Це і є передбачення Еджворта; тому міряю ще й його з коефіцієнтом `−t/6`.

════════════════ ЩО МІРЯЄТЬСЯ ════════════════
Ціль: `g_a = (m_істинне − m_замикання) / (σ_a·φ(t_a))` — рівно та ручка, що
проходить гейт. Вхід замикання — ІСТИННІ `(m_h, Cov_h)` ⟹ це ОДНОКРОКОВИЙ
залишок, без накопичення.
`R²` рахується **LOO по мережах** (одиниця незалежності — мережа, не нейрон).

════════════════ ПЛОМБИ ════════════════
P1 ознака `t` сама по собі: `R²` мусить бути ВИСОКИЙ у сирій регресії, але
   вона поглинається `_ALPHA` ⟹ це контроль «не той об'єкт», не виграш.
P2 `a3_модуль` мусить дати `R² ≈ 0` (парність). Якщо дасть багато —
   мій закон парності хибний, і це важливіше за все інше в стенді.
P3 Еджвортове передбачення `g_pred = −(γ₃/6)·t` з ОРАКУЛЬНОЮ `γ₃(h_a)`:
   це ВЕРХНЯ межа для будь-якої ознаки, що працює через асиметрію.
P4 ⛔ЧУЖА: ознаки з ПЕРЕМІШАНИХ нейронів ⟹ `R²` мусить впасти до ~0.
P5 масштабування по `N`: шум цілі падає, `R²` справжніх ознак стоїть.

════════════════ ПЕРЕКЛАД `R²` У БАЛ (щоб не міряти стелю) ════════════════
`k = 1/(1−R²)` · `виграш = (k+10.33)/11.33` (калібровка відтворює рекорд 0.08%)
```
R² 0.50 → k=2.0  → ×1.088      R² 0.90 → k=10  → ×1.79
R² 0.968→ k=31.2 → ×3.67 ЛІДЕР R² 0.99 → k=100 → ×9.75
```

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) `a3_модуль` дасть R² < 0.02 (парність)                    шанс 80%
 (б) знакова `g3` дасть R² 0.3–0.7                             шанс 50%
 (в) оракульна Еджвортова межа (P3) дасть R² > 0.8             шанс 45%
 (г) жодна комбінація не дійде до 0.968                        шанс 85%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 32
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i]
            got += 1


NAMES = ('t', 'off', 'a3mod', 's4', 'g3зн', 'едж_ор', 'g3_замик')
FEAT, TGT, NET, LAY = [], [], [], []
t0 = time.time()
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])).double()
              for i in range(32)]
        g = torch.Generator().manual_seed(240000 + 19 * q)
        x = torch.randn(NS, 256, generator=g)
        aprev = None
        for k in range(32):
            h = (x @ ws[k].float()).double()
            mh = h.mean(0)
            hc = h - mh
            v = (hc * hc).mean(0)
            s = v.clamp(min=1e-30).sqrt()
            t = mh / s
            p = 0.5 * (1.0 + torch.erf(t / SQ2))
            f = torch.exp(-0.5 * t * t) * INV_SQ2PI
            a = h.clamp(min=0.0)
            m_true = a.mean(0)
            m_pred = mh * p + s * f                      # замикання, точний вхід
            if aprev is not None:
                W = ws[k]
                ac = aprev - aprev.mean(0)
                vj = (ac * ac).mean(0)                    # diag Cov(a_prev)
                k3j = (ac * ac * ac).mean(0)              # κ₃(a_j), оракульна
                W2, W3, W4 = W * W, W * W * W, (W * W) ** 2
                dia = W2.T @ vj                           # діагональний внесок
                off = 1.0 - dia / v.clamp(min=1e-30)      # частка позадіаг.
                a3m = (W3.abs().T @ vj.pow(1.5)) / s.pow(3)
                s4 = (W4.T @ (vj * vj)) / s.pow(4)
                g3s = (W3.T @ k3j) / s.pow(3)             # ЗНАКОВА
                # ⛔P3 оракульна асиметрія самої h_a
                g3o = (hc.pow(3).mean(0)) / s.pow(3)
                tgt = (m_true - m_pred) / (s * f).clamp(min=1e-300)
                FEAT.append(torch.stack([t, off, a3m, s4, g3s,
                                         -(g3o / 6.0) * t, -(g3s / 6.0) * t], 1))
                TGT.append(tgt)
                NET.append(torch.full((256,), float(q)))
                LAY.append(torch.full((256,), float(k)))
            aprev = a
            x = a.float()
        if q % 8 == 0:
            print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

X = torch.cat(FEAT).numpy()
Y = torch.cat(TGT).numpy()
NN = torch.cat(NET).numpy()
LL = torch.cat(LAY).numpy()
ok = np.isfinite(Y) & np.isfinite(X).all(1)
X, Y, NN, LL = X[ok], Y[ok], NN[ok], LL[ok]
print(f"\n  час {time.time()-t0:.0f} с · {len(Y)} точок · {NNET} мереж · N={NS}\n")


def loo_r2(cols, Xa=None, Ya=None, Na=None):
    """R² з LOO по МЕРЕЖАХ (одиниця незалежності — мережа)."""
    Xa = X if Xa is None else Xa
    Ya = Y if Ya is None else Ya
    Na = NN if Na is None else Na
    A = np.concatenate([Xa[:, cols], np.ones((len(Ya), 1))], 1)
    pred = np.zeros_like(Ya)
    for q in np.unique(Na):
        tr, te = Na != q, Na == q
        b, *_ = np.linalg.lstsq(A[tr], Ya[tr], rcond=None)
        pred[te] = A[te] @ b
    ss = ((Ya - Ya.mean()) ** 2).sum()
    return 1.0 - ((Ya - pred) ** 2).sum() / ss


print("=" * 88)
print(f"R² НА ЗАЛИШКУ `m` (LOO по мережах) · ціль g = (m_іст − m_замик)/(σφ)")
print("=" * 88)
print(f"{'ознака':>12s}{'R² соло':>12s}{'⛔чужа':>12s}{'k':>10s}{'бал ×':>10s}")
rng = np.random.default_rng(7)
perm = rng.permutation(len(Y))
for i, nm in enumerate(NAMES):
    r = loo_r2([i])
    Xf = X.copy()
    Xf[:, i] = X[perm, i]
    rf = loo_r2([i], Xa=Xf)
    k = 1.0 / max(1e-9, 1.0 - max(r, 0.0))
    print(f"{nm:>12s}{r:>12.4f}{rf:>12.4f}{min(k,1e6):>10.2f}"
          f"{(k+10.33)/11.33:>10.3f}")

for label, cols in (('off+a3+s4 (його трійка)', [1, 2, 3]),
                    ('+ знакова g3', [1, 2, 3, 4]),
                    ('усе без оракула', [0, 1, 2, 3, 4, 6]),
                    ('усе + оракул (СТЕЛЯ)', [0, 1, 2, 3, 4, 5, 6])):
    r = loo_r2(cols)
    k = 1.0 / max(1e-9, 1.0 - max(r, 0.0))
    print(f"{label:>24s}  R²={r:>7.4f}  k={min(k,1e6):>8.2f}  бал ×{(k+10.33)/11.33:.3f}")

print("\n" + "=" * 88)
print("ПО ГЛИБИНІ (гроші живуть у глибині — С88)")
print("=" * 88)
print(f"{'шари':>12s}{'R²(його трійка)':>18s}{'R²(+g3зн)':>14s}{'R²(стеля)':>14s}")
for lo, hi in ((1, 8), (9, 16), (17, 24), (25, 31)):
    sel = (LL >= lo) & (LL <= hi)
    a = loo_r2([1, 2, 3], X[sel], Y[sel], NN[sel])
    b = loo_r2([1, 2, 3, 4], X[sel], Y[sel], NN[sel])
    c = loo_r2([0, 1, 2, 3, 4, 5, 6], X[sel], Y[sel], NN[sel])
    print(f"{f'{lo}..{hi}':>12s}{a:>18.4f}{b:>14.4f}{c:>14.4f}")
print(f"\n⛔P2 a3_модуль (парна) R² = {loo_r2([2]):.4f} "
      f"{'✅ закон парності тримає' if loo_r2([2]) < 0.05 else '🚨 ЗАКОН ПАРНОСТІ ХИБНИЙ'}")
print(f"📣 ПОРІГ ЛІДЕРА: потрібно R² = 0.968")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/mres91.npz",
         X=X, Y=Y, NET=NN, LAY=LL)
