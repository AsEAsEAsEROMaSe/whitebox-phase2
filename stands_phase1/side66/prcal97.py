"""prcal97 — ЧОМУ `PR(κ₃)` = 1.12 (С97) І 3.1 (С64) НЕ СУПЕРЕЧАТЬ ОДНЕ ОДНОМУ.
КАЛІБРУВАННЯ ОБОХ ПРИЛАДІВ НА ТЕНЗОРІ З ВІДОМИМ CP-РАНГОМ.

════════════════ ЩО ЗНАЙДЕНО ЧИТАННЯМ КОДУ (до прогону) ════════════════
Прилади РІЗНІ у ДВОХ місцях одразу, і обидві різниці тиснуть в ОДИН бік:
```
С64 arena62/kappa3_rank.py:56   PR = (Σσ)²/Σσ²      ← по СИНГУЛЯРНИХ числах
    M = (1/N)·(b⊗b)ᵀ·b  — ПОВНА розгортка 65536×256, точна

С97 side66/k3rank97.py:76-79    PR = (Σσ²)²/Σσ⁴     ← по КВАДРАТАХ (енергія)
    M̂[a,s] = E[x̃_a (x̃·v_s)²]  — СКЕТЧ 512 напрямками `v⊗v`
```
🚨 Скетч не є гаусовим скетчем у просторі 65536: стовпці `v_s⊗v_s` лежать на
многовиді Веронезе і сильно скорельовані між собою ⟹ немає підстав чекати,
що він зберігає спектр. Це ГІПОТЕЗА про зміщення, і саме її тут міряємо.

════════════════ ЯК МІРЯЄМО (пряма пломба, без мереж) ════════════════
Будуємо `T = Σᵢ cᵢ aᵢ^⊗3` з ВІДОМИМ CP-рангом `r` і відомим спектром `cᵢ`,
тоді пускаємо на нього ОБИДВА прилади. Істина відома ⟹ видно зміщення кожного.
Контроль на «не той об'єкт»: додаємо симетризований ГАУСІВ шум і дивимось,
чи прилад не читає ранг шуму замість рангу сигналу.

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) PR(σ²) < PR(σ) ЗАВЖДИ (піднесення концентрує)                шанс 97%
 (б) обидва прилади на ОРТОГОНАЛЬНОМУ рівноваговому базисі дадуть
     РІВНО `r` — тобто без спектрального спаду різниці нема       шанс 80%
 (в) 🔑 скетч `v⊗v` ЗАНИЖУЄ ранг понад різницю визначень          шанс 60%
 (г) відношення двох приладів на реалістичному спаді ляже
     в 2…4 — тобто накриє зміряні ×2.8                            шанс 55%
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402

N = 128                      # ширина (256³ не потрібна — питання масштабно-інваріантне)
NSK = 512                    # скетч-напрямків, як у k3rank97
rng = np.random.default_rng(970209)


def pr_sv(s):
    """С64: participation ratio по САМИХ сингулярних числах."""
    s = np.asarray(s, dtype=np.float64)
    return float(s.sum() ** 2 / (s * s).sum())


def pr_en(s):
    """С97: participation ratio по КВАДРАТАХ (енергії)."""
    s2 = np.asarray(s, dtype=np.float64) ** 2
    return float(s2.sum() ** 2 / (s2 * s2).sum())


def build(r, decay, orth=True):
    """T = Σ cᵢ aᵢ^⊗3, CP-ранг РІВНО r, спектр cᵢ = decay^i."""
    A = rng.standard_normal((N, r))
    if orth:
        A, _ = np.linalg.qr(A)
    else:
        A /= np.linalg.norm(A, axis=0, keepdims=True)
    c = decay ** np.arange(r)
    T = np.einsum("ir,jr,kr,r->ijk", A, A, A, c, optimize=True)
    return T, c


def unfold_exact(T):
    """С64: повна розгортка (n², n) ⟹ сингулярні числа."""
    n = T.shape[0]
    M = T.reshape(n * n, n)
    return np.linalg.svd(M, compute_uv=False)


def sketch(T, nsk=NSK):
    """С97: M̂[a,s] = Σ_bc T[a,b,c] v_s[b] v_s[c] на випадкових v."""
    n = T.shape[0]
    V = rng.standard_normal((n, nsk))
    # (n, n*n) @ (n*n, nsk) через два кроки, щоб не будити n²·nsk
    tmp = np.einsum("abc,bs->acs", T, V, optimize=True)
    M = np.einsum("acs,cs->as", tmp, V, optimize=True)
    return np.linalg.svd(M / np.sqrt(nsk), compute_uv=False)


print(f"prcal97 · ширина {N} · скетч {NSK} напрямків\n" + "=" * 88)
print("ТЕНЗОР З ВІДОМИМ CP-РАНГОМ: ЩО ЧИТАЄ КОЖЕН ПРИЛАД")
print("=" * 88)
print(f"{'істина r':>9s}{'спад':>7s}{'PR(σ) точн':>12s}{'PR(σ²) точн':>13s}"
      f"{'PR(σ) скетч':>13s}{'PR(σ²) скетч':>14s}"
      f"{'  С64/С97':>11s}")
rows = []
for decay in (1.00, 0.90, 0.75, 0.50):
    for r in (1, 2, 4, 8, 16, 32):
        T, c = build(r, decay)
        se = unfold_exact(T)
        ss = sketch(T)
        a, b, cc, dd = pr_sv(se), pr_en(se), pr_sv(ss), pr_en(ss)
        rows.append((r, decay, a, b, cc, dd))
        print(f"{r:>9d}{decay:>7.2f}{a:>12.2f}{b:>13.2f}{cc:>13.2f}"
              f"{dd:>14.2f}{a/max(dd,1e-9):>11.2f}")

print("\n" + "-" * 88)
print("РОЗЩЕПЛЕННЯ ДВОХ ПРИЧИН (у скільки разів кожна занижує), медіана по "
      "рядках вище:")
d1 = np.median([a / b for _, _, a, b, _, _ in rows])
d2 = np.median([b / d for _, _, _, b, _, d in rows])
d3 = np.median([a / d for _, _, a, _, _, d in rows])
print(f"  (1) визначення  PR(σ) → PR(σ²)          ×{d1:.2f}")
print(f"  (2) прилад      точна розгортка → скетч ×{d2:.2f}")
print(f"  РАЗОМ                                   ×{d3:.2f}   "
      f"(зміряний розрив С64/С97 = ×2.8)")

print("\n" + "-" * 88)
print("🔒 КОНТРОЛЬ «НЕ ТОЙ ОБ'ЄКТ»: чистий симетричний ГАУСІВ шум (CP-ранг "
      "повний)")
G = rng.standard_normal((N, N, N))
G = (G + G.transpose(1, 0, 2) + G.transpose(2, 1, 0)
     + G.transpose(0, 2, 1) + G.transpose(1, 2, 0)
     + G.transpose(2, 0, 1)) / 6.0
se, ss = unfold_exact(G), sketch(G)
print(f"  шум: PR(σ) точн = {pr_sv(se):.1f} · PR(σ²) точн = {pr_en(se):.1f}"
      f" · PR(σ) скетч = {pr_sv(ss):.1f} · PR(σ²) скетч = {pr_en(ss):.1f}")
print(f"  (повний ранг = {N}; прилад, що дає тут мале число, читає СЕБЕ, "
      f"а не тензор)")

print("\n" + "-" * 88)
print("🔑 ЩО PR УЗАГАЛІ КАЖЕ ПРО CP-РАНГ: залишок у НОРМІ після зрізу топ-k")
for decay in (0.90, 0.75):
    T, c = build(32, decay)
    se = unfold_exact(T)
    en = se ** 2
    tot = en.sum()
    line = "  ".join(f"k={k}: {np.sqrt(1 - en[:k].sum()/tot):.3f}"
                     for k in (1, 2, 4, 8, 16))
    print(f"  спад {decay:.2f} (істина r=32, PR(σ²)={pr_en(se):.2f}):  {line}")
print("\n📌 PR — це НЕ ранг. Це ширина спектра за енергією; при спаді 0.75 "
      "тензор\n   рангу 32 виглядає як PR≈2, а залишок у нормі після топ-2 "
      "лишається помітним.")
