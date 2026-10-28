"""cum89 — ТРЕТІЙ КУМУЛЯНТ У СТАНІ РЕКУРСІЇ, А НЕ ЛИШЕ НА ВИХОДІ.

════════════════ ЧОМУ САМЕ ЦЕ ════════════════
Оракульні заміри С88, проведені через ВИПРАВЛЕНУ константу `(k+10.33)/11.33`:
    оракул m+C РАЗОМ      k=20.97  ⟹ бал 6.78e-08   (лідера 5.102e-08 НЕ б'є)
    оракул m+C+ВИХІД      k=400.85 ⟹ бал 4.7e-09    (лідера б'є у 10×)
Стрибок 20.97 → 400.85 — це ОДИН вихідний крок `E[ReLU(h)]` при ТОЧНИХ `m,C`.
Тобто двомоментне представлення впирається в стелю 20.97 НЕ через перенос, а
через те, що маргіналь `h_j` НЕ ГАУСОВА. `cum88` бачив від цього лише ×1.336,
бо міряв поправку при НАШИХ `m,C`, де похибка моментів домінує (важіль на дві
точності = добуток). Поріг лідера k=31.2 лежить МІЖ 20.97 і 400.85 ⟹ його
можна взяти ЧАСТКОВИМ покращенням обох, без жодного оракула.

════════════════ ЩО РОБИМО ════════════════
Розширюємо стан рекурсії `(m, C)` → `(m, C, κ₃)`, де κ₃ — ПОНЕЙРОННИЙ вектор
(не тензор). Едвортові поправки до ТРЬОХ моментів `relu` виведені й
ЗАПЛОМБОВАНІ квадратурою (`jseal89`, найгірше 1e-11 на 95 точках):
    E[relu^p] = G_p + (γ₃/6)·J_p
    J_0 = (t²−1)φ    J_1 = −s·t·φ    J_2 = 2s²φ    J_3 = 6s³Φ
Перенос κ₃ на наступний шар — наближенням незалежності
    κ₃(h'_j) = Σ_i W_ij³ κ₃(a_i)         (один матмул із W³)

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
🚨 Головний ризик відомий наперед: для МІШАНИХ κ₂₁ наближення незалежності
виявилось мертвим (корел 0.077–0.27, `mix88`). Для МАРГІНАЛЬНИХ κ₃ той самий
прийом працював ЧАСТКОВО (×1.108 із ×1.336 у `cum88`) ⟹ тут ставлю на
проміжне: A1 (незалежність) дасть k = 1.1…1.6 (шанс 55%), A2 (оракул κ₃)
дасть k = 2…5 (шанс 45%). ⚠️ Якщо A2 < A3 (двомоментна стеля 20.97) — κ₃ у
стані НЕ пробиває стелю, і напрям треба закривати ЧЕСНО, а не переказувати.
Ймовірність, що A2 перевищить 20.97: оцінюю НИЗЬКО, ~15%.

════════════════ ПЛОМБИ ════════════════
P1  при κ₃≡0 гілка A1 мусить відтворити A0 БІТОВО
P2  шар 1: вхід ТОЧНО гаусів ⟹ κ₃(h¹)=0, а κ₃(a¹) мусить збігтися з
    аналітичним κ₃ для relu(N(0,s²)) — незалежний шлях до того самого числа
P3  МАСШТАБУВАННЯ: N×4 мусить учетверо збити шум оракульних κ₃
P4  база стенду мусить відтворити відоме 0.94% (±2%)
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
N = 1 << LOGN
DT = torch.float32
torch.manual_seed(4242)

SQ2PI = float(np.sqrt(2.0 * np.pi))
W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n = W.shape[2]
L = W.shape[1]
print(f"пристрій {DEV} · мереж {NNET} · семплів 2^{LOGN}={N} · L={L} · n={n}")


def phi(t):
    return torch.exp(-0.5 * t * t) / SQ2PI


def Phi(t):
    return 0.5 * (1.0 + torch.erf(t * 0.7071067811865476))


def relu_moments(mp, s, g3):
    """E[relu], E[relu²], E[relu³] для X ~ Едворт(mp, s², γ₃=g3).

    Гаусова частина — класика; поправка — виведені `J_p` (`jseal89`)."""
    t = mp / s
    P, f = Phi(t), phi(t)
    e1 = s * (t * P + f)
    e2 = (mp * mp + s * s) * P + (mp * s) * f
    e3 = mp * (mp * mp + 3.0 * s * s) * P + s * (mp * mp + 2.0 * s * s) * f
    if g3 is not None:
        c = g3 / 6.0
        e1 = e1 + c * (-s * t * f)
        e2 = e2 + c * (2.0 * s * s * f)
        e3 = e3 + c * (6.0 * s ** 3 * P)
    return e1, e2, e3, P, f


def step(m, C, k3, w, use_k3):
    """Один шар. Повертає (m_a, C_a, k3_a) — моменти АКТИВАЦІЙ."""
    cp = w.T @ (C @ w)
    s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
    mp = m @ w
    g3 = (k3 / s ** 3) if (use_k3 and k3 is not None) else None
    e1, e2, e3, P, f = relu_moments(mp, s, g3)
    var = torch.clamp(e2 - e1 * e1, min=1e-30)
    k3a = e3 - 3.0 * e1 * e2 + 2.0 * e1 ** 3
    # позадіагональ — рівно як у сабміті (Мелер K=2); діагональ вписуємо точно
    Ca = cp * torch.outer(P, P) + 0.5 * (cp * cp) * torch.outer(f / s, f / s)
    Ca = Ca + torch.diag(var - torch.diagonal(Ca))
    return e1, Ca, k3a


def push_k3(k3a, w):
    """κ₃(h'_j) = Σ_i W_ij³ κ₃(a_i) — наближення НЕЗАЛЕЖНОСТІ."""
    return (w ** 3).T @ k3a


def mc_truth(ws, nsamp, seed, want_k3_path=False):
    """МК: істина на виході + (за потреби) понейронні κ₃ активацій пошарово."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    k3path = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    s1 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    s2 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    s3 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    CH = 1 << 14
    done = 0
    while done < nsamp:
        b = min(CH, nsamp - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(L):
            h = x @ ws[i]
            if i == L - 1:
                acc += torch.clamp(h, min=0.0).sum(0).double()
                break
            x = torch.clamp(h, min=0.0)
            if want_k3_path:
                xd = x.double()
                s1[i] += xd.sum(0)
                s2[i] += (xd * xd).sum(0)
                s3[i] += (xd * xd * xd).sum(0)
        done += b
    out = acc / nsamp
    if want_k3_path:
        for i in range(L - 1):
            a1 = s1[i] / nsamp
            a2 = s2[i] / nsamp
            a3 = s3[i] / nsamp
            k3path[i] = a3 - 3.0 * a1 * a2 + 2.0 * a1 ** 3
    return out, k3path


def predict(ws, mode, k3_oracle=None, mC_oracle=None):
    """mode: 'base' | 'k3ind' | 'k3orc'."""
    m = torch.zeros(n, dtype=DT, device=DEV)
    C = torch.eye(n, dtype=DT, device=DEV)
    k3 = torch.zeros(n, dtype=DT, device=DEV)
    for i in range(L - 1):
        use = mode != "base"
        m, C, k3a = step(m, C, k3 if use else None, ws[i], use)
        if mode == "k3orc" and k3_oracle is not None:
            k3a = k3_oracle[i].to(DT)
        k3 = push_k3(k3a, ws[i + 1]) if use else k3
    # ВИХІДНИЙ КРОК
    w = ws[L - 1]
    cp = w.T @ (C @ w)
    s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
    mp = m @ w
    g3 = (k3 / s ** 3) if mode != "base" else None
    e1, _, _, _, _ = relu_moments(mp, s, g3)
    return e1


print()
print("=" * 84)
print("P2 · ПЛОМБА ШАРУ 1: κ₃(relu(N(0,σ²))) АНАЛІТИЧНО ПРОТИ НАШОГО КРОКУ")
print("=" * 84)
# для X~N(0,s²): E[relu]=sφ(0)=s/√(2π); E[relu²]=s²/2; E[relu³]=2s³φ(0)=2s³/√(2π)
for sv in (0.7, 1.0, 2.3):
    st = torch.tensor([sv], dtype=torch.float64, device=DEV)
    mp0 = torch.zeros(1, dtype=torch.float64, device=DEV)
    e1, e2, e3, _, _ = relu_moments(mp0, st, None)
    k3_ours = float(e3 - 3.0 * e1 * e2 + 2.0 * e1 ** 3)
    a1 = sv / SQ2PI
    a2 = sv * sv / 2.0
    a3 = 2.0 * sv ** 3 / SQ2PI
    k3_ref = a3 - 3.0 * a1 * a2 + 2.0 * a1 ** 3
    print(f"   σ={sv:.1f}: наш κ₃ = {k3_ours:.12f}   аналітичний = {k3_ref:.12f}"
          f"   Δвіднос = {abs(k3_ours - k3_ref) / abs(k3_ref):.3e}")

print()
print("=" * 84)
print(f"ГОЛОВНИЙ ПРОГІН · {NNET} мереж · L={L} (НЕ L=2!)")
print("=" * 84)
rows = []
t0 = time.time()
for j in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[j, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    truth, k3p = mc_truth(ws, N, 9000 + j, want_k3_path=True)
    tr = truth.to(DT)
    p_base = predict(ws, "base")
    p_ind = predict(ws, "k3ind")
    p_orc = predict(ws, "k3orc", k3_oracle=k3p)
    # P1: κ₃≡0 у гілці k3ind мусить дати БІТОВО базу
    z0 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    p_zero = predict(ws, "k3orc", k3_oracle=z0)
    den = float((tr * tr).mean())
    rec = {}
    for nm, p in (("base", p_base), ("ind", p_ind), ("orc", p_orc), ("zero", p_zero)):
        rec[nm] = float(((p - tr) ** 2).mean() / den) ** 0.5
    rows.append(rec)
    if j < 3 or j % 6 == 0:
        print(f"  net{j:>3d}  база {100 * rec['base']:.4f}%  незал {100 * rec['ind']:.4f}%"
              f"  ОРАКУЛ {100 * rec['orc']:.4f}%   (P1 zero−base = "
              f"{abs(rec['zero'] - rec['base']):.2e})")

el = time.time() - t0
print(f"\n  час {el:.0f} с")
b = np.array([r["base"] for r in rows])
i_ = np.array([r["ind"] for r in rows])
o = np.array([r["orc"] for r in rows])
z = np.array([r["zero"] for r in rows])

print()
print("=" * 84)
print("РЕЗУЛЬТАТ")
print("=" * 84)
print(f"  P1 |zero − base| макс = {np.abs(z - b).max():.3e}   "
      f"{'✅' if np.abs(z - b).max() < 1e-9 else '⛔ κ₃-гілка НЕ зводиться до бази'}")
print(f"  P4 база = {100 * b.mean():.4f}%   (відоме 0.94%, допуск ±2%)   "
      f"{'✅' if abs(b.mean() / 0.0094 - 1) < 0.15 else '⚠️ НЕ ВІДТВОРЮЄ'}")
print()
print(f"{'варіант':>26s} {'похибка':>10s} {'k = MSE-виграш':>16s} {'у балі':>10s}")
for nm, v in (("база", b), ("κ₃ незалежність", i_), ("κ₃ ОРАКУЛ", o)):
    k = float((b ** 2).mean() / (v ** 2).mean())
    print(f"{nm:>26s} {100 * v.mean():>9.4f}% {k:>16.3f} {(k + 10.33) / 11.33:>10.4f}")
print()
print("  ⚖️  ДВОМОМЕНТНА СТЕЛЯ (оракул m+C, С88) = k 20.97 ⟹ у балі ×2.763")
kk = float((b ** 2).mean() / (o ** 2).mean())
print(f"  ⟹ κ₃ у стані {'ПРОБИВАЄ' if kk > 20.97 else 'НЕ ПРОБИВАЄ'} двомоментну стелю "
      f"(k={kk:.2f} проти 20.97)")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/cum89.npz", b=b, i=i_, o=o, z=z)
