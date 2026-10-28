#!/usr/bin/env python3
"""ПАКУВАННЯ ЧЕРЕЗ БЛОКОВЕ КВАНТУВАННЯ (С113) — ідея ЮЗЕРА після iq-схем.

════════ НАВІЩО ════════
Тариф flopscope — НА ЕЛЕМЕНТ (🔒зміряно на 0.12.0: matmul = 2·M·K·N, float64 ×1.999,
елементні 0.5 флопа/елемент у float32). ⟹ два семпли в одному елементі = півціни.
Наш бал = a·p/B, і ми програємо ВИКЛЮЧНО ціною семпла `p` ⟹ ×2 тут = ×2 у балі
за дисперсією.

════════ ЧОМУ САМЕ КВАНТУВАННЯ ════════
З одного неперервного числа два неперервні не дістати. Пакування `u = q_a + S·q_b`
працює лише для ЦІЛИХ ⟹ квантувати треба і активації, і ваги (як iq/Q8_0).
Після матмулу суми лишаються цілими, і float32 тримає їх ТОЧНО до 2²⁴.
🚨 ТАРИФ ЛІНІЙНИЙ ЗА БІТАМИ ВИЩЕ 32: float64 дає 2× бітів за 2× ціни ⟹ НУЛЬ.
   Уся вигода можлива лише всередині 24 бітів float32 ⟹ по 12 бітів на поле.
Бюджет:  2·log₂(K_блоку · Qa_max · Qw_max) ≤ 24

════════ ЩО МІРЯЄМО ════════
Пакування САМЕ ПО СОБІ похибки не додає (цілі точні). Уся похибка — від
квантування входів. Тож міряємо РІВНО її: поблокове квантування активацій
(масштаб на (семпл, блок)) і ваг (масштаб на (блок, вихід)) — масштаби
виносяться з блокової суми ТОЧНО.

🔒 ПЛОМБИ
S1 бюджет бітів: K·Qa·Qw ≤ 2¹² для кожної конфігурації (інакше поля злипнуться)
S2 Qa=Qw=∞ (без квантування) ⟹ похибка РІВНО нуль
S3 масштабування N: похибка квантування НЕ падає з N (це зсув, не шум)
S4 контроль: та сама кількість бітів, але БЕЗ поблокових масштабів ⟹ мусить
   бути істотно гіршою (інакше блоки ні до чого)
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nsamp", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=6113113)
    a = ap.parse_args()
    import cupy as cp
    print(f"cupy {cp.__version__} · {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}",
          flush=True)
    n, D, N, CH = a.width, a.depth, a.nsamp, a.chunk
    rng = cp.random.default_rng(a.seed)
    s = float(np.sqrt(2.0 / n))
    Ws = [(rng.standard_normal((n, n), dtype=cp.float32) * s) for _ in range(D)]

    def qmm(h, W, Qa, Qw, Kb, blockwise=True):
        """поблоково квантований матмул: h[B,n] @ W[n,n]"""
        B = h.shape[0]; nb = n // Kb
        hb = h.reshape(B, nb, Kb)
        Wb = W.reshape(nb, Kb, n)
        if blockwise:
            sa = cp.abs(hb).max(axis=2, keepdims=True)
            sw = cp.abs(Wb).max(axis=1, keepdims=True)
        else:                                   # 🔒S4: один масштаб на все
            sa = cp.abs(h).max() * cp.ones((B, nb, 1), cp.float32)
            sw = cp.abs(W).max() * cp.ones((nb, 1, n), cp.float32)
        sa = cp.maximum(sa, 1e-30); sw = cp.maximum(sw, 1e-30)
        qa = cp.rint(hb / sa * Qa)
        qw = cp.rint(Wb / sw * Qw)
        # накопичуємо поблоково ⟹ пам'ять O(B·n), а не O(B·nb·n)
        out = cp.zeros((B, n), cp.float32)
        sa2 = sa[:, :, 0]; sw2 = sw[:, 0, :]
        for i in range(nb):
            out += (qa[:, i, :] @ qw[i]) * sa2[:, i:i + 1] * sw2[i][None, :]
        return out / (Qa * Qw)

    def fwd(gen, mode, Qa=0, Qw=0, Kb=0, blockwise=True):
        acc = cp.zeros(n, cp.float64); done = 0
        while done < N:
            b = min(CH, N - done)
            h = gen(b)
            for l in range(D):
                p = h @ Ws[l] if mode == "exact" else qmm(h, Ws[l], Qa, Qw, Kb, blockwise)
                h = cp.maximum(p, 0.0)
            acc += h.sum(0, dtype=cp.float64); done += b
        return acc / N

    def cos_err(x, y):
        x = x.astype(cp.float64); y = y.astype(cp.float64)
        c = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y)))
        return float(np.sqrt(max(0.0, 1 - c * c)))

    g0 = cp.random.default_rng(a.seed ^ 0xA)
    gen = lambda b: g0.standard_normal((b, n), dtype=cp.float32)   # noqa: E731
    base = fwd(gen, "exact")
    g0 = cp.random.default_rng(a.seed ^ 0xA)
    print(f"🔒S2 повтор точного проходу (тавтологія): {100*cos_err(fwd(gen,'exact'), base):.2e}%",
          flush=True)

    print(f"\n{'Qa біт':>7}{'Qw біт':>7}{'блок':>6}{'бюджет':>9}{'похибка':>11}"
          f"{'  контроль без блоків':>22}")
    print("─" * 66)
    for ba, bw, Kb in [(3, 3, 128), (4, 4, 32), (4, 4, 16), (5, 5, 8),
                       (6, 6, 4), (4, 8, 4), (8, 4, 4), (8, 8, 4)]:
        Qa = 2 ** ba - 1; Qw = 2 ** (bw - 1) - 1
        budget = Kb * Qa * Qw
        ok = "✓" if budget <= 4096 else "✗"
        g0 = cp.random.default_rng(a.seed ^ 0xA)
        e = cos_err(fwd(gen, "q", Qa, Qw, Kb), base)
        g0 = cp.random.default_rng(a.seed ^ 0xA)
        e2 = cos_err(fwd(gen, "q", Qa, Qw, Kb, blockwise=False), base)
        print(f"{ba:>7}{bw:>7}{Kb:>6}{budget:>7}{ok}{100*e:>10.4f}%{100*e2:>20.4f}%",
              flush=True)


if __name__ == "__main__":
    main()
