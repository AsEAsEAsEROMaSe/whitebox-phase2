"""lam89b — ЧЕСНИЙ ОБ'ЄКТ: понейронні λ-канали З ПЕРЕНОСОМ, як у сабміті.

════════════════ ЩО БУЛО НЕ ТАК У `lam89` ════════════════
Там я згортав ознаки в СКАЛЯРИ (середнє по нейронах) і брав скалярний вихід.
Сабміт робить інакше — канал є ВЕКТОРОМ і ПЕРЕНОСИТЬСЯ крізь решту шарів
тими самими вагами:
    v ← v + λ_c[i]·(ознака_c на шарі i)        (вектор n)
    v ← (v @ w_k)·gate_k     для k = i+1 … L-1
    ours = out[-1] − v
Через це `lam89` провалив пломбу: шари 1..8 дали ×2.30 замість відомих
3.1–3.7×. Тут відтворюю механізм ТОЧНО — інакше це третя хвороба контролю
(«не той об'єкт»), і будь-яке число нічого не варте.

════════════════ ЩО ПЕРЕВІРЯЄМО ════════════════
`lam89` показав дуже сильний і СТІЙКИЙ ДО МАСШТАБУВАННЯ тренд
(R² 0.5659 → 0.9688 від 8 до 31 шару, стабільний при 24/48/96 прогонах).
Якщо він виживе на чесному об'єкті — це найбільша знахідка сесії, бо б'є
прямо в дисперсію МК, тобто у 91.2% балу, і коштує лише 4 редукції на шар
(≈0.094% B на всі 23 нові шари).

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
Ставлю зараз: пломба 1..8 тепер ПРОЙДЕ (×3.1…3.7, шанс 60%), а глибина дасть
додатково ×2…5 (шанс 45%), а не ×14 як у скалярній версії — бо понейронні
канали вже й так несуть більше інформації, тож приріст від глибини буде
меншим. Шанс, що додатковий виграш перевищить ×3.66 (тобто сам по собі
закриє розрив до лідера): **20%**.
⚠️ Якщо пломба знову не пройде — стенд усе ще не той об'єкт, і жодного
вироку я з нього не робитиму.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 12
MREP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
NS = 4096
DT = torch.float32
SQ2PI = float(np.sqrt(2.0 * np.pi))
GATE_N = 1024

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · прогонів {MREP} · N={NS} · ЧЕСНИЙ об'єкт (з переносом)")


def phi(t):
    return torch.exp(-0.5 * t * t) / SQ2PI


def Phi(t):
    return 0.5 * (1.0 + torch.erf(t * 0.7071067811865476))


def closure(ws):
    m = torch.zeros(n, dtype=DT, device=DEV)
    C = torch.eye(n, dtype=DT, device=DEV)
    S, M, Q, P_ = [], [], [], []
    for i in range(L - 1):
        w = ws[i]
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
        mp = m @ w
        t = mp / s
        Pp, f = Phi(t), phi(t)
        e1 = s * (t * Pp + f)
        e2 = (mp * mp + s * s) * Pp + (mp * s) * f
        var = torch.clamp(e2 - e1 * e1, min=1e-30)
        Ca = cp * torch.outer(Pp, Pp) + 0.5 * (cp * cp) * torch.outer(f / s, f / s)
        Ca = Ca + torch.diag(var - torch.diagonal(Ca))
        S.append(s); M.append(e1); Q.append(torch.diagonal(Ca) + e1 * e1); P_.append(Pp)
        m, C = e1, Ca
    return S, M, Q, P_


def one_run(ws, S, M, Q, P_, seed, nlay):
    """Повертає шум виходу (n) і ПЕРЕНЕСЕНІ внески кожного каналу (4·nlay, n)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(NS, n, generator=g, device=DEV, dtype=DT)
    x = torch.cat([z, -z], 0)
    V = torch.zeros(4 * nlay, n, dtype=DT, device=DEV)
    gates = []
    for i in range(L):
        p = x @ ws[i]
        if i == L - 1:
            out = torch.clamp(p, min=0.0).mean(0)
            break
        gate = (p[:GATE_N] > 0).to(DT).mean(0)
        gates.append(gate)
        # ПЕРЕНОС уже накопичених каналів крізь цей шар — рівно як у сабміті
        if i > 0:
            V = (V @ ws[i]) * gate
        mh = p.mean(0)
        sh = torch.sqrt(torch.clamp((p * p).mean(0) - mh * mh, min=1e-30))
        xx = torch.clamp(p, min=0.0)
        if i < nlay:
            V[4 * i + 0] = phi(mh / sh) * (sh - S[i])
            V[4 * i + 1] = xx.mean(0) - M[i]
            V[4 * i + 2] = (xx * xx).mean(0) - Q[i]
            V[4 * i + 3] = gate - P_[i]
        x = xx
    return out, V


rows = {nl: [] for nl in (4, 8, 12, 16, 24, 31)}
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    S, M, Q, P_ = closure(ws)
    for nl in rows:
        Ys, Vs = [], []
        for r in range(MREP):
            o, V = one_run(ws, S, M, Q, P_, 88000 + 991 * q + r, nl)
            Ys.append(o)
            Vs.append(V)
        Y = torch.stack(Ys)                       # (M, n)
        Vv = torch.stack(Vs)                      # (M, 4nl, n)
        Yc = (Y - Y.mean(0)).double().cpu().numpy()
        Vc = (Vv - Vv.mean(0)).double().cpu().numpy()
        rows[nl].append((Yc, Vc))
    if q % 4 == 0:
        print(f"  net{q:>3d}  ({time.time() - t0:.0f} с)")


def loo_drop(data):
    """LOO по мережах: λ (скаляри на канал) фітяться на решті мереж."""
    num = den = 0.0
    K = data[0][1].shape[1]
    for i in range(len(data)):
        A = np.concatenate([data[j][1][:, :, :].transpose(0, 2, 1).reshape(-1, K)
                            for j in range(len(data)) if j != i])
        b = np.concatenate([data[j][0].reshape(-1) for j in range(len(data)) if j != i])
        lam, *_ = np.linalg.lstsq(A, b, rcond=1e-8)
        Yi, Vi = data[i]
        res = Yi.reshape(-1) - Vi.transpose(0, 2, 1).reshape(-1, K) @ lam
        num += float((res ** 2).sum())
        den += float((Yi ** 2).sum())
    return 1.0 / max(1.0 - (1.0 - num / den), 1e-9), 1.0 - num / den


print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 74)
print(f"ЧЕСНИЙ ОБ'ЄКТ · {NNET} мереж × {MREP} прогонів · понейронно + перенос")
print("=" * 74)
print(f"{'шарів':>7s} {'каналів':>9s} {'LOO R²':>10s} {'дисперсія падає в':>20s}")
d = {}
for nl in sorted(rows):
    drop, r2 = loo_drop(rows[nl])
    d[nl] = drop
    print(f"{nl:>7d} {4 * nl:>9d} {r2:>10.4f} {drop:>19.3f}×")
ok8 = "ОК: об'єкт той" if 2.8 < d[8] < 4.2 else "СТЕНД ВСЕ ЩЕ НЕ ТОЙ ОБ'ЄКТ — вироку не роблю"
print(f"\n  ПЛОМБА: шари 1..8 дають x{d[8]:.2f} (відоме 3.1-3.7x) -> {ok8}")
print(f"  🎯 ГЛИБИНА: 1..31 дає ×{d[31]:.2f} ⟹ додатково ×{d[31] / d[8]:.3f}"
      f" ⟹ бал {1.8676e-07 * d[8] / d[31]:.4e}")
