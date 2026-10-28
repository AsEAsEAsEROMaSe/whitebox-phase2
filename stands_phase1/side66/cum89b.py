"""cum89b — κ₃ (МАРГІНАЛЬ) І κ₂₁ (КОПУЛА) РАЗОМ, А НЕ ПОРІЗНО.

════════════════ ЧОМУ САМЕ ЦЕ ════════════════
Оракульні заміри С88 показали механізм, який ми досі не використали:
    оракул лише m       ×0.912   ← ПОГІРШУЄ
    оракул лише C       ×0.725   ← ПОГІРШУЄ
    оракул m+C РАЗОМ    ×20.97   ← СТРИБОК
Канали замикання ВЗАЄМНО СКОРОЧУЮТЬСЯ: точне виправлення ОДНОГО ламає баланс,
двох одразу — дає 20×. Досі кожну поправку ми міряли ПОРІЗНО і щоразу діставали
×1.0x, після чого закривали гілку.

Тепер маємо рівно дві поправки на РІЗНІ канали:
    κ₃  (маргіналь)   → входить у `m_a` і в ДІАГОНАЛЬ `C_a`   [cum89: сам ×1.008]
    κ₂₁ (копула)      → входить у ПОЗАДІАГОНАЛЬ `C_a`         [mix88: сам ×1.314]
РАЗОМ їх не пробував ніхто.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
Три сценарії, ставлю ЗАРАЗ, щоб не тлумачити заднім числом:
  (а) k(обидва) ≈ k(κ₃)·k(κ₂₁) ≈ 1.32  — просто множаться, ефекту немає  55%
  (б) k(обидва) > 2                     — часткове скорочення каналів     30%
  (в) k(обидва) > 5                     — той самий механізм, що m+C      15%
⚠️ Навіть (в) не дає лідера (потрібно k=31.2), але (в) означало б, що напрям
«міняти весь замикач одразу» ЖИВИЙ, і його треба добудовувати.

════════════════ ПЛОМБИ ════════════════
P1  вимкнені обидві поправки ⟹ БІТОВО база
P2  шар 1 (вхід точно гаусів): |κ₂₁| ≈ 0
P3  МАСШТАБУВАННЯ: N×4 мусить учетверо збити шум оракульних κ₂₁; якщо виграш
    стоїть — він справжній, якщо росте з N — це підгонка під шум
P4  база = 0.9382% (те саме число, що в cum89 на тих самих мережах)
🚨 КОНТРОЛЬ «ЧУЖА ПОПРАВКА» (закон С88): κ₂₁ від СУСІДНЬОЇ мережі. Якщо чужа
   дає стільки ж — виграш є ефектом калібровки, а не інформацією.
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
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 16
N = 1 << LOGN
DT = torch.float32
SAME_SEED = (len(sys.argv) > 4 and sys.argv[4] == "same")
SQ2PI = float(np.sqrt(2.0 * np.pi))

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"пристрій {DEV} · мереж {NNET} · семплів 2^{LOGN}={N} · L={L} · n={n}")
print(f"сіди істини й оракула: {'🚨СПІЛЬНІ (контрольний прогін)' if SAME_SEED else '✅РОЗДІЛЕНІ'}")


def phi(t):
    return torch.exp(-0.5 * t * t) / SQ2PI


def Phi(t):
    return 0.5 * (1.0 + torch.erf(t * 0.7071067811865476))


def relu_moments(mp, s, g3):
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


def mc_pass(ws, nsamp, seed, want):
    """Один МК-прохід: істина + (за потреби) понейронні κ₃ і мішані κ₂₁ пошарово.

    κ₂₁[j,k] = E[δh_j²·δh_k] здобувається з СИРИХ моментів за один прохід:
        E[h_j²h_k] − 2m_j E[h_j h_k] − m_k E[h_j²] + 2m_j² m_k
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    K3 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    K21 = [None] * L
    VARH = [None] * L
    if want:
        S1 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
        S2 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
        S3 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
        H1 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
        H2 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
        HH = [torch.zeros(n, n, dtype=torch.float64, device=DEV) for _ in range(L)]
        H2H = [torch.zeros(n, n, dtype=torch.float64, device=DEV) for _ in range(L)]
    CH = 1 << 13
    done = 0
    while done < nsamp:
        b = min(CH, nsamp - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(L):
            h = x @ ws[i]
            if i == L - 1:
                acc += torch.clamp(h, min=0.0).sum(0).double()
                break
            if want:
                hd = h.double()
                H1[i] += hd.sum(0)
                H2[i] += (hd * hd).sum(0)
                HH[i] += hd.T @ hd
                H2H[i] += (hd * hd).T @ hd
            x = torch.clamp(h, min=0.0)
            if want:
                xd = x.double()
                S1[i] += xd.sum(0)
                S2[i] += (xd * xd).sum(0)
                S3[i] += (xd * xd * xd).sum(0)
        done += b
    out = acc / nsamp
    if want:
        for i in range(L - 1):
            a1, a2, a3 = S1[i] / nsamp, S2[i] / nsamp, S3[i] / nsamp
            K3[i] = a3 - 3.0 * a1 * a2 + 2.0 * a1 ** 3
            m1, m2 = H1[i] / nsamp, H2[i] / nsamp
            hh, h2h = HH[i] / nsamp, H2H[i] / nsamp
            # E[δ_j² δ_k]
            K21[i] = (h2h - 2.0 * m1.view(-1, 1) * hh
                      - m1.view(1, -1) * m2.view(-1, 1)
                      + 2.0 * (m1 * m1).view(-1, 1) * m1.view(1, -1))
            VARH[i] = torch.clamp(m2 - m1 * m1, min=1e-30)
    return out, K3, K21, VARH


def predict(ws, use_k3, K21o=None, K3o=None, k21_dimless=False, lam=1.0):
    m = torch.zeros(n, dtype=DT, device=DEV)
    C = torch.eye(n, dtype=DT, device=DEV)
    k3 = torch.zeros(n, dtype=DT, device=DEV)
    for i in range(L - 1):
        w = ws[i]
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
        mp = m @ w
        g3 = (k3 / s ** 3) if use_k3 else None
        e1, e2, e3, P, f = relu_moments(mp, s, g3)
        var = torch.clamp(e2 - e1 * e1, min=1e-30)
        k3a = e3 - 3.0 * e1 * e2 + 2.0 * e1 ** 3
        if K3o is not None:
            k3a = K3o[i].to(DT)
        Ca = cp * torch.outer(P, P) + 0.5 * (cp * cp) * torch.outer(f / s, f / s)
        if K21o is not None and K21o[i] is not None:
            # δCov_jk = s_j s_k[Φ_j(φ_k/2)κ̃₁₂ + (φ_j/2)Φ_k κ̃₂₁]
            # 🚨 для КОНТРОЛЮ «чужа» беремо БЕЗРОЗМІРНИЙ κ̃₂₁ сусіда і вживаємо
            #    з НАШИМИ `s`. Інакше контроль порожній: чужий МАСШТАБ сам по
            #    собі рознесе рекурсію, і ми переплутаємо це з «немає інформації».
            k21 = (K21o[i].to(DT) if k21_dimless
                   else (K21o[i].to(DT)) / torch.outer(s * s, s))
            corr = torch.outer(P, f * 0.5) * k21.T + torch.outer(f * 0.5, P) * k21
            Ca = Ca + lam * corr * torch.outer(s, s)
            # 🛡️ чужа поправка може зробити діагональ від'ємною ⟹ nan далі.
            #    Клампимо, щоб КОНТРОЛЬ давав число, а не порожнечу.
            Ca = torch.nan_to_num(Ca, nan=0.0, posinf=0.0, neginf=0.0)
        Ca = Ca + torch.diag(var - torch.diagonal(Ca))
        m, C = e1, Ca
        k3 = (w if i + 1 >= L else ws[i + 1])
        if use_k3:
            k3 = torch.nan_to_num((ws[i + 1] ** 3).T @ k3a, nan=0.0,
                                  posinf=0.0, neginf=0.0)
    w = ws[L - 1]
    cp = w.T @ (C @ w)
    s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
    mp = m @ w
    g3 = (k3 / s ** 3) if use_k3 else None
    e1, _, _, _, _ = relu_moments(mp, s, g3)
    return torch.nan_to_num(e1, nan=0.0, posinf=0.0, neginf=0.0)


LAMS = (0.25, 0.5, 0.75, 1.0)
VAR = (["base", "k3", "k21", "k3+k21"]
       + [f"своя λ={l}" for l in LAMS] + [f"ЧУЖА λ={l}" for l in LAMS])
res = {v: [] for v in VAR}
prevK21 = None
t0 = time.time()
seal2 = []
for j in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[j, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    # 🚨 РОЗДІЛЕНІ СІДИ: істина і оракульні κ з НЕЗАЛЕЖНИХ МК-прогонів.
    #    Спільний сід дав би шум κ, підігнаний під шум істини ⟹ фіктивний виграш.
    truth, _, _, _ = mc_pass(ws, N, 7000 + j, want=False)
    _, K3o, K21o, VARo = mc_pass(ws, N, 555000 + j, want=True)
    if SAME_SEED:
        truth, K3o, K21o, VARo = mc_pass(ws, N, 7000 + j, want=True)
    tr = truth.to(DT)
    den = float((tr * tr).mean())
    # P2 нормовано: сирий κ₂₁ має РІЗНИЙ масштаб по шарах (s росте), тож
    #    порівнювати сирі величини безглуздо — беремо безрозмірний κ₂₁/(s_j²s_k).
    def _rel(i):
        sd = torch.sqrt(VARo[i])
        return float((K21o[i].abs() / torch.outer(sd * sd, sd)).mean())
    seal2.append((_rel(0), _rel(8), _rel(20)))
    outs = {
        "base": predict(ws, False),
        "k3": predict(ws, True, K3o=K3o),
        "k21": predict(ws, False, K21o=K21o),
        "k3+k21": predict(ws, True, K3o=K3o, K21o=K21o),
    }
    # 🚨 КОНТРОЛЬ ЧЕРЕЗ СИЛУ ПОПРАВКИ. «Чужа» в лоб тут ПОРОЖНІЙ контроль: будь-яка
    #    невірна κ₂₁ рознесе 31-крокову рекурсію, тож він проходить завжди. Натомість
    #    скануємо λ: своя мусить мати оптимум при λ>0, чужа — при λ→0.
    sK = [None if K21o[i] is None else
          (K21o[i] / torch.outer(VARo[i], torch.sqrt(VARo[i]))).to(DT)
          for i in range(L)]
    for l in LAMS:
        outs[f"своя λ={l}"] = predict(ws, True, K3o=K3o, K21o=sK,
                                      k21_dimless=True, lam=l)
        outs[f"ЧУЖА λ={l}"] = (predict(ws, True, K3o=K3o, K21o=prevK21,
                                       k21_dimless=True, lam=l)
                               if prevK21 is not None else outs[f"своя λ={l}"])
    for v in VAR:
        res[v].append(float(((outs[v] - tr) ** 2).mean() / den) ** 0.5)
    prevK21 = sK
    if j < 2 or j % 6 == 0:
        print(f"  net{j:>3d}  база {100 * res['base'][-1]:.4f}%  κ₃ {100 * res['k3'][-1]:.4f}%"
              f"  κ₂₁ {100 * res['k21'][-1]:.4f}%  РАЗОМ {100 * res['k3+k21'][-1]:.4f}%")

print(f"\n  час {time.time() - t0:.0f} с")
b = np.array(res["base"])
print()
print("=" * 86)
print(f"РЕЗУЛЬТАТ · {NNET} мереж · N=2^{LOGN} · L={L}")
print("=" * 86)
print(f"  P4 база = {100 * b.mean():.4f}%   (cum89 на тих самих мережах дав 0.9382%)")
sc = np.array(seal2)
print(f"  P2 безрозмірний |κ₂₁|/(s²s): шар1 = {sc[:, 0].mean():.5f} · "
      f"шар9 = {sc[:, 1].mean():.5f} · шар21 = {sc[:, 2].mean():.5f}")
print(f"     {'✅ шар1 МАЛИЙ (вхід точно гаусів)' if sc[:, 0].mean() < 0.4 * sc[:, 1].mean() else '⚠️ шар1 НЕ малий — або баг, або МК-шум домінує'}")
print()
print(f"{'варіант':>16s} {'похибка':>10s} {'k':>9s} {'у балі':>9s}")
ks = {}
for v in VAR:
    a = np.array(res[v])
    k = float((b ** 2).mean() / (a ** 2).mean())
    ks[v] = k
    print(f"{v:>16s} {100 * a.mean():>9.4f}% {k:>9.3f} {(k + 10.33) / 11.33:>9.4f}")
print()
print(f"  🔬 ЧИ Є СКОРОЧЕННЯ КАНАЛІВ: k(разом)={ks['k3+k21']:.3f} проти "
      f"добутку k(κ₃)·k(κ₂₁)={ks['k3'] * ks['k21']:.3f}")
print(f"     ⟹ {'🎯 Є НАДЛИШОК' if ks['k3+k21'] > 1.15 * ks['k3'] * ks['k21'] else 'просто множаться'}")
print()
print("  🚨 КОНТРОЛЬ ЧЕРЕЗ СИЛУ ПОПРАВКИ λ (чужа мусить мати оптимум при λ→0):")
bs, bc = max(LAMS, key=lambda l: ks[f"своя λ={l}"]), max(LAMS, key=lambda l: ks[f"ЧУЖА λ={l}"])
for l in LAMS:
    print(f"     λ={l:<5} своя k={ks[f'своя λ={l}']:>7.3f}   чужа k={ks[f'ЧУЖА λ={l}']:>7.3f}")
print(f"     оптимум: своя λ={bs} (k={ks[f'своя λ={bs}']:.3f}) · "
      f"чужа λ={bc} (k={ks[f'ЧУЖА λ={bc}']:.3f})")
print(f"     ⟹ {'✅ ІНФОРМАЦІЯ' if ks[f'своя λ={bs}'] > 1.3 * ks[f'ЧУЖА λ={bc}'] else '⛔ ефект калібровки'}")
np.savez(f"/mnt/win_d/llm/White-Box/arena62/logs/cum89b_n{NNET}_l{LOGN}.npz",
         **{v: np.array(res[v]) for v in VAR})
