"""ann90 — АНІГІЛЯЦІЯ: чи слабшає скорочення плюсів і мінусів з глибиною?

════════════════ ТЕЗА ЮЗЕРА ════════════════
«вагів цілий квадрат 256×256, але вхід 256 і вихід 256, і велику долю грають
мінуси, які йдуть ДО нейрона… має бути закон, при якому ці мінуси анігілюють
розмірність… на глибині з нами ці мінуси тільки і грають».

📐 МЕХАНІЗМ, ЯКИЙ ЗА ЦИМ СТОЇТЬ:
після ReLU всі `a_i ≥ 0`, тож у колонці нейрона `j` доданки `W_ij·a_i` мають
знак САМОЇ ВАГИ. Приблизно половина додатних, половина від'ємних ⟹
    h_j = Σ₊ − Σ₋   — РІЗНИЦЯ ДВОХ МАЙЖЕ РІВНИХ ВЕЛИКИХ СУМ.
Міра виживання:  A_j = |h_j| / Σ_i |W_ij·a_i|.
Для незалежних доданків теорія дає A = √(π/2)/√n = 1.2533/16 = 0.07833,
тобто з 256 доданків «виживає» ефективно 16.

════════════════ ГІПОТЕЗА, ЯКУ ПЕРЕВІРЯЮ ════════════════
З глибиною активації корелюються ⟹ доданки перестають бути незалежними ⟹
анігіляція СЛАБШАЄ ⟹ `|m|/σ` росте. Якщо так, то ОДИН механізм пояснює
відразу: `RMS|t|` 0.002→3.378 · 62% детермінованих нейронів · падіння
ефективного рангу 125.8→12.

════════════════ ПЛОМБИ ════════════════
P1 ШАР 1 (вхід `z~N(0,I)`, доданки НЕЗАЛЕЖНІ): A мусить дати 0.0783 ± 0.003.
   Якщо ні — прилад або теорія хибні, і решта чисел нічого не варта.
P2 КОНТРОЛЬ ЗНАКІВ: якщо перевісити знаки ваг ВИПАДКОВО (зберігши модулі),
   A на глибині мусить ПОВЕРНУТИСЬ ближче до 0.0783 — бо саме узгодженість
   знаків із активаціями створює виживання. Це відрізняє «анігіляція слабшає
   через кореляції активацій» від «слабшає з будь-якої причини».
P3 МАСШТАБУВАННЯ: N=2^12 проти 2^14 — числа мусять стояти.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
 (а) P1 пройде (шанс 85%);
 (б) A РОСТИМЕ з глибиною, з 0.078 до 0.15…0.40 (шанс 70%);
 (в) corr(A_j, |t_j|) по нейронах буде СИЛЬНА, >0.8 (шанс 75%) — бо це
     майже те саме число, записане двома способами;
 (г) 🚨 ключове і НЕ очевидне: чи A_j пояснює похибку ЗАМИКАННЯ понейронно?
     Тут ставлю НИЗЬКО (шанс 20%), бо C87 уже показала: похибка замикання —
     ВЛАСТИВІСТЬ ПАРИ, і понейронні скани її не ловлять (R²≈0).
🎯 Шанс, що з цього вийде щось із `k>1.2` (гейт окупності): 15%.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 8
NS = 1 << int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 13
DT = torch.float32
CH = 1 << 11

W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
THEO = float(np.sqrt(np.pi / 2) / np.sqrt(n))
print(f"{DEV} · мереж {NNET} · N={NS} · теорія A = √(π/2)/√{n} = {THEO:.5f}"
      f"  (виживає {1/THEO:.1f} з {n})\n")


def run(ws, seed, shuffle_signs=False):
    """A_j по шарах · t_j · еф.число доданків · кореляція активацій."""
    if shuffle_signs:
        g0 = torch.Generator(device=DEV).manual_seed(seed + 5)
        ws = [w.abs() * torch.where(torch.rand(w.shape, generator=g0, device=DEV)
                                    < 0.5, -1.0, 1.0) for w in ws]
    g = torch.Generator(device=DEV).manual_seed(seed)
    SA = torch.zeros(L, n, dtype=torch.float64, device=DEV)   # Σ|h|
    SS = torch.zeros(L, n, dtype=torch.float64, device=DEV)   # Σ Σ|W a|
    SH = torch.zeros(L, n, dtype=torch.float64, device=DEV)
    S2 = torch.zeros(L, n, dtype=torch.float64, device=DEV)
    NE = torch.zeros(L, n, dtype=torch.float64, device=DEV)   # еф.число доданків
    AC = torch.zeros(L, dtype=torch.float64, device=DEV)      # сер.попарна corr a
    done = 0
    while done < NS:
        b = min(CH, NS - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(L):
            w = ws[i]
            h = x @ w
            absum = x.abs() @ w.abs()                 # Σ_i |W_ij a_i|
            sq = (x * x) @ (w * w)                    # Σ_i (W_ij a_i)²
            SA[i] += h.abs().double().sum(0)
            SS[i] += absum.double().sum(0)
            NE[i] += (absum * absum / sq.clamp(min=1e-30)).double().sum(0)
            SH[i] += h.double().sum(0)
            S2[i] += (h * h).double().sum(0)
            x = torch.relu(h)
            if done == 0:
                xc = x - x.mean(0, keepdim=True)
                sd = xc.std(0).clamp(min=1e-20)
                Cm = (xc.T @ xc) / b / torch.outer(sd, sd)
                AC[i] = float((Cm.sum() - Cm.diagonal().sum()) / (n * n - n))
        done += b
    A = (SA / SS).cpu().numpy()
    m = SH / NS
    v = (S2 / NS - m * m).clamp(min=1e-30)
    t = (m / v.sqrt()).cpu().numpy()
    return A, t, (NE / NS).cpu().numpy(), AC.cpu().numpy()


res = [run([torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                            device=DEV) for i in range(L)], 700 + q)
       for q in range(NNET)]
A = np.mean([r[0] for r in res], 0)
T = np.mean([np.abs(r[1]) for r in res], 0)
NEF = np.mean([r[2] for r in res], 0)
ACR = np.mean([r[3] for r in res], 0)

print("=" * 92)
print("АНІГІЛЯЦІЯ ПО ГЛИБИНІ · A = |h_j| / Σ|W_ij·a_i|")
print("=" * 92)
print(f"{'шар':>5s} {'A (виживає)':>13s} {'у разах':>10s} {'еф.доданків':>13s} "
      f"{'RMS|t|':>9s} {'сер.corr(a)':>12s}")
for i in (0, 1, 2, 3, 7, 11, 15, 23, 29, 31):
    print(f"{i+1:>5d} {A[i].mean():>13.5f} {1/A[i].mean():>9.1f}× "
          f"{NEF[i].mean():>13.1f} {np.sqrt((T[i]**2).mean()):>9.3f} {ACR[i]:>12.4f}")

print(f"\n⛔ P1 ШАР 1 (доданки незалежні): A = {A[0].mean():.5f}  "
      f"теорія {THEO:.5f}  "
      f"{'✅' if abs(A[0].mean()-THEO) < 0.004 else '⛔ ПРИЛАД АБО ТЕОРІЯ ХИБНІ'}")

# ═══ P2: контроль — перевісити знаки випадково ═══
res2 = [run([torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                             device=DEV) for i in range(L)], 700 + q, True)
        for q in range(min(NNET, 4))]
A2 = np.mean([r[0] for r in res2], 0)
print(f"\n⛔ P2 КОНТРОЛЬ «ЗНАКИ ПЕРЕВІШЕНІ ВИПАДКОВО» (модулі ті самі):")
print(f"{'шар':>5s} {'A справжня':>12s} {'A перевішена':>14s} {'різниця':>10s}")
for i in (0, 7, 15, 31):
    print(f"{i+1:>5d} {A[i].mean():>12.5f} {A2[i].mean():>14.5f} "
          f"{A[i].mean()-A2[i].mean():>+10.5f}")

# ═══ зв'язок A з t понейронно ═══
print(f"\n📊 ЗВ'ЯЗОК A_j ↔ |t_j| ПОНЕЙРОННО (кореляція всередині шару)")
print(f"{'шар':>5s} {'corr(A_j,|t_j|)':>17s} {'R²':>8s}")
for i in (1, 7, 15, 23, 31):
    cs = [np.corrcoef(r[0][i], np.abs(r[1][i]))[0, 1] for r in res]
    c = float(np.mean(cs))
    print(f"{i+1:>5d} {c:>+17.4f} {c*c:>8.4f}")

print(f"\n📣 ВИРОК")
r = A[31].mean() / A[0].mean()
print(f"   анігіляція від шару 1 до 32: A {A[0].mean():.5f} → {A[31].mean():.5f}"
      f"  (×{r:.3f})")
print(f"   ⟹ {'СЛАБШАЄ з глибиною — гіпотеза ЮЗЕРА підтверджена' if r > 1.15 else ('ПОСИЛЮЄТЬСЯ' if r < 0.87 else 'практично НЕ МІНЯЄТЬСЯ')}")
print(f"   еф.число доданків: {NEF[0].mean():.1f} → {NEF[31].mean():.1f}"
      f"  (з {n} наявних)")
print(f"   сер.кореляція активацій: {ACR[0]:.4f} → {ACR[31]:.4f}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/ann90.npz",
         A=A, T=T, NEF=NEF, ACR=ACR, A2=A2, theo=THEO)
