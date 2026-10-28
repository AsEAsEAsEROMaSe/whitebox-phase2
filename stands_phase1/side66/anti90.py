"""anti90 — ЧИ ПРАЦЮЄ АНТИТЕТИКА `[z, −z]`, ЯКА СТОЇТЬ У САБМІТІ?

ЗВІДКИ ПИТАННЯ: ЮЗЕР — «форму ламають мінуси». Пломба P3 (`cone90`) показала,
що `−u` перевертає патерн активації ПОВНІСТЮ вже на шарі 1 (0.00% збігу).
Якщо на виході `f(z)` і `f(−z)` стають майже незалежними, антитетика ЗАЙМАЄ
ПОЛОВИНУ СЕМПЛІВ І НЕ КУПУЄ НІЧОГО.

Ми ніколи цього не міряли. Вага питання — 91.2% балу (дисперсія МК).

════════════════ ЩО РІВНО МІРЯЮ ════════════════
1. `corr(f(z), f(−z))` понейронно на виході. Антитетика допомагає ⟺ corr<0.
2. ГОЛОВНЕ: дисперсія оцінки при ОДНАКОВОМУ ЧИСЛІ ПРОГОНІВ `N`:
     ANTI  = N/2 незалежних `z` + їхні дзеркала
     INDEP = N незалежних `z`
   Виграш = Var(INDEP)/Var(ANTI). >1 — антитетика купує, <1 — ШКОДИТЬ.
3. 🚨 ТРЕТЯ ХВОРОБА КОНТРОЛЮ: у сабміті стоїть ВИБІЛЮВАННЯ. Міряю ОБИДВА
   режими, бо вибілювання само по собі ×1.798, і на його тлі відповідь може
   бути іншою. Об'єкт мусить бути той самий, що в сабміті.
4. Побічно: чи має пара `(z,−z)` спільну норму (так, тотожно) ⟹ антитетика
   НЕ МОЖЕ знімати радіальну дисперсію — це робить лише вибілювання.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
Ставлю до першого числа:
 (а) `corr(f(z),f(−z))` на виході буде близька до 0, у діапазоні −0.15…+0.10
     (шанс 65%) — бо патерн перевертається і глибина все перемішує;
 (б) виграш антитетики БЕЗ вибілювання: ×1.0…1.15 (шанс 60%);
 (в) виграш ПОВЕРХ вибілювання: ×0.98…1.06, тобто практично нуль (шанс 55%).
🎯 Шанс, що антитетика ШКОДИТЬ (виграш <0.97) і її зняття дасть >×1.03: 20%.
⚠️ Якщо (в) підтвердиться — це НЕ привід її знімати: вона коштує нуль флопів
і нуль residual. Але тоді половина «пар» — це просто половина семплів, і всі
наші оцінки «скільки семплів у бюджеті» треба читати інакше.

ПЛОМБА НЕЗМІЩЕНОСТІ: обидві схеми при великому N мусять збігтися з істиною.
ПЛОМБА ШАРУ 1: для ОДНОШАРОВОЇ мережі антитетика мусить дати ВЕЛИКИЙ виграш,
бо `ReLU(h)+ReLU(−h) = |h|` тотожно ⟹ знак усереднюється ТОЧНО. Якщо на L=1
виграшу немає — прилад зламаний.
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
MREP = int(sys.argv[3]) if len(sys.argv) > 3 else 40
NTOT = 5432                       # як у сабміті: 2716 пар = 5432 прогони
NTRUTH = 1 << 19
DT = torch.float32

W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · повторів {MREP} · прогонів {NTOT} (як у сабміті)")


def fwd(ws, x, depth=None):
    D = L if depth is None else depth
    for i in range(D):
        h = x @ ws[i]
        if i == D - 1:
            return torch.clamp(h, min=0.0)
        x = torch.clamp(h, min=0.0)


def whiten_of(z):
    cov = torch.einsum("ia,ib->ab", z, z) / z.shape[0]
    Lc = torch.linalg.cholesky(cov.double())
    return torch.linalg.inv(Lc).T.to(DT)


def est(ws, mode, wh, seed, depth=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    if mode == "anti":
        z = torch.randn(NTOT // 2, n, generator=g, device=DEV, dtype=DT)
        z = torch.cat([z, -z], 0)
    else:
        z = torch.randn(NTOT, n, generator=g, device=DEV, dtype=DT)
    if wh:
        z = z @ whiten_of(z)
    return fwd(ws, z, depth).mean(0)


def truth(ws, seed, depth=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    done = 0
    while done < NTRUTH:
        b = min(1 << 13, NTRUTH - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        acc += fwd(ws, x, depth).sum(0).double()
        done += b
    return (acc / NTRUTH).to(DT)


MODES = [("indep", False), ("anti", False), ("indep", True), ("anti", True)]
mse = {m: [] for m in MODES}
bias = {m: [] for m in MODES}
corrs, cor1 = [], []
gain1 = []
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth(ws, 7000 + q)
    den = float((tr * tr).mean())
    for m in MODES:
        acc, ms = 0.0, torch.zeros(n, dtype=torch.float64, device=DEV)
        for r in range(MREP):
            p = est(ws, m[0], m[1], 808000 + 977 * q + r)
            acc += float(((p - tr) ** 2).mean() / den)
            ms += p.double()
        mse[m].append(acc / MREP)
        bias[m].append(float((ms / MREP - tr.double()).mean()
                             / max(float(tr.double().mean()), 1e-30)))
    # 1. кореляція f(z) проти f(−z) на ВИХОДІ
    g = torch.Generator(device=DEV).manual_seed(555 + q)
    z = torch.randn(1 << 13, n, generator=g, device=DEV, dtype=DT)
    fp, fm = fwd(ws, z), fwd(ws, -z)
    a = fp - fp.mean(0, keepdim=True)
    b = fm - fm.mean(0, keepdim=True)
    c = (a * b).mean(0) / torch.clamp(a.std(0) * b.std(0), min=1e-30)
    corrs.append(float(c.mean()))
    # ПЛОМБА ШАРУ 1: одношарова мережа
    f1p, f1m = fwd(ws, z, 1), fwd(ws, -z, 1)
    a1 = f1p - f1p.mean(0, keepdim=True)
    b1 = f1m - f1m.mean(0, keepdim=True)
    cor1.append(float(((a1 * b1).mean(0)
                       / torch.clamp(a1.std(0) * b1.std(0), min=1e-30)).mean()))
    tr1 = truth(ws, 7000 + q, 1)
    d1 = float((tr1 * tr1).mean())
    vi = va = 0.0
    for r in range(12):
        vi += float(((est(ws, "indep", False, 909000 + 977 * q + r, 1) - tr1) ** 2).mean() / d1)
        va += float(((est(ws, "anti", False, 909000 + 977 * q + r, 1) - tr1) ** 2).mean() / d1)
    gain1.append(vi / max(va, 1e-30))
    if q % 6 == 0:
        print(f"  net{q:>3d}  corr(вихід)={corrs[-1]:+.4f}  corr(шар1)={cor1[-1]:+.4f}"
              f"  L=1 виграш ×{gain1[-1]:.2f}  ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n")
print("=" * 88)
print(f"АНТИТЕТИКА `[z,−z]` · {NNET} мереж × {MREP} повторів · {NTOT} прогонів")
print("=" * 88)
print("⛔ ПЛОМБА ШАРУ 1 (ReLU(h)+ReLU(−h)=|h| ⟹ знак усереднюється ТОЧНО):")
g1 = float(np.mean(gain1))
print(f"   corr(f(z),f(−z)) на L=1 = {np.mean(cor1):+.4f}   (мусить бути СИЛЬНО <0)")
print(f"   виграш антитетики L=1   = ×{g1:.3f}   "
      f"{'✅ прилад бачить ефект' if g1 > 1.5 else '⛔ ПРИЛАД НЕ БАЧИТЬ ЕФЕКТУ'}")

print(f"\n📊 КОРЕЛЯЦІЯ НА ВИХОДІ (L={L}): {np.mean(corrs):+.5f}"
      f"  ± {np.std(corrs, ddof=1)/np.sqrt(NNET):.5f}")
print(f"   ⟹ {'антитетика МОЖЕ купувати' if np.mean(corrs) < -0.02 else 'дзеркало практично НЕЗАЛЕЖНЕ'}")

print(f"\n{'режим':>22s} {'віднос.похибка':>16s} {'виграш проти indep':>20s}")
for wh in (False, True):
    bi = np.mean(mse[("indep", wh)])
    ba = np.mean(mse[("anti", wh)])
    tagw = "з вибілюванням" if wh else "БЕЗ вибілювання"
    print(f"{tagw + ' · indep':>22s} {100*bi**0.5:>15.4f}% {'—':>20s}")
    print(f"{tagw + ' · anti':>22s} {100*ba**0.5:>15.4f}% {bi/ba:>19.4f}×")
gw = np.mean(mse[("indep", True)]) / np.mean(mse[("anti", True)])
print(f"\n🎯 ВИРІШАЛЬНЕ ЧИСЛО (режим сабміта, поверх вибілювання): ×{gw:.4f}")
print(f"   ⟹ бал за зняття антитетики: {1.86762e-07 * gw:.4e}  (рекорд 1.86762e-07)")
print(f"   {'⛔ АНТИТЕТИКА ШКОДИТЬ — знімати' if gw < 0.97 else ('✅ купує — лишати' if gw > 1.03 else '⚪ нуль — байдуже')}")

print("\n⛔ ПЛОМБА НЕЗМІЩЕНОСТІ (відносний зсув середнього по прогонах):")
for m in MODES:
    b = np.array(bias[m])
    se = b.std(ddof=1) / np.sqrt(len(b))
    print(f"   {('anti' if m[0]=='anti' else 'indep'):>5s}"
          f" {'+вибіл' if m[1] else '      '}: {b.mean():+.3e} ± {se:.3e}"
          f"  (t={b.mean()/max(se,1e-30):+.2f})")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/anti90.npz",
         **{f"{m[0]}_{int(m[1])}": np.array(mse[m]) for m in MODES},
         corrs=np.array(corrs), cor1=np.array(cor1), gain1=np.array(gain1))
