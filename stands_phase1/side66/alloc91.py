"""alloc91 — БЮДЖЕТ Є СПІЛЬНИМ КАЗАНОМ: розподіляти семпли МІЖ МЕРЕЖАМИ.

════════════════ ЗМІНА ОБ'ЄКТА ОПТИМІЗАЦІЇ ════════════════
91 сесію ми оптимізуємо ОДНУ мережу. Але бал рахується по ВСІХ одразу, і
множник ОДИН на весь сабміт (`sub99`: MSE 1.86325e-06 × 0.10023 = 1.86762e-07).
⟹ бюджет — спільний казан, а ми ділимо його ПОРІВНУ. Це не оптимально.

    min Σ Vᵢ/Nᵢ  при  Σ Nᵢcᵢ = const   ⟹   Nᵢ ∝ √(Vᵢ/cᵢ)
    виграш = (Σcᵢ)(Σ Vᵢ/cᵢ · cᵢ) / (Σ√(Vᵢcᵢ))²   [загальний випадок]

`Vᵢ` = ПОСЕМПЛОВА дисперсія оцінки виходу мережі `i` = Σ_j Var_z(f_j(z)).
`cᵢ` = ціна семпла після обрізання (вона теж різна між мережами!).

🎁 ОЦІНКА `Vᵢ` КОШТУЄ НУЛЬ: пілот УЖЕ ганяє 256 семплів крізь усю мережу
   ⟹ сума квадратів виходу = 2·256·256 = 131k флопів = 0.0005% бюджету.

════════════════ ПЛОМБИ ════════════════
P1 ВІДТВОРИТИ РІВНОМІРНИЙ РЕЖИМ: при `Nᵢ=const` таблиця мусить дати
   похибку рівня сабміта (~0.24%) і частку дисперсії ~90%.
P2 ТОТОЖНІСТЬ ОПТИМУМУ: підставити `Nᵢ ∝ √(Vᵢ/cᵢ)` у `Σ Vᵢ/Nᵢ` і звірити з
   аналітичною формулою — мусить збігтись до 1e-12.
P3 ⛔КОНТРОЛЬ «ЧУЖА ОЗНАКА»: розподіл за ВИПАДКОВИМ `Ṽᵢ` тієї ж дисперсії.
   Мусить дати ×<1.0 (гірше за рівномірний). Якщо чужа теж виграє — виграє
   сама нерівномірність, а не інформація.
P4 ⚠️ ЗМІЩЕННЯ НЕ ПАДАЄ З `N`: `MSE = V/N + b²`. Оптимізувати треба ПОВНУ
   MSE, а не саму дисперсію, інакше перекинемо семпли туди, де їх не треба.

════════════════ ПЕРЕДБАЧЕННЯ (записано ДО запуску) ════════════════
 (а) розкид `V` між мережами ≥3×                    — шанс 75%
 (б) оптимум з ІСТИННИМ `V` дасть ×1.2…2.0          — шанс 60%
 (в) пілотний проксі збереже ≥70% виграшу           — шанс 55%
 (г) чужа ознака дасть <×1.00                       — шанс 85%
 🎯 підсумково ×1.15…1.6 до балу з шансом **50%**
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 64
PILOT = 128
NTRUTH = 1 << 19
NVAR = 1 << 16          # семплів для ІСТИННОЇ посемплової дисперсії
DT = torch.float32

W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · ширина {n} · глибина {L}\n", flush=True)


def pilot_alive(ws, seed):
    """РІВНО як у сабміті + сума квадратів виходу (проксі `V` задарма)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    zp = torch.randn(PILOT, n, generator=g, device=DEV, dtype=DT)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(L):
        h = a @ ws[i]
        if i < L - 1:
            alive.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    o = a.double()                       # 256 реалізацій виходу
    vhat = float((o.var(0, unbiased=True)).sum())
    return alive, vhat


def cut(ws, alive):
    out, fl = [], 0.0
    prev = torch.ones(n, dtype=torch.bool, device=DEV)
    for i in range(L):
        k = alive[i] if i < L - 1 else torch.ones(n, dtype=torch.bool, device=DEV)
        out.append(ws[i][prev][:, k].contiguous())
        fl += 2.0 * int(prev.sum()) * int(k.sum())
        prev = k
    return out, fl


def fwd(ws, z):
    x = z
    for i in range(len(ws)):
        h = x @ ws[i]
        x = torch.relu(h) if i < len(ws) - 1 else h
    return torch.relu(h)


def acc_moments(ws, seed, N, whiten_batch=None):
    """Σf і Σf² по `N` семплах (без вибілювання — потрібна ЧИСТА посемплова V)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    s1 = torch.zeros(n, dtype=torch.float64, device=DEV)
    s2 = torch.zeros(n, dtype=torch.float64, device=DEV)
    done = 0
    while done < N:
        b = min(1 << 13, N - done)
        f = fwd(ws, torch.randn(b, n, generator=g, device=DEV, dtype=DT)).double()
        s1 += f.sum(0)
        s2 += (f * f).sum(0)
        done += b
    return s1 / N, s2 / N


VT, VH, CI, TR, SC = [], [], [], [], []
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    al, vh = pilot_alive(ws, 4242 + q)
    wcut, cost = cut(ws, al)
    m1, m2 = acc_moments(ws, 5100 + 31 * q, NVAR)
    vtrue = float((m2 - m1 * m1).sum())          # Σ_j Var(f_j) — ПОСЕМПЛОВА
    tr, _ = acc_moments(ws, 3000 + 17 * q, NTRUTH)
    VT.append(vtrue); VH.append(vh); CI.append(cost)
    TR.append(float((tr * tr).sum())); SC.append(float(tr.sum()))
    if q % 16 == 0:
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)
VT, VH, CI, TR = map(np.array, (VT, VH, CI, TR))
print(f"\n  час {time.time()-t0:.0f} с\n", flush=True)

print("=" * 96)
print(f"РОЗКИД МІЖ МЕРЕЖАМИ · {NNET} мереж")
print("=" * 96)
for nm, arr in (("V (посемплова дисп.)", VT), ("V̂ (пілот, 256 семплів)", VH),
                ("ціна семпла c", CI), ("‖істина‖²", TR)):
    print(f"{nm:>24s}  мін {arr.min():.4e}  медіана {np.median(arr):.4e}  "
          f"макс {arr.max():.4e}  розмах ×{arr.max()/max(arr.min(),1e-300):.2f}  "
          f"CV {arr.std()/arr.mean():.4f}")

rho = float(np.corrcoef(np.log(VT), np.log(VH))[0, 1])
print(f"\n🔗 кореляція log V̂ ↔ log V (якість безкоштовного проксі): {rho:.4f}")


def total_mse(Nv):
    """Σ Vᵢ/Nᵢ — сумарна ДИСПЕРСІЙНА частина MSE (зміщення не залежить від N)."""
    return float((VT / np.maximum(Nv, 1e-9)).sum())


BUD = float((CI * 5717).sum())          # той самий сумарний казан, що зараз


def alloc(key):
    """Nᵢ ∝ √(keyᵢ/cᵢ), нормовано на спільний бюджет Σ Nᵢcᵢ = BUD."""
    r = np.sqrt(np.maximum(key, 0) / CI)
    return BUD * r / (r * CI).sum()


rng = np.random.default_rng(0)
foreign = np.exp(rng.normal(np.log(VT).mean(), np.log(VT).std(), NNET))
schemes = {
    "рівномірно (сабміт)": BUD / CI.sum() * np.ones(NNET),
    "оптимум з ІСТИННИМ V": alloc(VT),
    "оптимум з ПІЛОТНИМ V̂": alloc(VH),
    "⛔ЧУЖА (випадковий V)": alloc(foreign),
    "лише ціна (V=const)": alloc(np.ones(NNET)),
}
base = total_mse(schemes["рівномірно (сабміт)"])
print(f"\n{'схема':>24s} {'Σ V/N':>14s} {'× до рівномірного':>19s}")
for k, Nv in schemes.items():
    t = total_mse(Nv)
    print(f"{k:>24s} {t:>14.6e} {base/t:>19.4f}")
    assert abs(float((Nv * CI).sum()) / BUD - 1) < 1e-9, f"бюджет порушено: {k}"

opt = alloc(VT)
anal = (np.sqrt(VT * CI).sum()) ** 2 / BUD
print(f"\n⛔ P2 тотожність оптимуму: чисельно {total_mse(opt):.10e} проти "
      f"аналітики {anal:.10e} · Δ={abs(total_mse(opt)/anal-1):.2e} "
      f"{'✅' if abs(total_mse(opt)/anal-1) < 1e-9 else '❌'}")
print(f"⛔ P4 УВАГА: тут оптимізується лише `V/N`; зміщення `b²` НЕ входить "
      f"⟹ верхня оцінка виграшу")
g = base / total_mse(alloc(VH))
print(f"\n📣 ВИРОК: пілотний проксі дає ×{g:.4f} "
      f"(істинний V ×{base/total_mse(opt):.4f}, "
      f"чужа ×{base/total_mse(alloc(foreign)):.4f})")
print(f"   {'✅ Є ЩО БРАТИ' if g > 1.05 else '⛔ РОЗКИД ЗАМАЛИЙ'}")
print(f"   ⟹ бал {1.86762e-07/max(g,1e-9):.4e} (рекорд 1.86762e-07)")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/alloc91.npz",
         VT=VT, VH=VH, CI=CI, TR=TR, SC=np.array(SC))
