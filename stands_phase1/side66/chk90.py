"""chk90 — ПЕРЕВІРКА БЛОКУ (Б): ЧИ СПРАВДІ ХМАРА НА ГЛИБИНІ «ВУЗЬКА»?

════════════════ ЩО ПЕРЕВІРЯЄМО І ЧОМУ ════════════════
ЮЗЕР сказав перевірити блок (Б). Він спирався на ТРИ числа з пам'яті:
    еф.ранг 127.6 → 3.1 · RMS|t| 0.002 → 3.378 · 27.44% детерм. на шарі 30
і на них стоїть ЄДИНИЙ живий пункт «стан, точніший за двомоментний».

🚨 ЗНАЙДЕНА ДІРА В ПРИЛАДІ: `tprof88` рахував ранг як participation ratio
`PR = tr(C)²/tr(C²)`. Він РІЗКО занижується однією домінантною компонентою:
спектр {100, 1×255} дає PR = 12.3 при 256 ЖИВИХ напрямках. Тому «3.1» може
означати НЕ «хмара вузька», а «одна вісь у ~100× більша, решта жива».
Різниця вирішальна: у першому випадку стан справді низьковимірний (можна
тягнути 3 числа замість матриці), у другому — ні, і живий пункт помирає.

🚨 ДРУГА НЕТОЧНІСТЬ (моя, у відповіді ЮЗЕРУ): я сказав «ранг АКТИВАЦІЙ»,
а `tprof88` міряв ранг ПРЕАКТИВАЦІЙ `h`. Міряю обидва.

════════════════ ПЛОМБИ ════════════════
P1 ВІДТВОРЕННЯ: мій `PR` на преактиваціях шару 32 мусить дати 3.1 ± 0.5,
   а `RMS|t|` шару 1 → ~0.002, шару 32 → 3.378. Якщо ні — приладу вірити не
   можна і решта чисел нічого не значить.
P2 ПОВНИЙ РАНГ: для `z ~ N(0,I)` на ВХОДІ (шар 0) усі міри рангу мусять
   дати ~256 (PR, ентропійний, N90 ≈ 230).
P3 МАСШТАБУВАННЯ: N=2^15 проти N=2^17 — ранги мусять СТОЯТИ (не рости з N),
   інакше це шум оцінки коваріації, а не властивість мережі.
   ⚠️ Критично: при N < 256 вибіркова коваріація ВИРОДЖЕНА за побудовою.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
Ставлю до першого числа:
 (а) PR = 3.1 ВІДТВОРИТЬСЯ (шанс 85%) — прилад справний, число справжнє;
 (б) АЛЕ ентропійний ранг вийде 15…60, а `N90` (скільки компонент тримають
     90% дисперсії) — 5…40 (шанс 65%). Тобто «вузький клин» — перебільшення,
     і правильне формулювання: «одна домінантна вісь + довгий живий хвіст»;
 (в) шанс, що хмара СПРАВДІ майже одновимірна (N90 ≤ 3): **25%**.
Якщо (в) — живий пункт сильно зміцнюється. Якщо (б) — я мушу переписати
блок (Б) ЮЗЕРУ, бо переказав його неточно.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NS = 1 << int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 17
DT = torch.float32
BATCH = 1 << 12

W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
PROBE = [0, 1, 3, 7, 15, 23, 29, 31]
print(f"{DEV} · мереж {NNET} · N={NS} · ширина {n} · глибина {L}")


def ranks(C):
    """П'ять мір «скільки напрямків живі» для симетричної PSD-матриці."""
    sv = torch.linalg.eigvalsh(C.double())
    sv = torch.clamp(sv, min=0.0).flip(0)                 # спадно
    tot = float(sv.sum())
    p = sv / max(tot, 1e-300)
    pr = tot * tot / max(float((sv * sv).sum()), 1e-300)   # participation ratio
    ent = float(torch.exp(-(p * torch.log(p + 1e-300)).sum()))
    cs = torch.cumsum(p, 0)
    n90 = int((cs < 0.90).sum()) + 1
    n99 = int((cs < 0.99).sum()) + 1
    return pr, ent, float(p[0]), float(p[:3].sum()), float(p[:10].sum()), n90, n99


accH = np.zeros((NNET, len(PROBE), 7))
accA = np.zeros((NNET, len(PROBE), 7))
trms = np.zeros((NNET, L))
det30 = np.zeros((NNET, len(PROBE), 3))
t0 = time.time()

for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    with torch.no_grad():
        SH = torch.zeros(L, n, dtype=torch.float64, device=DEV)
        S2 = torch.zeros(L, n, dtype=torch.float64, device=DEV)
        SHH = torch.zeros(len(PROBE), n, n, dtype=torch.float64, device=DEV)
        SA = torch.zeros(len(PROBE), n, dtype=torch.float64, device=DEV)
        SAA = torch.zeros(len(PROBE), n, n, dtype=torch.float64, device=DEV)
        SP = torch.zeros(len(PROBE), n, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV).manual_seed(9000 + 13 * q)
        done = 0
        while done < NS:
            b = min(BATCH, NS - done)
            a = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
            for i in range(L):
                h = a @ ws[i]
                hd = h.double()
                SH[i] += hd.sum(0)
                S2[i] += (hd * hd).sum(0)
                a = torch.relu(h)
                if i in PROBE:
                    j = PROBE.index(i)
                    SHH[j] += (h.T @ h).double()
                    SA[j] += a.double().sum(0)
                    SAA[j] += (a.T @ a).double()
                    SP[j] += (h > 0).double().sum(0)
            done += b
        for i in range(L):
            m = SH[i] / NS
            v = torch.clamp(S2[i] / NS - m * m, min=1e-30)
            trms[q, i] = float(torch.sqrt(((m / torch.sqrt(v)) ** 2).mean()))
        for j, i in enumerate(PROBE):
            mh = SH[i] / NS
            Ch = SHH[j] / NS - torch.outer(mh, mh)
            accH[q, j] = ranks(Ch)
            ma = SA[j] / NS
            Ca = SAA[j] / NS - torch.outer(ma, ma)
            accA[q, j] = ranks(Ca)
            pj = SP[j] / NS
            det30[q, j] = [float((pj > 0.999).double().mean()),
                           float((pj < 0.001).double().mean()),
                           float(((pj - 0.5).abs() < 0.4).double().mean())]
    print(f"  net{q}  PR(h,шар32)={accH[q,-1,0]:.2f}  ент={accH[q,-1,1]:.1f}"
          f"  N90={accH[q,-1,5]:.0f}  RMS|t|32={trms[q,-1]:.3f}"
          f"  ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с")
print("=" * 96)
print("⛔ ПЛОМБИ")
print("=" * 96)
p1a, p1b, p1c = accH[:, -1, 0].mean(), trms[:, 0].mean(), trms[:, -1].mean()
print(f" P1 ВІДТВОРЕННЯ tprof88:")
print(f"    PR(h) шар32 = {p1a:.2f}   (запис: 3.1)  "
      f"{'✅' if abs(p1a - 3.1) < 0.6 else '⛔ ПРИЛАД НЕ ЗБІГАЄТЬСЯ'}")
print(f"    RMS|t| шар1 = {p1b:.4f}  (запис: 0.002) "
      f"{'✅' if p1b < 0.02 else '⛔'}")
print(f"    RMS|t| шар32= {p1c:.3f}   (запис: 3.378) "
      f"{'✅' if abs(p1c - 3.378) < 0.15 else '⛔'}")
print(f" P2 ПОВНИЙ РАНГ на шарі 1 (вхід ~ N(0,I)):")
print(f"    PR={accH[:,0,0].mean():.1f} ент={accH[:,0,1].mean():.1f} "
      f"N90={accH[:,0,5].mean():.0f}  (мусить бути ~256/~256/~230)")

for nm, arr in (("ПРЕАКТИВАЦІЇ h", accH), ("АКТИВАЦІЇ a", accA)):
    print()
    print("=" * 96)
    print(f"{nm} · центрована коваріація · {NNET} мереж × N={NS}")
    print("=" * 96)
    print(f"{'шар':>5s} {'PR':>8s} {'ентроп.':>9s} {'топ-1':>8s} {'топ-3':>8s}"
          f" {'топ-10':>8s} {'N90':>6s} {'N99':>6s}")
    for j, i in enumerate(PROBE):
        a = arr[:, j].mean(0)
        print(f"{i+1:>5d} {a[0]:>8.2f} {a[1]:>9.2f} {100*a[2]:>7.2f}%"
              f" {100*a[3]:>7.2f}% {100*a[4]:>7.2f}% {a[5]:>6.0f} {a[6]:>6.0f}")

print()
print("=" * 96)
print("ДЕТЕРМІНОВАНІСТЬ НЕЙРОНІВ (усередині мережі, потім середнє по мережах)")
print("=" * 96)
print(f"{'шар':>5s} {'p>0.999':>10s} {'p<0.001':>10s} {'РАЗОМ':>10s} {'граничні':>11s}")
for j, i in enumerate(PROBE):
    d = det30[:, j].mean(0)
    print(f"{i+1:>5d} {100*d[0]:>9.2f}% {100*d[1]:>9.2f}% "
          f"{100*(d[0]+d[1]):>9.2f}% {100*d[2]:>10.2f}%")
print("\n  (запис у пам'яті: 27.44% детермінованих на шарі 30)")

np.savez("/mnt/win_d/llm/White-Box/arena62/logs/chk90.npz",
         accH=accH, accA=accA, trms=trms, det=det30, probe=np.array(PROBE), NS=NS)
