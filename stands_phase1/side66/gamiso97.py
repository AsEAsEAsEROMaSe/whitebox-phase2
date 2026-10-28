"""gamiso97 — ЧИ НЕГАУСОВІСТЬ ОДНОВИМІРНА, ЧИ ІЗОТРОПНА. КОНТРОЛЬ, ЩО ВИСИТЬ.

════════════════ ДВА ПРОТИЛЕЖНІ ЗАПИСИ ПРО ОДИН ОБ'ЄКТ ════════════════
```
С95: γ₃(û) = 0.427 проти γ₃(ПК⊥) = −0.049  ⟹ «негаусовість ОДНОВИМІРНА, ×8.7»
С86: 🚨ПЛОМБА — ВИПАДКОВИЙ напрямок має ТІ САМІ γ (0.372/0.393)
     ⟹ «властивість РОЗПОДІЛУ В ЦІЛОМУ»
```
Реконструкція АРХІВУ (~70%, НЕ вимір): `−0.049` — ПІДПИСАНЕ середнє (≈0 за
побудовою, бо знак напрямку довільний), `0.427` — МОДУЛЬ ⟹ розрив із нічого.

🚨 ЧОМУ ЦЕ ВАЖИТЬ, ХОЧА ЦЕ НЕ БАЛ: на «мережа = промінь, негаусовість = ОДИН
СКАЛЯР» стоять ТРИ речі — суміш гаусів уздовж `û` (С95, ×1.39), ранг-1 `κ₃`,
і сам вибір `û` як привілейованого напрямку. Якщо розподіл ІЗОТРОПНО
негаусовий, привілейованого напрямку НЕ ІСНУЄ і всі троє втрачають опору.

════════════════ ЩО МІРЯЄМО ════════════════
Об'єкт — проєкція активацій `h_ℓ` на одиничний напрямок `u`:
```
   p = h_ℓ·u ,  γ₃(p) = E[(p−μ)³]/σ³
```
Три набори `u` на КОЖНОМУ шарі, і для кожного — ОБИДВІ статистики:
```
   NEUR  нормовані стовпці W[ℓ+1]   — РЕАЛЬНІ нейрони (те, що бачить мережа)
   RAND  100 випадкових ізотропних  — контроль «чужий напрямок»
   PC1   перша головна компонента   — те, що С95 звала `û`
   PCT   останні 128 ПК (хвіст)     — те, що С95 звала `ПК⊥`
```
🔑 РІШУЧА ПАРА ЧИСЕЛ — `mean(γ₃)` проти `median(|γ₃|)` в ОДНОМУ рядку.
Якщо підписане ≈0, а модуль ≈0.4 в УСІХ наборах — «розрив ×8.7» був
модулем проти знакозмінного середнього, і С86 права.

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) mean(γ₃) по RAND буде |·| < 0.05 на всіх шарах            шанс 90%
 (б) median(|γ₃|) по RAND буде 0.30…0.45 на глибоких шарах     шанс 75%
 (в) median(|γ₃|) RAND / NEUR буде в межах ×1.3                шанс 70%
     ⟹ ізотропно, С86 права, «промінь» падає
 (г) PC1 виб'ється з RAND більш ніж у ×2                       шанс 25%
     ⟹ якщо так — С95 права, привілейований напрямок Є
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NS = 1 << (int(sys.argv[3]) if len(sys.argv) > 3 else 18)
CH = 1 << 14
NR = 100
SHOW = (1, 4, 8, 16, 24, 31)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def gamma3(P):
    """P: (NS, k) fp64 ⟹ γ₃ по стовпцях."""
    mu = P.mean(0)
    d = P - mu
    v = (d * d).mean(0).clamp_min(1e-300)
    return (d ** 3).mean(0) / v.pow(1.5)


print(f"gamiso97 · {DEV} · {NNET} мереж · {NS} семплів · {NR} вип. напрямків",
      flush=True)
t0 = time.time()
ACC = {k: {s: [] for s in SHOW} for k in ("NEUR", "RAND", "PC1", "PCT")}

with torch.no_grad():
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        if got >= NNET:
            break
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        for i in range(t.num_rows):
            if got >= NNET:
                break
            W = [torch.as_tensor(np.ascontiguousarray(w[i][L]), device=DEV)
                 for L in range(32)]
            g = torch.Generator(device=DEV).manual_seed(970000 + 13 * got)
            U = torch.randn(256, NR, generator=g, device=DEV)
            U = U / U.norm(dim=0, keepdim=True)

            # ── прогін із накопиченням Cov на потрібних шарах ──
            Hs = {}
            for o in range(0, NS, CH):
                zg = torch.Generator(device=DEV).manual_seed(
                    880000 + 97 * got + o)
                z = torch.randn(CH, 256, generator=zg, device=DEV)
                h = torch.relu(z @ W[0])
                for L in range(1, 32):
                    if L in SHOW:
                        Hs.setdefault(L, []).append(h.clone())
                    h = torch.relu(h @ W[L])
                if 31 in SHOW and 31 not in Hs:
                    pass
            for L in SHOW:
                H = torch.cat(Hs[L], 0)                       # (NS, 256)
                Hd = H.double()
                mu = Hd.mean(0)
                Xc = Hd - mu
                # PC — з коваріації активацій
                Cv = (Xc.T @ Xc) / NS
                ev, evec = torch.linalg.eigh(Cv)              # зрост. порядок
                pc1 = evec[:, -1:].float()
                pct = evec[:, :128].float()
                Wn = W[L] / W[L].norm(dim=0, keepdim=True)
                for key, D in (("NEUR", Wn), ("RAND", U),
                               ("PC1", pc1), ("PCT", pct)):
                    ACC[key][L].append(gamma3((H @ D).double()).cpu().numpy())
                del Hs[L]
            del Hs
            got += 1
            print(f"  net{got-1} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 78)
print("γ₃ ПРОЄКЦІЇ h_ℓ — ПІДПИСАНЕ СЕРЕДНЄ проти МЕДІАНИ МОДУЛЯ")
print("=" * 78)
hdr = f"{'шар':>4s}"
for k in ("NEUR", "RAND", "PC1", "PCT"):
    hdr += f"{k+' mean':>13s}{k+' |med|':>13s}"
print(hdr)
for L in SHOW:
    row = f"{L:>4d}"
    for k in ("NEUR", "RAND", "PC1", "PCT"):
        a = np.concatenate(ACC[k][L])
        row += f"{a.mean():>13.4f}{np.median(np.abs(a)):>13.4f}"
    print(row)

print("\n" + "=" * 78)
print("🎯 ТЕСТ РАНГУ κ₃ — РОЗКИД |γ₃| ПО 100 ВИПАДКОВИХ НАПРЯМКАХ")
print("=" * 78)
print("  Якщо κ₃ = c·a⊗a⊗a (РАНГ-1), то γ₃(u) ∝ (a·u)³ ⟹ розкид ГІГАНТСЬКИЙ:")
print("  (a·u) ~ N(0,1/n) ⟹ p95/p05 від |(a·u)|³ ≈ (1.96/0.063)³ ≈ 3.0e4.")
print("  Якщо κ₃ «повний і випадковий», γ₃(u) ~ N(0,·) ⟹ p95/p05 ≈ 31.")
print(f"\n{'шар':>4s}{'p05':>10s}{'p50':>10s}{'p95':>10s}{'p95/p05':>11s}"
      f"{'вирок':>28s}")
for L in SHOW:
    a = np.abs(np.concatenate(ACC["RAND"][L]))
    q5, q50, q95 = (float(np.percentile(a, p)) for p in (5, 50, 95))
    r = q95 / max(q5, 1e-30)
    v = "РАНГ-1" if r > 3000 else ("повний/випадковий" if r < 200 else "?")
    print(f"{L:>4d}{q5:>10.4f}{q50:>10.4f}{q95:>10.4f}{r:>11.1f}{v:>28s}")
print("\n  🔒 ПЛОМБА ЕТАЛОНОМ: |N(0,1)| має p95/p05 = 1.960/0.0627 = 31.3")
print("\n" + "=" * 78)
print("КОНТРОЛЬ АРХІВУ: ЕКСЦЕС САМОЇ ВИБІРКИ {γ₃(v)} — ПІДПИСАНОЇ, НЕ МОДУЛЯ")
print("=" * 78)
print("  гаус (ЦГТ / CP з великим r)  ексцес ≈  0")
print("  «стала амплітуда + знак»     ексцес ≈ −2  (бімодальний)")
print("  ексцес > 0 (важкі хвости)    ⟹ пояснення АРХІВУ падає, аномалія справжня")
print(f"\n{'шар':>4s}{'ексцес γ₃(v)':>15s}{'ексцес |γ₃(v)|':>17s}{'вирок':>26s}")
for L in SHOW:
    a = np.concatenate(ACC["RAND"][L])
    for lab, x in (("", a), ("abs", np.abs(a))):
        c = x - x.mean()
        k = float((c ** 4).mean() / max((c * c).mean() ** 2, 1e-300) - 3.0)
        if lab == "":
            k0 = k
        else:
            k1 = k
    v = ("концентрація модуля" if k0 < -0.5
         else ("гаусів" if abs(k0) <= 0.5 else "важкі хвости"))
    print(f"{L:>4d}{k0:>15.4f}{k1:>17.4f}{v:>26s}")
print("=" * 78)
print("🔑 ЧИТАТИ ВИРОК:")
print("  RAND |med| ≈ NEUR |med| (у межах ×1.3)  ⟹ ІЗОТРОПНО, С86 права,")
print("     «мережа = промінь, негаусовість = один скаляр» ПАДАЄ")
print("  PC1 |med| > ×2 від RAND               ⟹ С95 права, напрямок Є")
print("  mean ≈ 0 скрізь при |med| ≈ 0.4       ⟹ «розрив ×8.7» був артефактом")
print("     порівняння МОДУЛЯ з ПІДПИСАНИМ середнім")
