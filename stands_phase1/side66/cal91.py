"""cal91 — КАЛІБРОВКА СТЕНДУ ПРОТИ ЧУЖИХ ЕТАЛОННИХ ЧИСЕЛ.

Таблиця стартер-кіту (`docs/concepts/scoring-model.md`), `mini` split, 100 MLP,
бюджет 2.72e11, істина 1e9 семплів:
```
zeros                    ~0.91      ✅ уже відтворено: 0.9093
mean_propagation         ~9.5e-04   діагональна дисперсія, O(d·w²), ~11M флопів
covariance_propagation   ~8.4e-05   повна коваріація, O(d·w³), ~1.6e9 флопів
```
🎓 НАШ ЗАКОН: спершу ЕТАЛОН, тоді прилад. Якщо стенд не відтворює ЧУЖІ числа —
всі наші вироки стоять на непроградуйованому приладі.

════════════════ ПЛОМБИ ════════════════
P1 mean_propagation → 9.5e-04 ±15%
P2 covariance_propagation → 8.4e-05 ±15%
P3 наше замикання (Мелер K=2) мусить бути НЕ ГІРШЕ за covariance_propagation
P4 zeros → 0.9093 (уже пройдено)
"""
import glob
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "mini"
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/" + SPLIT)
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327


def rows():
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        fm = np.asarray(t.column("final_means").combine_chunks().flatten()
                        ).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            yield w[i], fm[i].astype(np.float64)


def _relu_moments(mp, s):
    t = mp / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mp * p + s * f
    va = ((mp * mp + s * s) * p + mp * s * f - ma * ma).clamp(min=0.0)
    return p, f, ma, va


def mean_prop(ws):
    """ДІАГОНАЛЬНА дисперсія: стан (m, v), ціна O(d·w²)."""
    n = ws[0].shape[0]
    m = torch.zeros(n, dtype=torch.float64, device=DEV)
    v = torch.ones(n, dtype=torch.float64, device=DEV)
    for i, w in enumerate(ws):
        wd = w.double()
        mp = m @ wd
        s = (v @ (wd * wd)).clamp(min=1e-30).sqrt()
        _, _, ma, va = _relu_moments(mp, s)
        m, v = ma, va
    return m


def cov_prop(ws, mehler=False):
    """ПОВНА коваріація. mehler=False — гаусова позадіагональ `cp·Φ⊗Φ`
    (це `covariance_propagation`); True — наш ряд Мелера K=2."""
    n = ws[0].shape[0]
    m = torch.zeros(n, dtype=torch.float64, device=DEV)
    C = torch.eye(n, dtype=torch.float64, device=DEV)
    L = len(ws)
    for i, w in enumerate(ws):
        wd = w.double()
        cp = wd.T @ (C @ wd)
        s = cp.diagonal().clamp(min=1e-20).sqrt()
        mp = m @ wd
        p, f, ma, va = _relu_moments(mp, s)
        m = ma
        if i == L - 1:
            break
        C = cp * torch.outer(p, p)
        if mehler:
            C = C + (0.5 * (cp * cp)) * torch.outer(f / s, f / s)
        C = C + torch.diag(va - C.diagonal())
    return m


ROW = {"mean_propagation": [], "covariance_propagation": [],
       "наше замикання (Мелер K=2)": [], "zeros": []}
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows()):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
              for i in range(32)]
        tr = torch.as_tensor(truth, device=DEV, dtype=torch.float64)
        for name, pred in (
                ("mean_propagation", mean_prop(ws)),
                ("covariance_propagation", cov_prop(ws, mehler=False)),
                ("наше замикання (Мелер K=2)", cov_prop(ws, mehler=True)),
                ("zeros", torch.zeros_like(tr))):
            d = pred - tr
            ROW[name].append(float((d * d).mean()))
        if q % 25 == 0:
            print(f"  net{q:>4d} ({time.time()-t0:.0f} с)", flush=True)
print(f"\n  час {time.time()-t0:.0f} с · мереж {len(ROW['zeros'])}\n")

REF = {"zeros": 0.91, "mean_propagation": 9.5e-04,
       "covariance_propagation": 8.4e-05, "наше замикання (Мелер K=2)": None}
print("=" * 96)
print(f"КАЛІБРОВКА ПРОТИ ЧУЖИХ ЕТАЛОНІВ · split={SPLIT} · "
      f"{len(ROW['zeros'])} MLP")
print("=" * 96)
print(f"{'оцінювач':>30s} {'наш стенд':>14s} {'таблиця доків':>15s} "
      f"{'відношення':>12s}")
for k, v in ROW.items():
    a = np.array(v)
    r = REF[k]
    rr = f"{a.mean()/r:.4f}" if r else "—"
    print(f"{k:>30s} {a.mean():>14.4e} {(f'~{r:.2e}' if r else '—'):>15s} "
          f"{rr:>12s}")
ok = all(abs(np.mean(ROW[k]) / REF[k] - 1) < 0.15 for k in REF if REF[k])
print(f"\n📣 {'✅ СТЕНД ГРАДУЙОВАНИЙ — відтворює ЧУЖІ числа' if ok else '🚨 СТЕНД НЕ ГРАДУЙОВАНИЙ'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/cal91_" + SPLIT + ".npz",
         **{k.replace(" ", "_"): np.array(v) for k, v in ROW.items()})
