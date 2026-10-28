"""qmc96 — КВАЗІВИПАДКОВІ ТОЧКИ ПРОТИ ПСЕВДОВИПАДКОВИХ. НІКОЛИ НЕ МІРЯЛОСЬ.

════════════════ ЧОМУ ЦЕ ВЗАГАЛІ ПИТАННЯ ════════════════
`[ЗМІРЯНО ladder96/ladderm96, С96]` МК-дисперсія `V=2.05e-6` і є нашим балом:
`MSE = V·E²/(V+E²) ≈ 0.911·V`, бо `E² = 6.5e-5 ≫ V`. Тека `side66/../s76_qmc`
існує ПОРОЖНЬОЮ, і в пам'яті НЕМАЄ ЖОДНОГО числа про QMC ⟹ клас незміряний.
Функція строго 1-однорідна ⟹ задача живе на сфері.

🚨 ЧЕСНЕ ЗАСТЕРЕЖЕННЯ ДО ЗАПУСКУ: два записи працюють ПРОТИ гіпотези.
 (1) `[ЦИТАТА finding-function-is-worse-than-uniformly-high-dimensional]`
     функція ГІРША за рівномірно-256-вимірну ⟹ малої ефективної вимірності,
     на якій і живе перевага QMC, може просто не бути.
 (2) ВИБІЛЮВАННЯ вже робить вибіркові 1-й і 2-й моменти ТОЧНИМИ — а це рівно
     та частина інтегранда (лінійна + квадратична), де QMC зазвичай і виграє.
     ⟹ очікую СИЛЬНЕ перекриття, і саме воно є головним ризиком гілки.

════════════════ ЩО ПОРІВНЮЄМО (усе на ОДНАКОВОМУ N) ════════════════
 A   iid                                    гола база
 B   iid + антитетика
 C   iid + антитетика + вибілювання         ← НАШ ДІЮЧИЙ СХЕМА
 D   Соболь(scramble) + антитетика + вибілювання
 E   Соболь(scramble) + антитетика, БЕЗ вибілювання   ← ізолює перекриття
 F   Соболь БЕЗ scramble + антитетика + вибілювання

Оцінюємо ДИСПЕРСІЮ, а не одну реалізацію: `REP` незалежних розіграшів на
мережу, MSE усереднюється по (мережа × розіграш). `[ЦИТАТА закон С91]`
«шум МІРЯТИ РОЗІГРАШАМИ: `σ/√N` чинна ЛИШЕ без каналів, з ними завищує в
√g = 1.3–1.65» ⟹ аналітичну формулу тут не використовуємо взагалі.

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) D проти C дасть <×1.3                                   шанс 65%
 (б) E проти B дасть ×1.5…×3 (QMC сам по собі працює)        шанс 55%
 (в) 🔑 перекриття підтвердиться: (E/B) ≫ (D/C)              шанс 60%
 (г) scramble кращий за без-scramble (F гірший за D)         шанс 70%
 (д) C проти A відтворить записані ×1.798·×1.593 ≈ ×2.9      шанс 55%
     ← ПЛОМБА приладу на ВІДОМОМУ числі, до будь-яких висновків
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

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 8
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
NPAIR = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        fm = np.asarray(t.column("final_means").combine_chunks()
                        .flatten()).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i]
            got += 1


def sobol_normal(n, d, seed, scramble, dev):
    """Соболь → рівномірні → Φ⁻¹ → нормальні. `ndtri` через erfinv."""
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=scramble,
                                        seed=seed)
    u = eng.draw(n).to(dev).double().clamp(1e-12, 1.0 - 1e-12)
    return (torch.erfinv(2.0 * u - 1.0) * (2.0 ** 0.5)).float()


def forward_mean(z, W, whiten):
    """Антитетика [z;−z] + (опційно) вибілювання; повертає E[ReLU(a_32)]."""
    x = z @ whiten if whiten is not None else z
    y = x @ W[0]
    a = torch.cat([torch.relu(y), torch.relu(-y)], dim=0)
    for L in range(1, 32):
        a = torch.relu(a @ W[L]) if L < 31 else (a @ W[L])
    return torch.relu(a).double().mean(0)


def whitener(z):
    n = z.shape[0]
    cov = (z.T @ z).double() / n
    L = torch.linalg.cholesky(cov)
    return torch.linalg.inv(L).T.float()


print(f"qmc96 · {DEV} · {NNET} мереж × {REP} розіграшів · {NPAIR} пар "
      f"(={2*NPAIR} семплів)", flush=True)
t0 = time.time()
KEYS = ['A', 'B', 'C', 'D', 'E', 'F']
SE = {k: [] for k in KEYS}
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        tn2 = float((tru * tru).sum())
        for r in range(REP):
            sd = 700000 + 9173 * q + 31 * r
            g = torch.Generator(device=DEV).manual_seed(sd)
            zi = torch.randn(NPAIR, 256, generator=g, device=DEV)
            zs = sobol_normal(NPAIR, 256, sd, True, DEV)
            zf = sobol_normal(NPAIR, 256, sd, False, DEV)
            eye = torch.eye(256, device=DEV)

            def rel(v):
                return float(((v - tru) ** 2).sum() / tn2)
            # A: iid БЕЗ антитетики, але з ТИМ САМИМ N=2·NPAIR свіжих семплів
            # 🚨 перша версія дублювала ті самі NPAIR ⟹ ефективне N удвічі
            #    менше і `C/A` завищувалось РІВНО вдвічі. Виправлено.
            g2 = torch.Generator(device=DEV).manual_seed(sd + 500003)
            za = torch.randn(2 * NPAIR, 256, generator=g2, device=DEV)
            aa = torch.relu(za @ W[0])
            for L in range(1, 32):
                aa = torch.relu(aa @ W[L]) if L < 31 else (aa @ W[L])
            SE['A'].append(rel(torch.relu(aa).double().mean(0)))
            SE['B'].append(rel(forward_mean(zi, W, None)))
            SE['C'].append(rel(forward_mean(zi, W, whitener(zi))))
            SE['D'].append(rel(forward_mean(zs, W, whitener(zs))))
            SE['E'].append(rel(forward_mean(zs, W, None)))
            SE['F'].append(rel(forward_mean(zf, W, whitener(zf))))
        if q % 2 == 1 or q == 0:
            print(f"  net{q}: C={np.mean(SE['C'][-REP:])**0.5*100:.4f}% "
                  f"D={np.mean(SE['D'][-REP:])**0.5*100:.4f}% "
                  f"({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 76)
print("QMC ПРОТИ ПСЕВДОВИПАДКОВИХ — MSE, усереднена по мережах × розіграшах")
print("=" * 76)
names = {'A': 'iid, без антитетики (гола база)',
         'B': 'iid + антитетика',
         'C': '🔑iid + антитетика + вибілювання = НАША СХЕМА',
         'D': '🔥Соболь(scr) + антитетика + вибілювання',
         'E': 'Соболь(scr) + антитетика, БЕЗ вибілювання',
         'F': 'Соболь БЕЗ scramble + антитетика + вибілювання'}
mA = float(np.mean(SE['A']))
mC = float(np.mean(SE['C']))
mB = float(np.mean(SE['B']))
print(f"{'вар':>4s}{'віднос.RMS':>13s}{'MSE ×A':>9s}{'MSE ×C':>9s}  опис")
for k in KEYS:
    m = float(np.mean(SE[k]))
    print(f"{k:>4s}{m**0.5*100:>12.4f}%{mA/m:>9.2f}{mC/m:>9.2f}  {names[k]}")
print(f"\n🔒 ПЛОМБА НА ВІДОМИХ ЧИСЛАХ: антитетика (B/A) записана ×1.593 — "
      f"вийшло ×{mA/mB:.2f}; вибілювання (C/B) записане ×1.798 — вийшло "
      f"×{mB/mC:.2f}; разом ×{mA/mC:.2f} проти записаних ×2.86")
print(f"🔑 ПЕРЕКРИТТЯ: QMC БЕЗ вибілювання дає ×{mB/float(np.mean(SE['E'])):.2f} "
      f"(E проти B), З вибілюванням ×{mC/float(np.mean(SE['D'])):.2f} (D проти C).")
print("   Якщо перше ≫ друге — вибілювання вже забрало те, що дає QMC.")
print(f"💰 ПЕРЕКЛАД У БАЛ: V=2.05e-6·(C/D) ⟹ бал = 0.1·V·E²/(V+E²), E=0.85%")
for k in ('C', 'D'):
    V = 2.05e-6 * float(np.mean(SE[k])) / mC
    e2 = 0.9097 * (0.0085 ** 2)
    print(f"   {k}: V={V:.3e} ⟹ бал {0.1*V*e2/(V+e2):.4e}")
