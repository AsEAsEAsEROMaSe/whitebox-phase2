"""richa101 — ГОЛОВНИЙ ТЕСТ: ЧИ ВИЖИВАЄ ВИГРАШ ПІСЛЯ ОПТИМАЛЬНОГО МАСШТАБУ?

`[ЗМІРЯНО rich101]` різниця двох замикань різної ширини `Δcl = m_cl(128) −
m_cl(256)` корелює з реальною похибкою: corr −0.5704, 🔒чужа 0.0208;
β переноситься (розкид 38.5%), LOO ×1.2239, крос A→B ×1.2326, 🔒чужа Δcl ×0.8646.

🚨 АЛЕ ЦЕ СИРА ПОХИБКА. Наш власний записаний закон (С97, що вбив κ₃):
   «понад ПОЛОВИНА того, що виправляє поправка, — це МАСШТАБ, а `_ALPHA`
    бере масштаб ДАРОМ». Сира похибка 0.96% → після вільного скаляра 0.45%.
⟹ ЧЕСНИЙ ТЕСТ: обидві гілки з ВЛАСНИМ оптимальним `a`, бо ціль є ТОЧНО
   косинусною (С100): min_a‖a·m − T‖² = ‖T‖²(1 − cos²(m,T)).

І ДРУГЕ ПИТАННЯ: чи працює це на ДРУГІЙ половині нейронів (підмережа
W[128:,128:]) — без цього покриваються лише 128 із 256.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) виграш після зняття масштабу впаде щонайменше вдвічі          80%
 (б) але лишиться > ×1.05                                          45%
 (в) на другій половині нейронів ефект буде того ж порядку         70%
"""
import math

import numpy as np

SP = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
      "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/")
d = np.load(SP + "rich101.npz")
NS1 = [96, 128, 160, 192, 224]
NNET = d["cl256"].shape[0]
print(f"richa101 · {NNET} мереж\n")


def cos_err(m, t):
    """похибка ПІСЛЯ оптимального скаляра — рівно те, що бачить сабміт"""
    a = float(np.dot(m, t) / max(np.dot(m, m), 1e-300))
    return math.sqrt(((a * m - t) ** 2).mean() / (t * t).mean())


def raw_err(m, t):
    return math.sqrt(((m - t) ** 2).mean() / (t * t).mean())


print(f"{'n₁':>5s}{'СИРА база':>11s}{'СИРА +β':>10s}{'×':>8s}"
      f"{'  |  ':>5s}{'КОСИН.база':>12s}{'КОСИН.+β':>11s}{'×':>8s}{'🔒чужа':>9s}")
print("=" * 82)
for n in NS1:
    CLn, CL2, TR2 = d[f"cl{n}"], d["cl256"], d["tr256"]
    D = [CLn[k] - CL2[k][:n] for k in range(NNET)]
    M = [CL2[k][:n] for k in range(NNET)]
    T = [TR2[k][:n] for k in range(NNET)]
    # β фітується на КОСИНУСНІЙ похибці (те, що реально мінімізується)
    bs = []
    for k in range(NNET):
        grid = np.linspace(-0.03, 0.03, 1201)
        v = [cos_err(M[k] - b * D[k], T[k]) for b in grid]
        bs.append(float(grid[int(np.argmin(v))]))
    bs = np.array(bs)
    r0 = np.mean([raw_err(M[k], T[k]) for k in range(NNET)])
    c0 = np.mean([cos_err(M[k], T[k]) for k in range(NNET)])
    # LOO — вирок
    rl, cl_, cf = [], [], []
    for k in range(NNET):
        bk = float(np.median(np.delete(bs, k)))
        rl.append(raw_err(M[k] - bk * D[k], T[k]))
        cl_.append(cos_err(M[k] - bk * D[k], T[k]))
        cf.append(cos_err(M[k] - bk * D[(k + 1) % NNET], T[k]))
    rl, cl_, cf = np.mean(rl), np.mean(cl_), np.mean(cf)
    print(f"{n:>5d}{100*r0:>10.4f}%{100*rl:>9.4f}%{r0/rl:>8.4f}"
          f"{'  |  ':>5s}{100*c0:>11.4f}%{100*cl_:>10.4f}%{c0/cl_:>8.4f}"
          f"{c0/cf:>9.4f}")

print("\n🔒 «СИРА» — те, що міряв rich101 · «КОСИН.» — те, що бачить САБМІТ")
print("   β медіана(n=128) =", f"{np.median(bs):.5f}" if NS1[-1] == 128 else "див. вище")
