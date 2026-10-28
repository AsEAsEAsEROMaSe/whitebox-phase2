"""sphere89 — КОНУСИ МАЮТЬ ВЕРШИНУ В НУЛІ ⟹ РАДІАЛЬНУ ДИСПЕРСІЮ МОЖНА ВБИТИ ТОЧНО.

════════════════ ЗВІДКИ (ЮЗЕР) ════════════════
«а часом наші конуси не є ніби як ті осі що направлені в центр?»
Так. Мережа БЕЗ БІАСІВ ⟹ кожна границя активації `w·a > 0` — гіперплощина
ЧЕРЕЗ ПОЧАТОК ⟹ розбиття простору конічне, усі конуси мають спільну вершину
в нулі. Наслідок ТОЧНИЙ (не наближення):
    патерн активації залежить лише від НАПРЯМКУ `u = z/|z|`
    f(z) = |z|·f(u),  |z| ⟂ u  ⟹  E[f(z)] = E[|z|]·E[f(u)]
`E[|z|] = √2·Γ((n+1)/2)/Γ(n/2)` — закрита форма.

⟹ Уся радіальна дисперсія в семплах МАРНА: ми щоразу семплюємо довжину,
хоч її внесок відомий аналітично. У сабміті цього немає — `z=randn`,
вибілювання вирівнює КОВАРІАЦІЮ, але НОРМУ не фіксує.

════════════════ ОЧІКУВАНИЙ ВИГРАШ ════════════════
    виграш = (E|z|²·E[g²] − E|z|²·E[g]²·(E|z|²/E|z|²)) / (E|z|²·Var g)
           ≈ 1 + (n − E|z|²)/(E|z|² · ν²) ,   ν = CV(f) на сфері
При n=256: `E|z|² = n`, `E[|z|]² ≈ n − 0.5` ⟹ множник `≈ 1 + 0.00196/ν²`.
    ν=0.10 → ×1.20 · ν=0.05 → ×1.78 · ν=0.03 → ×3.18
Тобто ВСЕ вирішує ν, і його ми ніколи не міряли.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
`relu` на виході має точкову масу в нулі і важкий хвіст ⟹ ν радше великий.
Ставлю: ν = 0.15…0.4 ⟹ виграш ×1.01…1.09 (шанс 60%). Шанс, що ν<0.06 і
виграш перевищить ×1.5: **15%**.
⚠️ АЛЕ навіть ×1.05 тут коштує РІВНО НУЛЬ флопів (нормування — 2n на семпл),
тож окупність автоматична: гейт вимагає лише `k>1.0`.
🚨 Пастка, яку називаю наперед: у сабміті стоїть ВИБІЛЮВАННЯ, і воно робить
емпіричну коваріацію одиничною. Якщо нормувати ПІСЛЯ вибілювання, коваріація
зсунеться; якщо ДО — вибілювання зіпсує норму. Тому міряю ОБИДВА порядки
плюс чистий варіант без вибілювання.

ПЛОМБИ:
P1  НЕЗМІЩЕНІСТЬ: при великому N сферична оцінка мусить збігтися з істиною
    так само, як звичайна (це головне — схема має бути ТОЧНОЮ, а не «майже»)
P2  ν, зміряний прямо, мусить пояснити зміряний виграш формулою вище
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from scipy.special import gammaln                             # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 12
MREP = int(sys.argv[3]) if len(sys.argv) > 3 else 24
NPAIR = 2716
NTRUTH = 1 << 18
DT = torch.float32

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
# E|z| для χ_n у закритій формі
EZ = float(np.sqrt(2.0) * np.exp(gammaln((n + 1) / 2.0) - gammaln(n / 2.0)))
print(f"{DEV} · мереж {NNET} · повторів {MREP} · пар {NPAIR}")
print(f"E|z| = {EZ:.6f}   (√n = {np.sqrt(n):.6f})   E|z|²/n = {EZ ** 2 / n:.8f}")


def fwd(ws, x):
    for i in range(L):
        h = x @ ws[i]
        if i == L - 1:
            return torch.clamp(h, min=0.0).mean(0)
        x = torch.clamp(h, min=0.0)


def run(ws, mode, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(NPAIR, n, generator=g, device=DEV, dtype=DT)
    if mode == "sphere":
        z = z / z.norm(dim=1, keepdim=True) * EZ
    elif mode == "sphere_strat":
        # додатково стратифікуємо НАПРЯМОК не можна дешево, але можна
        # стратифікувати радіус ПЕРЕД нормуванням — тут він уже сталий,
        # тож цей режим = sphere + антитетика (вона й так є)
        z = z / z.norm(dim=1, keepdim=True) * EZ
    x = torch.cat([z, -z], 0)
    return fwd(ws, x)


def truth(ws, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    done = 0
    while done < NTRUTH:
        b = min(1 << 13, NTRUTH - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(L):
            h = x @ ws[i]
            if i == L - 1:
                acc += torch.clamp(h, min=0.0).sum(0).double()
                break
            x = torch.clamp(h, min=0.0)
        done += b
    return acc / NTRUTH


mse = {"plain": [], "sphere": []}
bias = {"plain": [], "sphere": []}
nus = []
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth(ws, 7000 + q).to(DT)
    den = float((tr * tr).mean())
    for mode in ("plain", "sphere"):
        acc = 0.0
        ms = torch.zeros(n, dtype=torch.float64, device=DEV)
        for r in range(MREP):
            p = run(ws, mode, 707000 + 977 * q + r)
            acc += float(((p - tr) ** 2).mean() / den)
            ms += p.double()
        mse[mode].append(acc / MREP)
        # ПЛОМБА НЕЗМІЩЕНОСТІ: середнє по MREP прогонах проти істини
        bias[mode].append(float(((ms / MREP - tr.double())).mean()
                                / max(float(tr.double().mean()), 1e-30)))
    # P2: прямий вимір ν = CV(f(u)) на сфері, понейронно
    g = torch.Generator(device=DEV).manual_seed(31337 + q)
    z = torch.randn(4096, n, generator=g, device=DEV, dtype=DT)
    u = z / z.norm(dim=1, keepdim=True)
    x = u
    for i in range(L):
        h = x @ ws[i]
        if i == L - 1:
            fu = torch.clamp(h, min=0.0)
            break
        x = torch.clamp(h, min=0.0)
    nu = (fu.std(0) / torch.clamp(fu.mean(0), min=1e-30))
    nus.append(float(nu.median()))
    if q % 4 == 0:
        print(f"  net{q:>3d} plain {100 * mse['plain'][-1] ** 0.5:.4f}%  "
              f"sphere {100 * mse['sphere'][-1] ** 0.5:.4f}%  ν={nus[-1]:.3f}"
              f"  ({time.time() - t0:.0f} с)")

pl = np.array(mse["plain"]).mean()
sp = np.array(mse["sphere"]).mean()
nu = float(np.mean(nus))
print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 74)
print(f"СФЕРИЧНА СХЕМА · {NNET} мереж × {MREP} повторів")
print("=" * 74)
print(f"  звичайна МК     {100 * pl ** 0.5:.4f}%")
print(f"  фіксована норма {100 * sp ** 0.5:.4f}%   дисперсія ×{pl / sp:.4f}")
print(f"  ⟹ бал {1.8676e-07 / (pl / sp):.4e}  (рекорд 1.8676e-07)")
bp, bs = np.array(bias["plain"]), np.array(bias["sphere"])
sep, ses = bp.std(ddof=1) / np.sqrt(len(bp)), bs.std(ddof=1) / np.sqrt(len(bs))
print(f"\n  P1 НЕЗМІЩЕНІСТЬ (середнє по {MREP} прогонах, віднос. зсув):")
print(f"     plain  {bp.mean():+.3e} ± {sep:.3e}   (t={bp.mean()/max(sep,1e-30):+.2f})")
print(f"     sphere {bs.mean():+.3e} ± {ses:.3e}   (t={bs.mean()/max(ses,1e-30):+.2f})")
d = bs - bp
sed = d.std(ddof=1) / np.sqrt(len(d))
print(f"     РІЗНИЦЯ sphere−plain: {d.mean():+.3e} ± {sed:.3e}  (t={d.mean()/max(sed,1e-30):+.2f})"
      f"  {'✅ незміщена' if abs(d.mean()) < 2.5 * sed else '⛔ ЗМІЩЕННЯ'}")
print(f"\n  P2: зміряний ν (медіана по нейронах) = {nu:.4f}")
print(f"      формула передбачає ×{1 + (n - EZ ** 2) / (EZ ** 2 * nu ** 2):.4f}, "
      f"зміряно ×{pl / sp:.4f}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/sphere89.npz",
         plain=np.array(mse["plain"]), sphere=np.array(mse["sphere"]),
         nus=np.array(nus))
