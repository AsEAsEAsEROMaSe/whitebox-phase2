"""ІНВАРІАНТ `RMS(γ₃) ≈ ⟨γ₄⟩` І ЧИ ЗВОДИТЬСЯ ВІН ДО ОДНІЄЇ ВЕЛИЧИНИ. (С86)

## ЗВІДКИ — ІДЕЯ ЮЗЕРА (метафора шарнірів)
«Вісь кожного шарніра спрямована рівно в центр — тому куля не рухається,
хоч би як шарніри йшли.» ⟹ шукати те, що ЗБЕРІГАЄТЬСЯ, коли решта росте.
У `perneuron.py` таке видно і я повз нього пройшов:
```
шар:        2       8      16      32
RMS γ₃:   0.0369  0.1768  0.3150  0.4382     ← росте у 12×
⟨γ₄⟩:     0.0376  0.1629  0.2821  0.4220     ← росте так само
відношення: 0.98    1.09    1.12    1.04     ← ТРИМАЄТЬСЯ
```

## 📐 ТЕОРІЯ, ЯКА ЦЕ ПОЯСНЮЄ (а не просто «зміряємо»)
Rank collapse: `h_i ≈ c_i·u(z)` ⟹ стандартизовані кумулянти масштабно
інваріантні і несуть лише ЗНАК:
```
γ₃(h_i) = sign(c_i)³·γ₃(u) = ±γ₃(u)   ⟹  RMS_i(γ₃) = |γ₃(u)|
γ₄(h_i) = sign(c_i)⁴·γ₄(u) =  γ₄(u)   ⟹  ⟨γ₄⟩_i    =  γ₄(u)
```
⟹ 🔑 інваріант зводиться до властивості ОДНІЄЇ величини: **`|γ₃(u)| = γ₄(u)`**.

## 🎯 ФАЛЬСИФІКОВАНІ ПРОГНОЗИ ЦІЄЇ ТЕОРІЇ
Якщо колапс справжній, то ОБОВ'ЯЗКОВО:
 (A) `|γ₃(h_i)|` майже ОДНАКОВЕ по нейронах ⟹ `std(|γ₃|)/mean(|γ₃|)` МАЛЕ;
 (B) `γ₄(h_i)` тим паче однакове ⟹ `std(γ₄)/⟨γ₄⟩` МАЛЕ;
 (C) `γ₃(h_i)` міняє знак разом із `c_i`, а `c_i ∝ t_i` ⟹ corr(γ₃, sign t) ВИСОКА;
 (D) головна компонента `u` має `|γ₃(u)| ≈ γ₄(u)` — і саме ці числа мусять
     збігтися з `RMS_i(γ₃)` та `⟨γ₄⟩_i`.
🚨 ЯКЩО (A) чи (B) ПРОВАЛЯТЬСЯ — інваріант НЕ від колапсу, і теорія хибна.

## 📊 ЩО ЩЕ МІРЯЄМО — ДВІ ЗОНИ
`perneuron` вимір B: діагональна рекурсія `κ₄` має масштаб 0.7475 на шарі 2 і
0.0246 на шарі 32 ⟹ на РАННІХ шарах активації майже НЕЗАЛЕЖНІ (там працює
теорія суми), на ГЛИБОКИХ — колапс (там працює теорія однієї величини).
⟹ міряємо ЕФЕКТИВНИЙ РАНГ коваріації `h` пошарово: він каже, ДЕ межа зон.
📌 якщо зони чисто розділені — це не апроксимація, а дві АСИМПТОТИКИ, кожна
   точна у своїй області, і їх можна зшити.

⛔ПЛОМБИ:
 (1) шар 1: `h₁=zᵀW₀` ТОЧНО гаусів ⟹ `γ₃`,`γ₄` ≈0 і еф.ранг ≈256;
 (2) ⛔контроль: `u` замінене на ВИПАДКОВИЙ напрямок ⟹ його `γ` не мусять
     збігатися з понейронними RMS/mean.

📣 ПЕРЕДБАЧЕНО: (A) і (B) пройдуть на ГЛИБОКИХ шарах (std/mean < 0.3) і
   ПРОВАЛЯТЬСЯ на ранніх. Еф.ранг впаде 256 → одиниці. Шанс, що `|γ₃(u)|`
   і `γ₄(u)` збіжаться з понейронними в межах 15% на шарах 20+: **50%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/invar.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 21
NS = 1 << LOG2N
BATCH = 1 << 12
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} · {DEV}\n', flush=True)

COLS = ('rms3', 'mabs3', 'sd3', 'm4', 'sd4', 'erank', 'g3u', 'g4u',
        'g3r', 'g4r', 'ctg')
R = {k: np.zeros((len(FS), L_)) for k in COLS}

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        # прохід 1: понейронні моменти h + коваріація h (для головного напрямку)
        S1 = [torch.zeros(4, W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        CV = [torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        g = torch.Generator(device=DEV)
        g.manual_seed(9300 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W[q]
                hd = h.double()
                p = hd
                for k in range(4):
                    S1[q][k] += p.sum(0)
                    p = p * hd
                CV[q] += hd.T @ hd
                a = torch.relu(h)
            d += b
        UV = []
        for q in range(L_):
            m1, m2 = S1[q][0] / NS, S1[q][1] / NS
            C = CV[q] / NS - torch.outer(m1, m1)
            ev, evec = torch.linalg.eigh(C)
            ev = torch.clamp(ev, min=0)
            R['erank'][n, q] = float(ev.sum() ** 2 / (ev * ev).sum())
            UV.append(evec[:, -1])                      # головний напрямок
        # прохід 2: моменти проєкції на головний напрямок (та на випадковий)
        S2 = torch.zeros(L_, 4, dtype=torch.float64, device=DEV)
        S3 = torch.zeros(L_, 4, dtype=torch.float64, device=DEV)
        rv = [torch.randn(W_, dtype=torch.float64, device=DEV,
                          generator=torch.Generator(device=DEV).manual_seed(77 + q))
              for q in range(L_)]
        rv = [v / v.norm() for v in rv]
        g.manual_seed(9300 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W[q]
                hd = h.double()
                for arr, vec in ((S2, UV[q]), (S3, rv[q])):
                    u = hd @ vec
                    p = u
                    for k in range(4):
                        arr[q, k] += p.sum()
                        p = p * u
                a = torch.relu(h)
            d += b

        def gam(m1, m2, m3, m4):
            c2 = m2 - m1 * m1
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
            c4 = mu4 - 3.0 * c2 * c2
            s = torch.sqrt(torch.clamp(c2, min=1e-30))
            return c3 / s ** 3, c4 / s ** 4

        for q in range(L_):
            g3, g4 = gam(*[S1[q][k] / NS for k in range(4)])
            a3 = g3.abs()
            R['rms3'][n, q] = float(torch.sqrt((g3 * g3).mean()))
            R['mabs3'][n, q] = float(a3.mean())
            R['sd3'][n, q] = float(a3.std())
            R['m4'][n, q] = float(g4.mean())
            R['sd4'][n, q] = float(g4.std())
            t = (S1[q][0] / NS) / torch.sqrt(torch.clamp(
                S1[q][1] / NS - (S1[q][0] / NS) ** 2, min=1e-30))
            u_, v_ = g3 - g3.mean(), torch.sign(t) - torch.sign(t).mean()
            dn = float(torch.sqrt((u_ * u_).sum() * (v_ * v_).sum()))
            R['ctg'][n, q] = float((u_ * v_).sum()) / dn if dn > 0 else 0.0
            for arr, k3, k4 in ((S2, 'g3u', 'g4u'), (S3, 'g3r', 'g4r')):
                a_, b_ = gam(*[arr[q, k] / NS for k in range(4)])
                R[k3][n, q] = float(a_)
                R[k4][n, q] = float(b_)
    print(f'  … {n+1}/{len(FS)} seed {sd0} · шар32: RMSγ₃ {R["rms3"][n,31]:.4f} '
          f'⟨γ₄⟩ {R["m4"][n,31]:.4f} · |γ₃(u)| {abs(R["g3u"][n,31]):.4f} '
          f'γ₄(u) {R["g4u"][n,31]:.4f} · еф.ранг {R["erank"][n,31]:.1f}', flush=True)
    del W, S1, CV
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/invar.npz', **R)
print('\n' + '=' * 100)
print('📊 ІНВАРІАНТ І ЙОГО МЕХАНІЗМ')
print(f'{"шар":>4s} {"RMSγ₃":>8s} {"⟨γ₄⟩":>8s} {"відн.":>7s} | '
      f'{"sd|γ₃|/⟨|γ₃|⟩":>13s} {"sdγ₄/⟨γ₄⟩":>10s} {"еф.ранг":>8s} | '
      f'{"|γ₃(u)|":>8s} {"γ₄(u)":>8s} {"⛔|γ₃(rnd)|":>11s}')
print('─' * 100)
for q in [0, 1, 3, 5, 7, 11, 15, 19, 23, 27, 31]:
    r3, m4 = R['rms3'][:, q].mean(), R['m4'][:, q].mean()
    print(f'{q+1:4d} {r3:8.4f} {m4:8.4f} {m4/max(r3,1e-12):7.3f} | '
          f'{R["sd3"][:,q].mean()/max(R["mabs3"][:,q].mean(),1e-12):13.3f} '
          f'{R["sd4"][:,q].mean()/max(abs(m4),1e-12):10.3f} '
          f'{R["erank"][:,q].mean():8.1f} | '
          f'{np.abs(R["g3u"][:,q]).mean():8.4f} {R["g4u"][:,q].mean():8.4f} '
          f'{np.abs(R["g3r"][:,q]).mean():11.4f}')
print('─' * 100)
sl = slice(19, 32)
print(f'ШАРИ 20..32:  ⟨γ₄⟩/RMSγ₃ = {(R["m4"][:,sl]/np.maximum(R["rms3"][:,sl],1e-12)).mean():.4f}'
      f' ± {(R["m4"][:,sl]/np.maximum(R["rms3"][:,sl],1e-12)).std():.4f}')
print(f'   ПРОГНОЗ (D): |γ₃(u)| ≈ RMSγ₃ ⟹ {np.abs(R["g3u"][:,sl]).mean():.4f} '
      f'проти {R["rms3"][:,sl].mean():.4f} · '
      f'γ₄(u) ≈ ⟨γ₄⟩ ⟹ {R["g4u"][:,sl].mean():.4f} проти {R["m4"][:,sl].mean():.4f}')
print(f'   ПРОГНОЗ теорії: |γ₃(u)| ≈ γ₄(u) ⟹ {np.abs(R["g3u"][:,sl]).mean():.4f} '
      f'проти {R["g4u"][:,sl].mean():.4f}')
print(f'⛔ПЛОМБА(1) шар 1: RMSγ₃ {R["rms3"][:,0].mean():.5f} · '
      f'⟨γ₄⟩ {R["m4"][:,0].mean():+.5f} · еф.ранг {R["erank"][:,0].mean():.1f}')
print(f'⛔ПЛОМБА(2) випадковий напрямок: |γ₃| {np.abs(R["g3r"][:,sl]).mean():.4f} '
      f'γ₄ {R["g4r"][:,sl].mean():.4f}')
print('\n🎯 (A),(B) малі ⟹ колапс справжній і теорія вірна. Великі ⟹ інваріант')
print('   має ІНШУ природу, і моє пояснення хибне.')
