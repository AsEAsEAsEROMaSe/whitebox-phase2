"""ТОЧНЕ ПОДАННЯ `κ₃`,`κ₄` ЧЕРЕЗ ТЕНЗОР РАНГУ `R`. (С86)

## ЗВІДКИ — ВКАЗІВКА ЮЗЕРА
«Копай, але хочу, щоб ти не навмання формули придумував, а щоб виводив їх
математично.» Плюс `side66/invar.py`: еф. ранг коваріації `h` падає 127.6 → 3.2.

## 📐 ВИВЕДЕННЯ (точне, єдине припущення — ранг)
Кумулянти лінійних комбінацій, ТОЧНО:
    `κ₃(h_i) = Σ_{jkl} W_ji W_ki W_li · T_jkl`,  `T_jkl = κ₃(a_j,a_k,a_l)`   (n³)
Припущення: `ã = B u`, `B ∈ R^{n×R}`, `u ∈ R^R` центрований (при `R=n` — тотожність).
Кумулянти МУЛЬТИЛІНІЙНІ ⟹
    `T_jkl = Σ_{rst} B_jr B_ks B_lt · M_rst`,   `M_rst = κ₃(u_r,u_s,u_t)`    (R³)
Підстановка і зміна порядку сум:
    `κ₃(h_i) = Σ_{rst} M_rst (Σ_j W_ji B_jr)(Σ_k W_ki B_ks)(Σ_l W_li B_lt)`
Внутрішні суми — одна матриця `G = WᵀB` (n×R):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    `κ₃(h_i) = Σ_{rst} M_rst · G_ir G_is G_it`
    `κ₄(h_i) = Σ_{rstv} N_rstv · G_ir G_is G_it G_iv`
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔑 ЦІНА: `G = WᵀB` = `2n²R` = 3.9e5 флопів/шар при R=3, згортка `n·R³` = 2e4
⟹ ×32 шари ≈ **1.3e7** — рівно бюджет лідера (`F` = 1.30e7).

## ⚠️ ЧИМ ЦЕ НЕ Є
Це НЕ факторизація `S`, яка провалилась
([[finding-edgeworth-gain-lives-in-a-fragile-cancellation]], ранг-1 ×0.835):
там наближалася ВЖЕ ЗГОРНУТА величина. Тут наближається САМ РОЗПОДІЛ, а
згортка з вагами робиться ТОЧНО. Різниця принципова: похибка входить один раз,
а не через згортку.

## 🎯 ЩО МІРЯЄМО
`κ₃(h_i)`,`κ₄(h_i)` за формулою при R = 1,2,3,5,8,16,32,256 проти ІСТИНИ (МК):
корел, масштаб, і ГОЛОВНЕ — локальна похибка `E[ReLU]` (не корел!).

⛔ПЛОМБИ:
 (1) 🔑R = n = 256 мусить дати ТОЧНИЙ збіг (машинний нуль) — формула ТОЧНА за
     побудовою, тож розбіжність = баг у коді, а не в теорії;
 (2) шар 1: `h` гаусів ⟹ `κ₃`,`κ₄` ≈ 0 при будь-якому R;
 (3) ⛔контроль: `M` від ЧУЖОЇ мережі ⟹ корел мусить упасти.

📣 ПЕРЕДБАЧЕНО: пломба (1) пройде (це алгебра). Питання лише в тому, як швидко
   падає точність з `R`. Еф. ранг 3.2 на ВИХОДІ, але 127 на вході ⟹ малий `R`
   працюватиме лише на ГЛИБОКИХ шарах. Шанс, що R=8 дає локальну похибку
   ≤×2 від істинних κ на шарах 20+: **45%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/tensor3.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N
BATCH = 1 << 12
L_, W_ = 32, 256
# 🚨 `N` має R⁴ елементів: R=256 ⟹ 32 ГБ. Тензорні ранги обмежені 32 (8.4 МБ),
#    а ПЛОМБА `R=n` робиться окремо і ЛИШЕ для `κ₃` (256³·8 = 134 МБ).
RANKS = (1, 2, 3, 5, 8, 16, 32, 64)
SEAL_LAYER = 20
LAYERS = [4, 12, 20, 28, 31]
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
SQ2PI = math.sqrt(2.0 * math.pi)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} · ранги {RANKS} · {DEV}\n', flush=True)

C3 = {r: np.zeros((len(FS), L_)) for r in RANKS}
C4 = {r: np.zeros((len(FS), L_)) for r in RANKS}
ERR = {r: np.zeros((len(FS), L_)) for r in RANKS}
ERRG = np.zeros((len(FS), L_))
ERRT = np.zeros((len(FS), L_))
SEAL = np.zeros(len(FS))
SEALS = np.zeros(len(FS))
NRANK = np.zeros(len(FS))


def corr(x, y):
    u, v = x - x.mean(), y - y.mean()
    d = float(torch.sqrt((u * u).sum() * (v * v).sum()))
    return float((u * v).sum()) / d if d > 0 else 0.0


for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        # ── прохід 1: середні та коваріація активацій кожного шару ──
        SA = {i: torch.zeros(W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        CA = {i: torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        SH = {i: torch.zeros(4, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        AM = {i: torch.zeros(W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        g = torch.Generator(device=DEV)
        g.manual_seed(9900 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                L = q + 1
                if L in SA:                      # активації шару L−1 = вхід у W[q]
                    ad = a.double()
                    SA[L] += ad.sum(0)
                    CA[L] += ad.T @ ad
                h = a @ W[q]
                if L in SH:
                    hd = h.double()
                    p = hd
                    for k in range(4):
                        SH[L][k] += p.sum(0)
                        p = p * hd
                a = torch.relu(h)
                if L in AM:
                    AM[L] += a.double().sum(0)
            d += b
        MA = {i: SA[i] / NS for i in LAYERS}
        BAS = {}
        for i in LAYERS:
            C = CA[i] / NS - torch.outer(MA[i], MA[i])
            ev, U = torch.linalg.eigh(C)
            idx = torch.argsort(ev, descending=True)
            ev, U = torch.clamp(ev[idx], min=0), U[:, idx]
            BAS[i] = (U, torch.sqrt(torch.clamp(ev, min=1e-300)))

        # ── прохід 2: тензори M,N у базисі кожного рангу ──
        MT = {(i, r): torch.zeros((r,) * 3, dtype=torch.float64, device=DEV)
              for i in LAYERS for r in RANKS}
        NT = {(i, r): torch.zeros((r,) * 4, dtype=torch.float64, device=DEV)
              for i in LAYERS for r in RANKS}
        CU = {(i, r): torch.zeros(r, r, dtype=torch.float64, device=DEV)
              for i in LAYERS for r in RANKS}
        g.manual_seed(9900 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                L = q + 1
                if L in SA:
                    ac = a.double() - MA[L]
                    U, sq = BAS[L]
                    rmax = max(RANKS)
                    uu = (ac @ U[:, :rmax]) / torch.clamp(sq[:rmax], min=1e-150)
                    for r in RANKS:
                        v = uu[:, :r]
                        CU[(L, r)] += v.T @ v
                        MT[(L, r)] += torch.einsum('br,bs,bt->rst', v, v, v)
                        NT[(L, r)] += torch.einsum('br,bs,bt,bv->rstv', v, v, v, v)
                a = torch.relu(a @ W[q])
            d += b

        # ⛔ПЛОМБА(1): R = n = 256 для `κ₃` на одному шарі — формула ТОЧНА за
        #    побудовою, тож збіг мусить бути машинний.
        # 🚨 R=n НЕ годиться: коваріація вироджена (еф. ранг ~5), і ділення на
        #    `√λ≈0` дає inf. ТОЧНЕ подання потребує R = ЧИСЛОВИЙ ранг, тобто
        #    стільки компонент, скільки їх із λ > eps·λ_max. Це і є пломба.
        Us_, sqs_ = BAS[SEAL_LAYER]
        nr = int((sqs_ ** 2 > 1e-13 * float(sqs_[0] ** 2)).sum())
        Us, sqs = Us_[:, :nr], sqs_[:nr]
        NRANK[n] = nr
        Mfull = torch.zeros((nr,) * 3, dtype=torch.float64, device=DEV)
        g.manual_seed(9900 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                if q + 1 == SEAL_LAYER:
                    ac = a.double() - MA[SEAL_LAYER]
                    v = (ac @ Us) / torch.clamp(sqs, min=1e-150)
                    Mfull += torch.einsum('br,bs,bt->rst', v, v, v)
                a = torch.relu(a @ W[q])
            d += b
        Mfull /= NS
        Bs = Us * sqs
        Gs = W[SEAL_LAYER - 1].double().T @ Bs
        k3_seal = torch.einsum('rst,ir,is,it->i', Mfull, Gs, Gs, Gs)
        _hm = [SH[SEAL_LAYER][k] / NS for k in range(4)]
        k3_ref = _hm[2] - 3.0 * _hm[0] * _hm[1] + 2.0 * _hm[0] ** 3
        SEAL[n] = corr(k3_seal, k3_ref)
        SEALS[n] = float((k3_seal * k3_ref).sum() / (k3_ref * k3_ref).sum())
        del Mfull
        torch.cuda.empty_cache()

        for i in LAYERS:
            U, sq = BAS[i]
            hm = [SH[i][k] / NS for k in range(4)]
            c2h = hm[1] - hm[0] * hm[0]
            k3t = hm[2] - 3.0 * hm[0] * hm[1] + 2.0 * hm[0] ** 3
            mu4 = (hm[3] - 4.0 * hm[0] * hm[2] + 6.0 * hm[0] ** 2 * hm[1]
                   - 3.0 * hm[0] ** 4)
            k4t = mu4 - 3.0 * c2h * c2h
            sh = torch.sqrt(torch.clamp(c2h, min=1e-30))
            truth = AM[i] / NS
            base = float(torch.sqrt((truth * truth).mean()))
            Wn = W[i - 1].double()

            def err(g3, g4):
                t = hm[0] / sh
                ph = torch.exp(-0.5 * t * t) / SQ2PI
                Ph = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
                p = sh * (t * Ph + ph + (g3 / 6.0) * (-t * ph)
                          + (g4 / 24.0) * ((t * t - 1.0) * ph))
                return float(torch.sqrt(((p - truth) ** 2).mean())) / base

            z = torch.zeros_like(sh)
            ERRG[n, i] = err(z, z)
            ERRT[n, i] = err(k3t / sh ** 3, k4t / sh ** 4)
            for r in RANKS:
                M3 = MT[(i, r)] / NS
                Cv = CU[(i, r)] / NS
                # κ₄ з моменту: κ₄=μ₄−Σ(3 пари) для центрованих
                M4 = NT[(i, r)] / NS
                N4 = (M4
                      - torch.einsum('rs,tv->rstv', Cv, Cv)
                      - torch.einsum('rt,sv->rstv', Cv, Cv)
                      - torch.einsum('rv,st->rstv', Cv, Cv))
                B = U[:, :r] * sq[:r]
                G = Wn.T @ B                        # (n, r)
                k3p = torch.einsum('rst,ir,is,it->i', M3, G, G, G)
                k4p = torch.einsum('rstv,ir,is,it,iv->i', N4, G, G, G, G)
                C3[r][n, i] = corr(k3p, k3t)
                C4[r][n, i] = corr(k4p, k4t)
                ERR[r][n, i] = err(k3p / sh ** 3, k4p / sh ** 4)
    print(f'  … {n+1}/{len(FS)} seed {sd0} · ⛔ПЛОМБА(1) R=n=256, κ₃, шар {SEAL_LAYER}: '
          f'числовий ранг {NRANK[n]:.0f} · корел {SEAL[n]:.8f} '
          f'масштаб {SEALS[n]:.8f} (мусить 1.0)', flush=True)
    del W, SA, CA, SH, AM, MT, NT, CU
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/tensor3.npz', seal=SEAL, seals=SEALS, nrank=NRANK,
         **{f'c3_{r}': C3[r] for r in RANKS},
         **{f'c4_{r}': C4[r] for r in RANKS},
         **{f'e_{r}': ERR[r] for r in RANKS}, eg=ERRG, et=ERRT)
sel = [i for i in LAYERS if i >= 16]
print('\n' + '=' * 86)
print(f'⛔ПЛОМБА(1) R = ЧИСЛОВИЙ ранг = {NRANK.mean():.0f} (κ₃, шар {SEAL_LAYER}):')
print(f'   корел = {SEAL.mean():.8f} · масштаб = {SEALS.mean():.8f}  '
      f'{"✅" if abs(SEAL.mean()-1) < 1e-4 else "🚨 БАГ У КОДІ"}')
print('\n📊 КОРЕЛ І ЛОКАЛЬНА ПОХИБКА vs РАНГ (шари ≥16)')
print(f'{"R":>5s} {"корел κ₃":>10s} {"корел κ₄":>10s} {"похибка %":>11s} {"× проти гауса":>14s}')
print('─' * 86)
eg = np.mean([ERRG[:, i] for i in sel])
for r in RANKS:
    e = np.mean([ERR[r][:, i] for i in sel])
    print(f'{r:5d} {np.mean([C3[r][:,i] for i in sel]):10.4f} '
          f'{np.mean([C4[r][:,i] for i in sel]):10.4f} {e*100:11.4f} '
          f'{eg/max(e,1e-12):14.3f}')
print('─' * 86)
print(f'{"гаус":>5s} {"—":>10s} {"—":>10s} {eg*100:11.4f} {1.0:14.3f}')
print(f'{"істина":>5s} {1.0:10.4f} {1.0:10.4f} '
      f'{np.mean([ERRT[:,i] for i in sel])*100:11.4f} '
      f'{eg/np.mean([ERRT[:,i] for i in sel]):14.3f}')
print('\n🎯 ЧИТАТИ: при якому R похибка наближається до «істина». Якщо R≈3..8 —')
print('   гілка жива і коштує ~1.3e7 флопів. Якщо треба R≫32 — мертва.')
