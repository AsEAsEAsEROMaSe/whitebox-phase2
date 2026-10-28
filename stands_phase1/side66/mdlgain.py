"""РЕАЛЬНИЙ ВИГРАШ ВІД МОДЕЛЬНИХ `κ₃`,`κ₄` (не кореляція, а похибка). (С86)

## ЗВІДКИ
`srank.py`: `S_jk = κ₃(a_j,a_j,a_k)` НИЗЬКОРАНГОВА (r=1 несе 70% норми, на
глибині 94%), а модель `S ≈ α·σ²σᵀ` дає корел **0.845** з істинним `κ₃(h)`.
⚠️АЛЕ контроль (переставлена `S`) дав **+0.324** ⟹ корел тут завищує.
🎓 [[feedback-measure-the-gain-not-the-ceiling]] ⟹ міряємо ПОХИБКУ `E[ReLU]`.

## 📐 ЩО ДАЄ МОДЕЛЬ РАНГУ 1
`S ≈ α·σ²σᵀ` ⟹ (j,j,k)-член згортається у ДОБУТОК матвеків:
    `κ₃(h_i) ≈ 3α·( [(W²)ᵀσ²]_i·[Wᵀσ]_i − [(W³)ᵀσ³]_i )`
Аналогічно для `κ₄` через `Q_jk = κ₄(a_j,a_j,a_k,a_k) ≈ β·σ²σ²ᵀ`:
    `κ₄(h_i) ≈ 3β·( [(W²)ᵀσ²]_i² − [(W⁴)ᵀσ⁴]_i )`
⟹ 3–4 матвеки на шар ≈ **4e5 флопів**, ×32 = 1.3e7 — рівно бюджет лідера.
🔑 `σ` беремо з замикання: `perneuron` вимір B дав масштаб **1.0008**.

## 🎯 СХЕМИ, ЯКІ ПОРІВНЮЄМО (локальна похибка `E[ReLU]`, як у perneuron)
   G                — гаус (= наше замикання), база
   G+κ₃(мод)        — `κ₃` з моделі рангу 1, `α` ОРАКУЛЬНЕ (стеля моделі)
   G+κ₃(мод)+κ₄(мод)— обидва з моделі
   G+κ₃(іст)        — істинний `κ₃`   (референс ×2.755)
   G+κ₃+κ₄(іст)     — обидва істинні  (референс ×12.878)
⟹ питання: скільки з ×12.878 лишається, коли `κ` беруться з МОДЕЛІ.

⛔ПЛОМБИ:
 (1) `G+κ₃(іст)+κ₄(іст)` мусить відтворити ×12.878 з perneuron — інакше баг тут;
 (2) ⛔контроль: `α`,`β` від ЧУЖОЇ мережі ⟹ якщо виграш той самий, коефіцієнти
     універсальні (це ДОБРЕ); якщо виграш зникає — вони підганяються під мережу;
 (3) ⛔контроль: `σ` замінене на ВИПАДКОВИЙ вектор тієї ж норми ⟹ виграш
     мусить зникнути, інакше працює не структура, а масштаб.

📣 ПЕРЕДБАЧЕНО: `κ₃(мод)` дасть ×1.5–2.2 (з ×2.755), обидва разом ×3–6
   (з ×12.878). Головний ризик: `κ₃` і `κ₄` працюють ЛИШЕ РАЗОМ, тож похибка
   в одному з них ламає скорочення і виграш падає НЕЛІНІЙНО.
   Шанс, що обидва з моделі дадуть ≥×4: **40%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/mdlgain.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
BATCH = 1 << 12
L_, W_ = 32, 256
LAYERS = list(range(2, 32))
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
print(f'{len(FS)} мереж · 2^{LOG2N} · шари 2..31 · {DEV}\n', flush=True)

MODES = ('G', 'K3M', 'K34M', 'K3T', 'K34T', 'K3O', 'K34O', 'SVD1', 'SVD3', 'M3T4', 'T3M4', 'CTLS')


def corr_(x, y):
    u, v = x - x.mean(), y - y.mean()
    d = float(torch.sqrt((u * u).sum() * (v * v).sum()))
    return float((u * v).sum()) / d if d > 0 else 0.0
ERR = {k: np.zeros((len(FS), L_)) for k in MODES}
ALPHA = np.zeros((len(FS), L_))
BETA = np.zeros((len(FS), L_))
C3 = np.zeros((len(FS), L_))
C4 = np.zeros((len(FS), L_))
C3S = np.zeros((len(FS), L_))
C4S = np.zeros((len(FS), L_))


def predict(m, c2, g3, g4):
    s = torch.sqrt(torch.clamp(c2, min=1e-30))
    t = m / s
    ph = torch.exp(-0.5 * t * t) / SQ2PI
    Ph = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
    return s * (t * Ph + ph + (g3 / 6.0) * (-t * ph)
                + (g4 / 24.0) * ((t * t - 1.0) * ph))


for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        P1 = {i: torch.zeros(2, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        P2 = {i: torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        P3 = {i: torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        P4 = {i: torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        HM = {i: torch.zeros(4, W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        AM = {i: torch.zeros(W_, dtype=torch.float64, device=DEV) for i in LAYERS}
        g = torch.Generator(device=DEV)
        g.manual_seed(7100 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                L = q + 1
                if L in P1:
                    ad = a.double()
                    a2 = ad * ad
                    P1[L][0] += ad.sum(0)
                    P1[L][1] += a2.sum(0)
                    P2[L] += ad.T @ ad
                    P3[L] += a2.T @ ad
                    P4[L] += a2.T @ a2
                h = a @ W[q]
                if L in HM:
                    hd = h.double()
                    p = hd
                    for k in range(4):
                        HM[L][k] += p.sum(0)
                        p = p * hd
                a = torch.relu(h)
                if L in AM:
                    AM[L] += a.double().sum(0)
            d += b

        for i in LAYERS:
            m1, m2 = P1[i][0] / NS, P1[i][1] / NS
            Ejk, Ejjk, Ejjkk = P2[i] / NS, P3[i] / NS, P4[i] / NS
            va = m2 - m1 * m1
            sg = torch.sqrt(torch.clamp(va, min=1e-30))
            S = (Ejjk - 2.0 * m1[:, None] * Ejk - m2[:, None] * m1[None, :]
                 + 2.0 * (m1 * m1)[:, None] * m1[None, :])
            # Q_jk = κ₄(a_j,a_j,a_k,a_k) = E[ã_j²ã_k²] − E[ã_j²]E[ã_k²] − 2E[ã_jã_k]²
            Cjk = Ejk - torch.outer(m1, m1)
            Ejjkk_c = (Ejjkk - 2.0 * m1[:, None] * P3[i].T / NS
                       - 2.0 * m1[None, :] * P3[i] / NS
                       + (m1 * m1)[:, None] * m2[None, :]
                       + m2[:, None] * (m1 * m1)[None, :]
                       + 4.0 * torch.outer(m1, m1) * Ejk
                       - 3.0 * torch.outer(m1 * m1, m1 * m1))
            Q = Ejjkk_c - torch.outer(va, va) - 2.0 * Cjk * Cjk
            hm = [HM[i][k] / NS for k in range(4)]
            k3t = hm[2] - 3.0 * hm[0] * hm[1] + 2.0 * hm[0] ** 3
            mu4 = (hm[3] - 4.0 * hm[0] * hm[2] + 6.0 * hm[0] ** 2 * hm[1]
                   - 3.0 * hm[0] ** 4)
            c2h = hm[1] - hm[0] * hm[0]
            k4t = mu4 - 3.0 * c2h * c2h
            truth = AM[i] / NS
            Wn = W[i - 1].double()
            w2 = Wn * Wn

            def jjk(Sm):
                return 3.0 * ((w2 * (Sm @ Wn)).sum(0)
                              - (w2 * Wn * Sm.diagonal()[:, None]).sum(0))

            def jjkk(Qm):
                return 3.0 * ((w2 * (Qm @ w2)).sum(0)
                              - (w2 * w2 * Qm.diagonal()[:, None]).sum(0))

            def fit(M_, base):
                return float((M_ * base).sum()) / float((base * base).sum())

            b3 = torch.outer(sg * sg, sg)
            b4 = torch.outer(sg * sg, sg * sg)
            al, be = fit(S, b3), fit(Q, b4)
            ALPHA[n, i], BETA[n, i] = al, be
            sh = torch.sqrt(torch.clamp(c2h, min=1e-30))
            base = float(torch.sqrt((truth * truth).mean()))

            def err(g3, g4):
                p = predict(hm[0], c2h, g3, g4)
                return float(torch.sqrt(((p - truth) ** 2).mean())) / base

            z = torch.zeros_like(sh)
            k3m, k4m = jjk(al * b3), jjkk(be * b4)
            # 🔑 `α` фітився на МАТРИЦІ `S`, а важить похибка `κ₃` ПІСЛЯ згортки
            #    з вагами. Тут — ОРАКУЛЬНИЙ масштаб прямо в просторі `κ`: стеля
            #    моделі при ідеальному підборі одного числа на шар.
            def rescale(pred, true_):
                dn = float((pred * pred).sum())
                return pred * (float((pred * true_).sum()) / dn) if dn > 0 else pred
            k3o, k4o = rescale(k3m, k3t), rescale(k4m, k4t)
            # 🔑 ВИРІШАЛЬНИЙ ВИМІР: SVD-ранг-1 (корел 0.910 у srank) — це СТЕЛЯ
            #    будь-якої факторизації рангу 1, незалежно від того, чи вміємо ми
            #    здобути фактори. Якщо і вона дає <1 — гілка мертва по РАНГУ, а не
            #    по факторах. Якщо дає виграш — шукати правильні фактори варто.
            U3, s3v, V3 = torch.linalg.svd(S)
            U4, s4v, V4 = torch.linalg.svd(Q)
            k3s = jjk(torch.outer(U3[:, 0] * s3v[0], V3[0]))
            k4s = jjkk(torch.outer(U4[:, 0] * s4v[0], V4[0]))
            k3s3 = jjk((U3[:, :3] * s3v[:3]) @ V3[:3])
            k4s3 = jjkk((U4[:, :3] * s4v[:3]) @ V4[:3])
            # ⛔контроль(3): σ → випадковий вектор тієї ж норми
            rg = torch.rand(W_, dtype=torch.float64, device=DEV,
                            generator=torch.Generator(device=DEV).manual_seed(31 + n))
            rg = rg * float(sg.norm() / rg.norm())
            k3r = jjk(al * torch.outer(rg * rg, rg))
            k4r = jjkk(be * torch.outer(rg * rg, rg * rg))
            ERR['G'][n, i] = err(z, z)
            ERR['K3M'][n, i] = err(k3m / sh ** 3, z)
            ERR['K34M'][n, i] = err(k3m / sh ** 3, k4m / sh ** 4)
            ERR['K3T'][n, i] = err(k3t / sh ** 3, z)
            ERR['K34T'][n, i] = err(k3t / sh ** 3, k4t / sh ** 4)
            ERR['CTLS'][n, i] = err(k3r / sh ** 3, k4r / sh ** 4)
            ERR['K3O'][n, i] = err(k3o / sh ** 3, z)
            ERR['K34O'][n, i] = err(k3o / sh ** 3, k4o / sh ** 4)
            ERR['SVD1'][n, i] = err(k3s / sh ** 3, k4s / sh ** 4)
            ERR['SVD3'][n, i] = err(k3s3 / sh ** 3, k4s3 / sh ** 4)
            # 🔬 ДІАГНОСТИКА: хто саме винен — κ₃ чи κ₄. Змішуємо по одному.
            ERR['M3T4'][n, i] = err(rescale(k3s, k3t) / sh ** 3, k4t / sh ** 4)
            ERR['T3M4'][n, i] = err(k3t / sh ** 3, rescale(k4s, k4t) / sh ** 4)
            C3S[n, i] = corr_(k3s, k3t)
            C4S[n, i] = corr_(k4s, k4t)
            C3[n, i] = corr_(k3m, k3t)
            C4[n, i] = corr_(k4m, k4t)
    print(f'  … {n+1}/{len(FS)} seed {sd0} · шар16: G {ERR["G"][n,15]*100:.4f}% → '
          f'мод {ERR["K34M"][n,15]*100:.4f}% → іст {ERR["K34T"][n,15]*100:.4f}%',
          flush=True)
    del W, P1, P2, P3, P4, HM, AM
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/mdlgain.npz',
         **{f'e_{k}': ERR[k] for k in MODES}, alpha=ALPHA, beta=BETA)
sl = slice(4, 32)
print('\n' + '=' * 78)
print('📊 ЛОКАЛЬНА ПОХИБКА `E[ReLU]`, шари 5..31 (%)')
for k, nm in (('G', 'G — гаус (наше замикання)'), ('K3M', '🔑G+κ₃ З МОДЕЛІ'),
              ('K34M', '🔑G+κ₃+κ₄ З МОДЕЛІ'), ('K3T', 'G+κ₃ істинний'),
              ('K3O', '🔑G+κ₃ мод, ОРАКУЛ-масштаб'),
              ('K34O', '🔑G+κ₃+κ₄ мод, ОРАКУЛ-масштаб'),
              ('SVD1', '🔑🔑SVD-ранг-1 (стеля r=1)'),
              ('SVD3', '🔑🔑SVD-ранг-3 (стеля r=3)'),
              ('M3T4', '🔬κ₃ з SVD + κ₄ ІСТИННИЙ'),
              ('T3M4', '🔬κ₃ ІСТИННИЙ + κ₄ з SVD'),
              ('K34T', 'G+κ₃+κ₄ істинні'), ('CTLS', '⛔контроль: σ→випадк.')):
    v = ERR[k][:, sl].mean() * 100
    print(f'   {nm:<28s} {v:8.4f}%   ×{ERR["G"][:,sl].mean()/max(ERR[k][:,sl].mean(),1e-12):7.3f}')
print('─' * 78)
print(f'⛔ПЛОМБА(1): істинні κ мусять дати ≈×12.878 (perneuron) ⟹ '
      f'тут ×{ERR["G"][:,sl].mean()/ERR["K34T"][:,sl].mean():.3f}')
print(f'\n📌 α по шарах: {ALPHA[:,4:32].mean():.4f} ± {ALPHA[:,4:32].std():.4f} '
      f'(розкид по МЕРЕЖАХ у межах шару {ALPHA[:,4:32].std(0).mean():.4f})')
print(f'📌 β по шарах: {BETA[:,4:32].mean():.4f} ± {BETA[:,4:32].std():.4f} '
      f'(розкид по МЕРЕЖАХ {BETA[:,4:32].std(0).mean():.4f})')
print(f'\n📌 корел з істинними κ: модель σ²σᵀ — κ₃ {C3[:,4:32].mean():.4f} κ₄ {C4[:,4:32].mean():.4f}')
print(f'                        SVD-ранг-1  — κ₃ {C3S[:,4:32].mean():.4f} κ₄ {C4S[:,4:32].mean():.4f}')
print('\n🎯 Якщо ×(модель) ≥4 — гілка жива і коштує ~1.3e7 флопів на всю мережу.')
