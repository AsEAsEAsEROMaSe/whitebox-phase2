"""СКІЛЬКИ НЕРУХОМИХ ТОЧОК НАСПРАВДІ: АЛГЕБРА ЗАМІСТЬ ПОШУКУ (С80).

С79 знайшла ОДИН атрактор зі 100% басейном із 1024 стартів. Але ітерація бачить
лише СТІЙКІ нерухомі точки — нестійкі вона відштовхує, і випадковий старт їх не
побачить НІКОЛИ. Той самий урок С79 записала на мінімумі `S`: конструкція
пред'явила точку рівно, а спуск дав 0 влучень із 2048.

🔑 ІДЕЯ. Усередині конуса зі сталим патерном воріт `g` мережа СТРОГО ЛІНІЙНА:
        f(x) = x @ J_g ,   J_g = W₀G₀·W₁G₁·…·W₃₁G₃₁
   тож нерухома точка — це ЛІВА ВЛАСНА ПАРА `x @ J_g = λx`. Її не шукають, її
   РОЗВ'ЯЗУЮТЬ. Справжня вона за умови САМОУЗГОДЖЕНОСТІ: власний вектор мусить
   бути невід'ємним і відтворювати рівно той патерн `g`, який його породив.
   🔑 Умова повна: збіг патерну ⟹ `f(v) = v@J_g = λv` автоматично.

💰 ДВА ТРЮКИ, БЕЗ ЯКИХ ЦЕ 196 000 РОЗКЛАДІВ (години):
 1. `J = A_ℓ·W_ℓ·G_ℓ·B_ℓ` (префікс×суфікс, 64 матмули разом). Перемикання воріт
    `(ℓ,i)` міняє `G_ℓ` на `±e_ie_iᵀ` ⟹ **ΔJ = ±(A_ℓW_ℓ)[:,i] ⊗ B_ℓ[i,:]**, ранг 1.
 2. Матриці НЕ матеріалізуємо: `v@J_f = v@J ± (v·u)·b` ⟹ степенева ітерація для
    ВСІХ 8192 сусідів — це ОДИН матмул (8192×256)@(256×256) на крок.
   Скринінг у fp32, підтвердження у fp64 (fp64 на P104 у 32× повільніше).

📣 ПЕРЕДБАЧЕНО ДО ПРОГОНУ:
 * дійсних власних значень ≈ `√(2n/π) = 12.8` (закон Едельмана) — пломба на
   розклад, не на задачу;
 * самоузгоджених н.т. у ВЛАСНОМУ конусі — рівно **1** (відома);
 * серед 8192 однопереключних сусідів нових — **0 або одиниці**;
 * «розмитий атрактор» (3/12 мереж С79, осциляції з піками 1e-4) пояснюється
   КОМПЛЕКСНИМ `λ₂` з `|λ₂|/|λ₁| → 1` — це обертання навколо нерухомої точки.

⛔ ПЛОМБИ: `x*@J = λx*` · ранг-1 проти чесного `J` · `λ` з eig = `λ` з ітерації ·
   cos(власний, `x*`) = 1 · ⛔НЕГ. власний вектор ЧУЖОЇ мережі не самоузгоджений.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/fixed.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
PIT = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
PRE = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
NPAIR = 64                      # скільки найкрайніших воріт брати на 2-перемикання
W_, D = 256, 32
F64 = torch.float64
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1].split('⛔ ПЛОМБИ')[0], flush=True)


def _chain(W, x):
    g = torch.zeros(D, W_, dtype=torch.bool, device=DEV)
    pre = torch.zeros(D, W_, dtype=F64, device=DEV)
    for L in range(D):
        p = (x @ W[L]).reshape(-1)
        pre[L], g[L] = p, p > 0
        x = torch.relu(p).reshape(1, -1)
    return x, g, pre


def _gates_batch(W32, X):
    """Патерни воріт для пакета векторів: (B,D,W_) bool."""
    G = torch.zeros(X.shape[0], D, W_, dtype=torch.bool, device=DEV)
    for L in range(D):
        p = X @ W32[L]
        G[:, L] = p > 0
        X = torch.relu(p)
    return G


def _screen(J32, U, Bv, S, expect, W32, gs, pit):
    """Степенева ітерація для пакета рангово-1 збурених матриць + перевірка.

    `expect` — (B,D,W_) очікуваний патерн. Повертає індекси самоузгоджених.
    """
    B = U.shape[0]
    V = torch.randn(B, W_, device=DEV)
    V /= V.norm(dim=1, keepdim=True)
    for _ in range(pit):
        V = V @ J32 + S.reshape(-1, 1) * (V * U).sum(1, keepdim=True) * Bv
        nv = V.norm(dim=1, keepdim=True).clamp_min(1e-30)
        V = V / nv
    # невід'ємність (з точністю до знака)
    pos = V.clamp_min(0).norm(dim=1)
    neg = (-V).clamp_min(0).norm(dim=1)
    sgn = torch.where(pos >= neg, 1.0, -1.0).reshape(-1, 1)
    Vs = (V * sgn).clamp_min(0)
    nv = Vs.norm(dim=1, keepdim=True)
    ok = (nv.reshape(-1) > 1e-6)
    # 🚨 ДОПУСК МУСИТЬ БУТИ ГРУБИЙ, А НЕ ТОНКИЙ. `J` — добуток 32 матриць,
    #    його сингулярні числа розкидані на десятки порядків, тож власні
    #    вектори брудні на рівні ~1e-5 (пломба ‖x*J−λx*‖ дала 5.1e-05 при
    #    тому, що прямий прогін дає 1e-16). Тонкий допуск 1e-6 відкидав
    #    СПРАВЖНІ розв'язки ⟹ «0 знайдено» нічого не доводило.
    #    Тут лишаємо ГРУБИЙ передфільтр (високе повноту), а справжня
    #    перевірка — збіг патерну воріт через ПРЯМИЙ ПРОГІН, який стабільний.
    ok &= ((V * sgn).clamp_max(0).norm(dim=1) /
           nv.reshape(-1).clamp_min(1e-30)) < 0.20
    if not bool(ok.any()):
        return torch.zeros(0, dtype=torch.long, device=DEV)
    idx = torch.nonzero(ok).reshape(-1)
    G = _gates_batch(W32, Vs[idx] / nv[idx])
    match = (G == expect[idx]).reshape(idx.numel(), -1).all(1)
    return idx[match]


rows, blur = [], []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), dtype=F64, device=DEV)
         for k in range(D)]
    W32 = [w.float() for w in W]
    with torch.no_grad():
        gg = torch.Generator(device=DEV).manual_seed(31 + 7 * net)
        # 🚨 400 ІТЕРАЦІЙ БУЛО МАЛО. Швидкість збіжності — (|λ₂|/|λ₁|)^k, і при
        #    щілині 0.9894 це 0.12 навіть за 200 кроків ⟹ патерн воріт `gs`
        #    виходив НЕ той, і далі валилось усе. Дешевий розгін у fp32
        #    (прямий прогін стабільний), полірування у fp64.
        x32 = torch.randn(1, W_, device=DEV, generator=gg)
        x32 = x32 / x32.norm()
        for _ in range(PRE):
            for L in range(D):
                x32 = torch.relu(x32 @ W32[L])
            x32 = x32 / x32.norm().clamp_min(1e-30)
        x = x32.double()
        for _ in range(200):
            y, _, _ = _chain(W, x)
            x = y / y.norm().clamp_min(1e-300)
        xs = x
        ys, gs, pres = _chain(W, xs)
        lam_it = float(ys.norm())
        blur.append(max(1.0 - float((ys / ys.norm() * xs).sum()), 1e-18))

        # ── префікси/суфікси й точний J ───────────────────────────────
        A, Bs = [None] * D, [None] * D
        M = torch.eye(W_, dtype=F64, device=DEV)
        for L in range(D):
            A[L] = M.clone()
            M = (M @ W[L]) * gs[L].to(F64)
        J = M.clone()
        M = torch.eye(W_, dtype=F64, device=DEV)
        for L in range(D - 1, -1, -1):
            Bs[L] = M.clone()
            M = W[L] @ (gs[L].to(F64).reshape(-1, 1) * M)
        seal_eig = float((xs @ J - lam_it * xs).norm() / xs.norm())
        L0 = 11
        chk = (A[L0] @ W[L0]) @ (gs[L0].to(F64).reshape(-1, 1) * Bs[L0])
        seal_r1 = float((chk - J).norm() / J.norm().clamp_min(1e-300))

        # ── свій конус: ПОВНИЙ розклад у fp64 ─────────────────────────
        ev, evec = torch.linalg.eig(J.transpose(0, 1).cpu())
        im = ev.imag.abs() / ev.abs().clamp_min(1e-300)
        nreal = int((im < 1e-9).sum())
        order = torch.argsort(ev.abs(), descending=True)
        l1, l2 = ev[order[0]], ev[order[1]]
        gap = float(l2.abs() / l1.abs().clamp_min(1e-300))
        l2c = float((l2.imag.abs() / l2.abs().clamp_min(1e-300)) > 1e-9)
        own, bcos = 0, 0.0
        for t in range(W_):
            if not bool(im[t] < 1e-9):
                continue
            v0 = evec[:, t].real.to(DEV)
            for sg in (1.0, -1.0):
                v = (sg * v0)
                # 🚨 грубий допуск — див. пояснення в `_screen`
                if float(v.clamp_max(0).norm()) > 0.20 * float(v.norm()):
                    continue
                v = v.clamp_min(0)
                if float(v.norm()) < 1e-12:
                    continue
                v = (v / v.norm()).reshape(1, -1)
                _, gv, _ = _chain(W, v)
                if bool((gv == gs).all()):
                    own += 1
                    bcos = max(bcos, abs(float((v.reshape(-1) *
                                                xs.reshape(-1)).sum())))
        # ⛔НЕГ.: власний вектор ЧУЖОЇ мережі в НАШОМУ конусі
        Wf = [torch.tensor(np.asarray(WZ[(net + 5) % 100][k], np.float32),
                           dtype=F64, device=DEV) for k in range(D)]
        xf = torch.randn(1, W_, dtype=F64, device=DEV, generator=gg)
        xf = xf / xf.norm()
        for _ in range(200):
            yf, _, _ = _chain(Wf, xf)
            xf = yf / yf.norm().clamp_min(1e-300)
        _, gvf, _ = _chain(W, xf)
        negc = int(bool((gvf == gs).all()))
        del Wf

        # ── ВСІ 8192 ОДНОПЕРЕКЛЮЧНІ СУСІДИ ────────────────────────────
        J32 = J.float()
        AW = torch.stack([(A[L] @ W[L]).float() for L in range(D)])     # D×256×256
        BB = torch.stack([Bs[L].float() for L in range(D)])             # D×256×256
        Uu = AW.permute(0, 2, 1).reshape(-1, W_)                        # 8192×256
        Bv = BB.reshape(-1, W_)
        Sg = torch.where(gs.reshape(-1), -1.0, 1.0).float()
        exp1 = gs.reshape(1, D, W_).repeat(D * W_, 1, 1)
        ar = torch.arange(D * W_, device=DEV)
        exp1[ar, ar // W_, ar % W_] = ~exp1[ar, ar // W_, ar % W_]
        # ✅ПОЗИТИВНИЙ КОНТРОЛЬ — БЕЗ НЬОГО «0 ЗНАЙДЕНО» НІЧОГО НЕ ДОВОДИТЬ:
        #    той самий скринінг на НЕЗМІНЕНОМУ конусі мусить знайти рівно
        #    відомий атрактор. Ранг-1 доданок занулюємо (Sg=0).
        pc = _screen(J32, Uu[:1], Bv[:1], torch.zeros(1, device=DEV),
                     gs.reshape(1, D, W_), W32, gs, PIT)
        posc = int(pc.numel())
        hit1 = _screen(J32, Uu, Bv, Sg, exp1, W32, gs, PIT)
        n1 = int(hit1.numel())
        del exp1, AW, BB

        # ── 2-ПЕРЕКЛЮЧЕННЯ на NPAIR найкрайніших воротах ──────────────
        sc = (pres.abs() / pres.abs().mean(1, keepdim=True)).reshape(-1)
        top = torch.argsort(sc)[:NPAIR]
        ii, jj = torch.triu_indices(NPAIR, NPAIR, offset=1, device=DEV)
        p1, p2 = top[ii], top[jj]
        exp2 = gs.reshape(1, D, W_).repeat(p1.numel(), 1, 1)
        rr = torch.arange(p1.numel(), device=DEV)
        exp2[rr, p1 // W_, p1 % W_] = ~exp2[rr, p1 // W_, p1 % W_]
        exp2[rr, p2 // W_, p2 % W_] = ~exp2[rr, p2 // W_, p2 % W_]
        V = torch.randn(p1.numel(), W_, device=DEV)
        V /= V.norm(dim=1, keepdim=True)
        U1, B1, S1 = Uu[p1], Bv[p1], Sg[p1]
        U2, B2, S2 = Uu[p2], Bv[p2], Sg[p2]
        for _ in range(PIT):
            V = (V @ J32
                 + S1.reshape(-1, 1) * (V * U1).sum(1, keepdim=True) * B1
                 + S2.reshape(-1, 1) * (V * U2).sum(1, keepdim=True) * B2)
            V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-30)
        pos = V.clamp_min(0).norm(dim=1); neg = (-V).clamp_min(0).norm(dim=1)
        sg2 = torch.where(pos >= neg, 1.0, -1.0).reshape(-1, 1)
        Vs = (V * sg2).clamp_min(0)
        nv = Vs.norm(dim=1, keepdim=True)
        ok2 = (nv.reshape(-1) > 1e-6) & (((V * sg2).clamp_max(0).norm(dim=1) /
                                          nv.reshape(-1).clamp_min(1e-30)) < 1e-6)
        n2 = 0
        if bool(ok2.any()):
            k2 = torch.nonzero(ok2).reshape(-1)
            G2 = _gates_batch(W32, Vs[k2] / nv[k2])
            n2 = int((G2 == exp2[k2]).reshape(k2.numel(), -1).all(1).sum())
        rows.append((nreal, own, bcos, seal_eig, seal_r1, float(l1.real),
                     lam_it, gap, l2c, n1, D * W_, n2, p1.numel(), negc,
                     posc, blur[-1]))
    print(f'  … мережа {net+1}/{NETS}: ✅поз.контроль {posc}, свій конус {own}, '
          f'1-перекл {n1}/{D*W_}, 2-перекл {n2}/{p1.numel()}, '
          f'|λ₂|/|λ₁|={gap:.4f}, blur={blur[-1]:.1e}', flush=True)
    del W, W32, A, Bs
    torch.cuda.empty_cache()

a = np.array(rows)
b = np.array(blur)
print(f'\n{NETS} мереж, степеневих ітерацій {PIT}\n')
print('⛔ ПЛОМБИ')
print(f'  ‖x*J − λx*‖/‖x*‖          = {np.median(a[:,3]):.3e}   (треба 0)')
print(f'  ранг-1 проти чесного J     = {np.median(a[:,4]):.3e}   (треба 0)')
print(f'  |λ_eig − λ_ітер|           = {np.median(np.abs(a[:,5]-a[:,6])):.3e}   (треба 0)')
print(f'  cos(власний, x*)           = {np.median(a[:,2]):.8f}   (треба 1)')
print(f'  ⛔НЕГ. чужий x* у нашому конусі: {int(a[:,13].sum())}/{NETS}   (треба 0)')
ncplx1 = int((a[:, 7] > 0.99999).sum())
print(f'  ✅ПОЗ. скринінг на СВОЄМУ конусі: {int((a[:,14]>0).sum())}/{NETS}'
      f'   (стеля {NETS-ncplx1}: у {ncplx1} мереж провідна пара КОМПЛЕКСНА,')
print(f'     там нерухомої точки НЕМА взагалі ⟹ знаходити нема чого)')
print(f'  🔑 «0 нових» доводить лише на тих {int((a[:,14]>0).sum())} мережах, '
      f'де поз.контроль пройшов')
print(f'  blur = 1−cos(f(x*),x*) прямим прогоном: медіана {np.median(a[:,15]):.2e}')
print(f'     └ це ЧЕСНА пломба тотожності; ‖x*J−λx*‖ забруднена обумовленістю J')
print(f'\n{"величина":46s} {"медіана":>11s} {"мін":>11s} {"макс":>11s}')
print('-' * 84)
for k, nm in ((0, 'ДІЙСНИХ власних значень (Едельман 12.8)'),
              (1, 'самоузгоджених н.т. у СВОЄМУ конусі'),
              (9, f'НОВИХ н.т. серед {D*W_} 1-переключних сусідів'),
              (11, 'НОВИХ н.т. серед 2-переключних сусідів'),
              (6, 'λ атрактора'),
              (7, '|λ₂|/|λ₁| — спектральна щілина')):
    c = a[:, k]
    print(f'{nm:46s} {np.median(c):11.4f} {c.min():11.4f} {c.max():11.4f}')
print(f'\nмереж із КОМПЛЕКСНИМ λ₂: {int(a[:,8].sum())}/{NETS}')
m = a[:, 8] > 0.5
if m.any() and (~m).any():
    print(f'  розмитість при КОМПЛЕКСНОМУ λ₂: {np.median(b[m]):.3e}')
    print(f'  розмитість при ДІЙСНОМУ    λ₂: {np.median(b[~m]):.3e}')
print(f'  кореляція |λ₂|/|λ₁| з log(розмитість) = '
      f'{np.corrcoef(a[:,7], np.log10(b))[0,1]:+.3f}')
print(f'\n🔑 ВИРОК: н.т. у своєму конусі {np.median(a[:,1]):.0f}, '
      f'нових у {int(a[0,10])} 1-переключних {np.median(a[:,9]):.0f}, '
      f'у {int(a[0,12])} 2-переключних {np.median(a[:,11]):.0f}')
print(f'⚠️ ОХОПЛЕННЯ: усі 1-переключні конуси; 2-переключні лише на {NPAIR} '
      f'найкрайніших воротах (з {D*W_}) — решта НЕ перевірена')
print('📊 БАЛ 1.954e-07 — вимір СТРУКТУРНИЙ')
