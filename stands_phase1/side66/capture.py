"""КРИВА ЗАХОПЛЕННЯ: чи лежить E[o] у ПІДПРОСТОРІ, ЯКИЙ ЧИТАЄТЬСЯ З ВАГ? (С80)

Продовження атрактора. С79 зміряла, що атрактор дає НАПРЯМОК: cos(x*, E[o])=0.987.
Грепінг корпусу показав, що напрямок — НЕ дефіцитний ресурс (BBP: перекриття 0.90
з одного бюджету), тож «атрактор як напрямок» мертвий тим самим вироком.

🔑 АЛЕ ЛИШИЛАСЬ ОДНА ДІРА, І ВОНА ІНШОГО ТИПУ.
Усі попередні виміри брали напрямок, щоб ПЕРЕДБАЧИТИ похибку (і вмирали на
коефіцієнті). Тут — протилежна операція, «ЗНІМАТИ, НЕ ПЕРЕДБАЧАТИ»: якщо
справжнє середнє `µ` лежить у K-вимірному підпросторі `U`, який читається З ВАГ,
то проєкція `p̂ → P_U p̂` ВИКИДАЄ шум у решті 256−K напрямків і НЕ потребує
жодного коефіцієнта. Теорема тавтології не діє: `U` — інформація з `W`, не з
вибірки.
Підстава сподіватись: С79 зміряла колапс активацій PR 165.6 → **2.61** на шарі 31.
Якщо вихід реально живе у ~3 вимірах, проєкція мала б викинути ~99% шуму.

📐 АРИФМЕТИКА ВИРОКУ (виводиться ДО прогону):
    MSE_проєк = (1−f(K))·‖µ‖²  +  f_шум(K)·256·MSE
                 ^ЗМІЩЕННЯ, яке ми ВНЕСЛИ       ^шум, що лишився
  ‖µ‖² = 187.2, 256·MSE = 4.99e-04 ⟹ щоб зміщення не з'їло весь виграш, треба
                    **1 − f(K) < 2.7e-06**
  тобто підпростір з ваг мусить ловити середнє з точністю **0.00027%**, тоді як
  саме середнє ми вміємо рахувати лише на 0.02%. 🚨 Це пахне циклічністю, і
  вимір має сказати, чи вона фатальна.

📣 ПЕРЕДБАЧЕНО ДО ПРОГОНУ: при K=1 маємо f = cos² = 0.974 ⟹ 1−f = 0.026, це у
   9600× більше за поріг. Ставлю на те, що `1−f(K)` спадає ПОВІЛЬНО (як хвіст
   спектра), і порога не досягає ні при якому K < 256 — а K→256 робить проєкцію
   тотожною. Тобто виграш ≤ 1.0×. Але якщо колапс PR=2.61 «справжній», я маю
   помилитись уже на K=8.

⛔ ПЛОМБИ:
 * K=256 ⟹ f = 1.000000 РІВНО (повний базис) і виграш = 1.000×;
 * НЕГ.контроль: базис із ЧУЖОЇ мережі ⟹ f(K) ≈ K/256 (випадковий підпростір);
 * MC-оцінка мусить збігтися з T1 у межах свого шуму (перевірка конвенції істини).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/capture.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NMC = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
W_, D = 256, 32
KS = [1, 2, 4, 8, 16, 32, 64, 128, 192, 255, 256]

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
TZ = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
SEEDS = list(np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'))
IDX = [SEEDS.index(s) for s in TZ['seeds'].tolist()]
print(__doc__.split('📣')[1].split('⛔ ПЛОМБИ')[0], flush=True)


def _fwd(W, x):
    for L in range(D):
        x = torch.relu(x @ W[L])
    return x


def _attractor(W, iters=80):
    g = torch.Generator(device=DEV).manual_seed(7)
    x = torch.randn(1, W_, dtype=torch.float64, device=DEV, generator=g)
    for _ in range(iters):
        y = _fwd(W, x)
        x = y / y.norm().clamp_min(1e-300)
    return x


def _basis(W, x_star, gated):
    """W-only базис: праві сингулярні вектори лінеаризації `o = x @ J`,
    `J = W₀·diag(g₀)·W₁·diag(g₁)·…`, ворота взяті В АТРАКТОРІ.

    🚨 СИРИЙ ДОБУТОК 32 МАТРИЦЬ НЕ МОЖНА СИНГУЛЯРНО РОЗКЛАДАТИ: розкид
    сингулярних чисел добутку — це показники Ляпунова × 32, він виходить за
    діапазон fp64, і `torch.linalg.svd` чесно НЕ ЗБІГАЄТЬСЯ (попередження
    cusolver). Тому ортогональна ітерація з QR ПІСЛЯ КОЖНОГО ШАРУ: обумовленість
    лишається одиничною, а провідний підпростір той самий.
    Праві сингулярні вектори `J` = ліві сингулярні вектори `Jᵀ`, а
    `Jᵀ·Q = G₃₁W₃₁ᵀ(…(G₀W₀ᵀQ))` ⟹ шари йдуть у прямому порядку.
    """
    Q = torch.eye(W_, dtype=torch.float64, device=DEV)
    x = x_star
    for L in range(D):
        p = x @ W[L]
        Q = W[L].transpose(0, 1) @ Q
        if gated:
            Q = (p > 0).to(Q.dtype).reshape(-1, 1) * Q
        Q, _ = torch.linalg.qr(Q)
        x = torch.relu(p)
    return Q                                     # стовпці — ортонормований базис


def _capture(U, mu):
    """Накопичена частка енергії mu у перших K стовпцях U."""
    c = (mu @ U) ** 2
    return (torch.cumsum(c, 0) / c.sum()).cpu().numpy()


rows_g, rows_l, rows_n, rows_o = [], [], [], []
rows_ng, rows_no = [], []
gain_g, gain_o = [], []
mc_check, seal_full, mse_base = [], [], []

for j in range(min(NETS, len(IDX))):
    net = IDX[j]
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), dtype=torch.float64,
                      device=DEV) for k in range(D)]
    mu = torch.tensor(TZ['T1'][j], dtype=torch.float64, device=DEV)
    tgt = torch.tensor(TZ['T2'][j], dtype=torch.float64, device=DEV)
    with torch.no_grad():
        xs = _attractor(W)
        Ug = _basis(W, xs, True)
        Ul = _basis(W, xs, False)
        # ⛔НЕГ.КОНТРОЛЬ: базис із ЧУЖОЇ мережі
        Wf = [torch.tensor(np.asarray(WZ[IDX[(j + 7) % len(IDX)]][k], np.float32),
                           dtype=torch.float64, device=DEV) for k in range(D)]
        Un = _basis(Wf, _attractor(Wf), True)
        del Wf

        # MC-оцінка з незалежним жеребом (це наш «p̂» з ізотропнішим шумом,
        # тобто НАЙСПРИЯТЛИВІШИЙ для проєкції випадок)
        gmc = torch.Generator(device=DEV).manual_seed(90001 + 13 * j)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        done = 0
        while done < NMC:
            b = min(2048, NMC - done)
            z = torch.randn(b, W_, dtype=torch.float64, device=DEV, generator=gmc)
            acc += _fwd(W, z).sum(0)
            done += b
        phat = acc / NMC

        # ⛔ПЛОМБА КОНВЕНЦІЇ: MC мусить лягти на T1 у межах свого шуму
        mc_check.append(float((phat - mu).norm() / mu.norm()))

        # ОРАКУЛЬНИЙ базис: власні вектори E[o oᵀ] — форма розподілу виходу,
        # найкращий «структурний» підпростір, який узагалі буває
        zо = torch.randn(4096, W_, dtype=torch.float64, device=DEV, generator=gmc)
        O = _fwd(W, zо)
        _, Vo = torch.linalg.eigh(O.T @ O / O.shape[0])
        Uo = Vo.flip(1)

        rows_g.append(_capture(Ug, mu))
        rows_l.append(_capture(Ul, mu))
        rows_n.append(_capture(Un, mu))
        rows_o.append(_capture(Uo, mu))
        # 🔑 ГОЛОВНЕ: та сама крива, але для ШУМУ `n = p̂ − µ`. Проєкція виграє
        #    ЛИШЕ якщо шум захоплюється ГІРШЕ за сигнал (f_шум ≪ f_сигнал).
        rows_ng.append(_capture(Ug, phat - mu))
        rows_no.append(_capture(Uo, phat - mu))

        # ЧЕСНИЙ ВИГРАШ: MSE проєктованої оцінки проти НЕЗАЛЕЖНОЇ цілі T2
        base = float(((phat - tgt) ** 2).mean())
        mse_base.append(base)
        gg, go = [], []
        for K in KS:
            for U, out in ((Ug, gg), (Uo, go)):
                V = U[:, :K]
                pp = (phat @ V) @ V.transpose(0, 1)
                out.append(base / float(((pp - tgt) ** 2).mean()))
        gain_g.append(gg)
        gain_o.append(go)
        seal_full.append(gg[-1])
    if j % 8 == 0:
        print(f'  … мережа {j+1}/{min(NETS,len(IDX))}', flush=True)
    del W
    torch.cuda.empty_cache()

G = np.array(rows_g); L = np.array(rows_l)
N = np.array(rows_n); O = np.array(rows_o)
NG = np.array(rows_ng); NO = np.array(rows_no)
gg = np.array(gain_g); go = np.array(gain_o)
n = G.shape[0]
# 🚨 БУЛА ПОМИЛКА: `_capture` вертає криву по ВСІХ 256 K, тож стовпець для K
#    береться як [:, K−1], а не [:, індекс у KS] (інакше друкувались K=1..11
#    під підписами 1..256, і пломба K=256 показувала 0.031 замість 0).
CI = [K - 1 for K in KS]
mu2 = float((TZ['T1'][:n] ** 2).sum(1).mean())
mse = float(np.median(mse_base))
thr = 256 * mse / mu2

print(f'\n{n} мереж, MC N={NMC}, істина T1, ціль T2 (крос)\n')
print('⛔ ПЛОМБИ')
print(f'  K=256 (повний базис) виграш = {np.median(seal_full):.6f}   (треба 1.000000)')
print(f'  MC проти T1: відн. похибка = {np.median(mc_check)*100:.4f}%  (це шум самої MC)')
print(f'  повнота базису з ваг K=256: 1−f = {1-np.median(G[:,255]):.2e}  (треба 0)')
print(f'  НЕГ. чужий базис при K=32: f = {np.median(N[:,31]):.4f}'
      f'   (випадковий підпростір дав би {32/256:.4f})')

print(f'\n📐 ПОРІГ: щоб проєкція не внесла більше зміщення, ніж зняла шуму, треба')
print(f'   1−f(K) < 256·MSE/‖µ‖² = 256·{mse:.3e}/{mu2:.1f} = {thr:.3e}\n')
print(f'{"K":>5s} {"1−f СИГНАЛ ваги":>16s} {"1−f ШУМ ваги":>13s}'
      f' {"1−f СИГНАЛ орак":>16s} {"1−f ШУМ орак":>13s}'
      f' {"виграш ваги":>12s} {"виграш орак":>12s}')
print('-' * 92)
for i, K in enumerate(KS):
    c = CI[i]
    print(f'{K:5d} {1-np.median(G[:,c]):16.3e} {1-np.median(NG[:,c]):13.3e}'
          f' {1-np.median(O[:,c]):16.3e} {1-np.median(NO[:,c]):13.3e}'
          f' {np.median(gg[:,i]):12.4f} {np.median(go[:,i]):12.4f}')

best_g = float(np.median(gg, 0).max()); best_o = float(np.median(go, 0).max())
kg = KS[int(np.median(gg, 0).argmax())]; ko = KS[int(np.median(go, 0).argmax())]
print(f'\n🔑 НАЙКРАЩЕ: базис із ВАГ {best_g:.4f}× при K={kg} · '
      f'ОРАКУЛЬНИЙ базис {best_o:.4f}× при K={ko}')
ok = np.nonzero((1 - np.median(G, 0)) < thr)[0]
print(f'   порога 1−f<{thr:.2e} базис із ваг досягає при K = '
      f'{int(ok[0])+1 if len(ok) else "НІКОЛИ"}')
ok2 = np.nonzero((1 - np.median(O, 0)) < thr)[0]
print(f'   те саме для ОРАКУЛЬНОГО базису:            K = '
      f'{int(ok2[0])+1 if len(ok2) else "НІКОЛИ"}')
r = np.median(1 - NO, 0)[CI] / np.maximum(np.median(1 - O, 0)[CI], 1e-300)
print(f'\n🔑 МЕХАНІЗМ: (1−f_шум)/(1−f_сигнал) для оракула по K={KS[:6]} → '
      f'{np.array2string(r[:6], precision=1)}')
print('   якщо це ≫1 — шум лежить ПОЗА сигналом і проєкція його зніме;')
print('   якщо ≈1 — шум ділить підпростір із сигналом і знімати нема чого.')
print(f'\n📊 БАЛ 1.954e-07 ⟹ {1.954e-07/max(best_g,best_o):.4e} '
      f'(проєкція безкоштовна: базис читається з ваг)')
