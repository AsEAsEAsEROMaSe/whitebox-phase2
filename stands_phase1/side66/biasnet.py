"""ЧИ ДАЄ НАВЧЕНА МОДЕЛЬ ЩОСЬ ПОНАД ЛІНІЙНУ НА ТИХ САМИХ ОЗНАКАХ?

ЗВІДКИ. Ідея юзера «дати це вивчити мережі» — правильна за адресою, якщо
годувати ВАГИ, а не графіки ([[finding-running-mean-path-carries-no-information]]).
Мішень: зміщення замикання `b_ℓ = m_істина,ℓ − m_замик,ℓ` — ДЕТЕРМІНОВАНА
функція ваг, без шуму в цілі. С70 зміряла `R²_oos = 0.792` на 8 ручних
ознаках (константа сама дає 0.641, ⛔roll 0.627) і вивела прайс-лист
`R²=0.90 → 1.54×`, `R²=0.95 → 1.99×`, оракул 7.185× ⟹ бал 2.85e-08.

🚨 ЩО САМЕ ТУТ ПЕРЕВІРЯЄТЬСЯ, І ЧОГО ТУТ НЕМА. Це НЕ вимір виграшу в балі.
Це найдешевший відсіювальний вимір: чи є в цілі НЕЛІНІЙНІСТЬ, яку ручна
лінійна регресія не бачить. Якщо MLP не перегнав лінійну — «навчити модель»
мертве на вході, і будувати дорогу гілку не треба.

🚨 НОРМА. Балу важить не `Σb²`, а `‖Mᵀb‖²` (адьойнт має еф.ранг ≈8, тому СТАЛА
складова `b` складається КОГЕРЕНТНО). Тут міряємо в ПРОСТІЙ нормі — це
свідоме спрощення відсіювального кроку: нелінійність, якої нема в простій
нормі, не з'явиться і в правильній.

════════ ОЗНАКИ (усі безрозмірні, усі з ВАГ + ЗАМИКАННЯ, істини нема) ════════
  t = m_замик/s · Φ(t) · φ(t) · s/сер(s) · q/s² · ‖w_j‖²/сер
  частка «своєї» дисперсії (діаг проти повного `wᵀCw`) · номер шару/L

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 лінійна відтворить С70: `R²_oos` близько 0.75-0.80 (константа ~0.64);
📣 MLP додасть **≤0.03** — ціль майже лінійна за цими ознаками;
🟢 ЖИВЕ, якщо MLP − ЛІН ≥ 0.05 (тоді нелінійність реальна і гілку варто гнати).
🚨 ПЛОМБИ
  ✅ лише КОНСТАНТА має дати ≈0.64 (звірка з С70);
  ⛔ ROLL: ознаки від СУСІДНЬОЇ мережі ⟹ має впасти до рівня константи;
  ✅ усе OOS **ПО МЕРЕЖАХ** (не по нейронах — нейрони однієї мережі залежні).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts, W_            # noqa: E402
from jitprof import truth_means                                    # noqa: E402
from jitanim import bank_net                                       # noqa: E402
from q31dir import _phi, _Phi                                      # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
LAYERS = tuple(range(1, 9))


def feats_and_target(wc, dev, mt):
    """Ознаки нейрона й ціль `b/s` для шарів `LAYERS`."""
    d = len(wc)
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    X, Y = [], []
    for i in range(d):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        # частка «своєї» дисперсії: діагональний внесок проти повного
        diagpart = (w * w * C.diagonal()[:, None]).sum(0)
        frac = diagpart / cp.diagonal().clamp_min(1e-30)
        wn = (w * w).sum(0)
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(
            ph / s, ph / s)
        dd = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(dd - C.diagonal())
        q = dd + m * m
        if i in LAYERS:
            f = torch.stack([t, Ph, ph, s / s.mean(),
                             q / (s * s).clamp_min(1e-30),
                             wn / wn.mean(), frac,
                             torch.full_like(t, i / float(d))], 1)
            X.append(f.cpu().numpy())
            Y.append(((mt[i] - m) / s).cpu().numpy())
    return np.concatenate(X), np.concatenate(Y)


def r2(y, p):
    return float(1.0 - ((y - p) ** 2).mean() / max(y.var(), 1e-300))


def fit_lin(Xtr, ytr, Xte):
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    B = np.hstack([Xte, np.ones((len(Xte), 1))])
    b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return B @ b


def fit_mlp(Xtr, ytr, Xte, dev, h=64, ep=400, seed=0):
    torch.manual_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)[:, None]
    xe = torch.tensor(Xte, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], h), torch.nn.GELU(),
        torch.nn.Linear(h, h), torch.nn.GELU(),
        torch.nn.Linear(h, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    for _ in range(ep):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        return net(xe).squeeze(1).cpu().numpy()


def main(nets=16, dev=DEV):
    print('📣 ПЕРЕДБАЧЕНО: лінійна 0.75-0.80 (константа ~0.64), '
          'MLP додасть ≤0.03')
    print('🟢 ЖИВЕ при MLP − ЛІН ≥ 0.05\n', flush=True)
    XS, YS = [], []
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            mt, _ = truth_means(wc, dev, 310_000 + 877 * n)
            X, Y = feats_and_target(wc, dev, mt)
        XS.append(X)
        YS.append(Y)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    mu = np.concatenate(XS).mean(0)
    sd = np.concatenate(XS).std(0)
    sd[sd == 0] = 1.0
    XS = [(x - mu) / sd for x in XS]
    rl, rm_, rc, rr = [], [], [], []
    for i in range(nets):
        tr = [j for j in range(nets) if j != i]
        Xtr = np.concatenate([XS[j] for j in tr])
        ytr = np.concatenate([YS[j] for j in tr])
        rc.append(r2(YS[i], np.full_like(YS[i], ytr.mean())))
        rl.append(r2(YS[i], fit_lin(Xtr, ytr, XS[i])))
        rm_.append(r2(YS[i], fit_mlp(Xtr, ytr, XS[i], dev, seed=i)))
        j = (i + 1) % nets
        k = min(len(XS[i]), len(XS[j]))
        rr.append(r2(YS[i][:k], fit_lin(Xtr, ytr, XS[j][:k])))

    print(f'\n{nets} мереж, OOS ПО МЕРЕЖАХ, шари {LAYERS[0]}..{LAYERS[-1]}')
    print(f'{"модель":>26} {"R² OOS":>9}')
    for nm, v in (('лише КОНСТАНТА', rc), ('ЛІНІЙНА (8 ознак)', rl),
                  ('MLP 8→64→64→1', rm_),
                  ('⛔ROLL ознаки сусідньої мережі', rr)):
        print(f'{nm:>26} {np.mean(v):>9.4f}')
    g = np.mean(rm_) - np.mean(rl)
    print(f'\n🔑 MLP − ЛІН = {g:+.4f}  (🟢гейт +0.05)')
    print(f'   ВИРОК: {"ЖИВЕ — нелінійність реальна" if g >= 0.05 else "МЕРТВО — ціль лінійна за цими ознаками"}')
    print('📊 БАЗОВИЙ БАЛ 1.954e-07 · лідер 8e-9 ⟹ 24.4× ⟹ K≈29 на карті призу')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
