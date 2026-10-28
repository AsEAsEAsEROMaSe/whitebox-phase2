"""ЗАМИКАННЯ НАВИВОРІТ: «КІНЕЦЬ ЯК ВХІД». (С85, ідея ЮЗЕРА)

## ПИТАННЯ ЮЗЕРА
«Що буде, якщо замикання перевернути так, ніби кінець це вхід.»

## ЧОМУ ЦЕ НЕ ЛИШЕ ДІАГНОСТИКА
1️⃣ ДІАГНОСТИКА: похибка замикання — властивість НАБОРУ ваг чи ПОРЯДКУ?
   С75 зміряла, що якір із шару 0 доходить до виходу у 8.1× меншим ⟹ похибку
   роблять ОСТАННІ шари. Якщо так, перевернутий порядок мусить дати ІНШУ
   похибку. Якщо ж вона та сама — «останні шари» були артефактом.
2️⃣ 🎯 ВАЖІЛЬ: якщо похибка на ПЕРЕВЕРНУТІЙ мережі корелює з нашою МІЖ
   мережами, ми отримуємо **ДЕТЕРМІНОВАНУ ознаку мережевої складової** —
   без жеребного шуму. Саме на пожеребності помер sub95
   ([[finding-holographic-reference-beam-dies-on-draw-noise]]: розкид по
   жеребах ×1.71 більший за мережевий, нахил ПРОТИЛЕЖНИЙ). Перевернуте
   замикання рахується з ваг ⟹ у кожному жеребі те саме.
   Стеля персоналізації масштабу підтверджена тричі: **×1.086–1.091**.

## ВАРІАНТИ МЕРЕЖІ (той самий набір ваг, різний порядок/орієнтація)
  FWD   `W_0 … W_31`            — як є (база)
  REV   `W_31 … W_0`            — зворотний порядок, «кінець став входом»
  REVT  `W_31ᵀ … W_0ᵀ`          — зворотний + транспонування = СПРЯЖЕНА мережа
  ⛔SHUF випадкова перестановка — контроль: справа в НАПРЯМКУ чи в будь-якій
        зміні порядку? Якщо SHUF дає те саме, що REV, — напрямок ні до чого.

## ЩО МІРЯЄМО
Для кожного варіанта: замикання `m_deep` і МК-істина тієї ж (переставленої)
мережі з вибірки 2^LOG2N ⟹ відносна похибка `‖m_deep − T‖₁/‖T‖₁`.
Далі — кореляція похибки кожного варіанта з ОРАКУЛЬНИМ масштабом `c*`
(готовий із `side66/layalign.py`, 48 мереж × 24 жереби).

⛔ПЛОМБА: похибка FWD мусить збігтися з відомою (`m_deep` проти банкової
істини ≈ 6e-03 відносних) — інакше стенд рахує не ту мережу.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: похибка REV буде ІНШОЮ за FWD (бо накопичення має
   напрямок), але кореляція з `c*` — слабка, ~0.2, бо перевернута мережа це
   вже ІНША функція, і її похибка не зобов'язана нести інформацію про нашу.
   Шанс, що дасть важіль: ~15%. Але коштує один прогін, і питання чисте.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/revclose.py [cuda:N] [NETS] [LOG2N]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import _phi, _Phi                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N
BATCH = 1 << 15
L_, W_ = 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж, вибірка 2^{LOG2N}, {DEV}\n', flush=True)


def closure(W, dev):
    """Замикання Мелера на ПОВНИХ вагах ⟹ m_deep."""
    m = torch.zeros(W[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(W[0].shape[0], dtype=torch.float64, device=dev)
    for i in range(len(W)):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        if i < len(W) - 1:
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            d = (mp*mp + s*s) * Ph + mp*s*ph - m*m
            C = C + torch.diag(d - C.diagonal())
    return m


def mc_truth(W, dev, ns, batch, seed):
    """МК-істина: E[ReLU(останній шар)] по N(0,I) на вході."""
    acc = torch.zeros(W[-1].shape[1], dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    done = 0
    while done < ns:
        b = min(batch, ns - done)
        a = torch.randn(b, W[0].shape[0], device=dev, generator=g)
        for i in range(len(W)):
            a = torch.relu(a @ W[i])
        acc += a.double().sum(0)
        done += b
    return acc / ns


def relerr(p, t):
    return float((p - t).abs().mean() / t.abs().mean())


# 🎯 ціль: оракульний масштаб c*, ГОТОВИЙ із layalign.py
LA = np.load('/mnt/win_d/llm/White-Box/arena62/logs/layalign.npz')
CSTAR_ALL = LA['PT'] / (2.0 * LA['PP'])

VAR = ['FWD', 'REV', 'REVT', '⛔SHUF']
ERR = {v: np.zeros(len(FS)) for v in VAR}
ERRB = np.zeros(len(FS))                 # FWD проти БАНКОВОЇ істини (пломба)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        gg = torch.Generator(); gg.manual_seed(777 + n)
        perm = torch.randperm(L_, generator=gg).tolist()
        SETS = {'FWD': W,
                'REV': W[::-1],
                'REVT': [w.T.contiguous() for w in W[::-1]],
                '⛔SHUF': [W[j] for j in perm]}
        for v in VAR:
            Ws = SETS[v]
            md = closure(Ws, DEV)
            tt = mc_truth(Ws, DEV, NS, BATCH, 4242 + n)
            ERR[v][n] = relerr(md, tt)
            if v == 'FWD':
                b1, b2 = TRUTH2[sd0]
                T = 0.5*(torch.tensor(b1, dtype=torch.float64, device=DEV)
                         + torch.tensor(b2, dtype=torch.float64, device=DEV))
                ERRB[n] = relerr(md, T)
        print(f'  … {n+1}/{len(FS)} seed {sd0} · ' +
              ' '.join(f'{v}={ERR[v][n]:.4f}' for v in VAR) +
              f' · ⛔ПЛОМБА FWD-банк={ERRB[n]:.4f}', flush=True)
    del W, SETS
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/revclose.npz',
         ERRB=ERRB, **ERR)
cstar = CSTAR_ALL[:len(FS)]


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


print(f'\n⛔ПЛОМБА: FWD проти МК-істини {ERR["FWD"].mean():.5f} vs проти '
      f'БАНКОВОЇ {ERRB.mean():.5f} · розбіжність '
      f'{abs(ERR["FWD"].mean()-ERRB.mean())/ERRB.mean()*100:.1f}%')
print(f'\n{"варіант":>10s} {"похибка замик.":>15s} {"розкид":>10s} '
      f'{"корел з FWD":>12s} {"корел з c*":>11s}')
print('─' * 62)
for v in VAR:
    print(f'{v:>10s} {ERR[v].mean():15.5f} {ERR[v].std():10.5f} '
          f'{corr(ERR[v], ERR["FWD"]):12.4f} {corr(ERR[v], cstar):11.4f}')
print('─' * 62)
print(f'{"c* сам":>10s} {cstar.mean():15.6f} {cstar.std():10.6f}')
print('\n🎯 ЧИТАТИ ТАК:')
print('   похибка REV ≈ FWD  ⟹ накопичення СИМЕТРИЧНЕ, порядок неважливий;')
print('   похибка REV ≠ FWD, а ⛔SHUF ≈ REV ⟹ справа не в НАПРЯМКУ, а в будь-якій')
print('     перестановці ⟹ «останні шари роблять похибку» — артефакт;')
print('   |корел REV↔c*| ≥ 0.3 ⟹ 🎯 ДЕТЕРМІНОВАНА ознака мережі, гілка ЖИВА.')
