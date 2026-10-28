"""ПОЖЕРЕБНИЙ ЗРІЗ ДЛЯ ВАГИ БЛЕНДУ — ЩОБ ОЗНАКА БУЛА ТАКА, ЯК У САБМІТІ (С83)

## ЗВІДКИ
`wsplit.py` показав, що вага бленду передбачувана: ознака `σ²/‖Δ‖²` корелює
з оракульним оптимумом на **+0.61…+0.85** проти НЕЗАЛЕЖНОГО піветалона (НЕГ
+0.06, шум −0.23), а попередній вирок «предиктора нема, ρ=0.28» був наслідком
бага: у `wpred.py` ОЗНАКА рахувалась зі знаменником `‖ours−m_deep‖²`, а
ЕТАЛОН — з `‖(1+ALPHA)·m_deep−ours‖²`. Через те, що `ours ≈ m_deep`, множник
`(1+ALPHA)=0.99249` міняє квадрат норми в **1.6…3.1×** — знаменники були
різними об'єктами.

## 🚨 ЧОГО БРАКУЄ ДЛЯ ВИРОКУ
У `wsplit.py` ознака усереднена по 16 жеребах. У САБМІТІ жереб ОДИН.
Для півподілу це вже виявилось критично (×1.0109 усер. → ×1.0044 з одного).
Тут зберігаю ВСЕ пожеребно, щоб рахувати будь-яке розбиття офлайн задарма:
    VO[n,d]  vout                       — чисельник
    DE[n,d]  ‖(1+ALPHA)m_deep − ours‖²  — знаменник
    NU[n,d]  ⟨a1,Δ⟩+⟨a2,Δ⟩              — для MSE(w) і для w_орак
    ZZ[n,d]  ⟨a1,a2⟩                    — вільний член MSE(w)
    OU[n,d]  ‖ours‖²  · MD[n,d] ‖m_deep‖²  — на пізніші ознаки
⟹ MSE(w) = Σ (ZZ + w·NU + w²·DE) точно, для будь-якої підмножини жеребів.

## ⛔ ПЛОМБА
Сума по всіх жеребах мусить відтворити `wsplit.npz` до 1e-12 (той самий
`seed = 77_000_000 + 733·n + d`, ті самі мережі, той самий порт).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/wvout.py [cuda:N] [NETS] [NDRAW]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, closure91, predict91,      # noqa: E402
                        ALPHA)
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 16
L_ = 32

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
NETS = len(FS)
VO, DE, NU, ZZ, OU, MD = (np.zeros((NETS, NDRAW)) for _ in range(6))
NP = np.zeros(NETS)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, 8, DEV, kdeep=L_)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        NP[n] = npair
        for d in range(NDRAW):
            _, _, _, (ours, m_deep, vout) = predict91(
                wc, npair, DEV, seed=77_000_000 + 733 * n + d, cl=cl,
                diag=True, raw=True)
            dl = (1.0 + ALPHA) * m_deep - ours
            a1, a2 = ours - T1, ours - T2
            VO[n, d] = vout
            DE[n, d] = float((dl * dl).sum())
            NU[n, d] = float((a1 * dl).sum()) + float((a2 * dl).sum())
            ZZ[n, d] = float((a1 * a2).sum())
            OU[n, d] = float((ours * ours).sum())
            MD[n, d] = float((m_deep * m_deep).sum())
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/wvout.npz',
         VO=VO, DE=DE, NU=NU, ZZ=ZZ, OU=OU, MD=MD, NP=NP)

# ⛔ ПЛОМБА проти `wsplit.npz`
try:
    S = np.load('/mnt/win_d/llm/White-Box/arena62/logs/wsplit.npz')
    for nm, mine, his in (('Z0', ZZ.sum(1), S['Z0']),
                          ('NUM', NU.sum(1), S['NUM']),
                          ('DEN', DE.sum(1), S['DEN'])):
        r = float(np.abs(mine - his).max() / max(np.abs(his).max(), 1e-300))
        print(f'⛔ ПЛОМБА {nm:4s} vs wsplit.npz: відн. {r:.2e} '
              f'{"✅" if r < 1e-10 else "🚨 РОЗІЙШЛОСЬ"}')
except FileNotFoundError:
    print('(wsplit.npz нема — пломбу пропущено)')
print(f'\n{NETS} мереж × {NDRAW} жеребів збережено пожеребно.')
print('📊 БАЛ 1.954e-07 (sub87) · sub92 ≈1.946e-07 — вимір діагностичний')
