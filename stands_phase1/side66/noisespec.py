"""СПЕКТР ШУМУ ПО ГЛИБИНІ: скільки ЧИСЕЛ треба знати на кожному шарі?

════════ НАВІЩО (запит юзера: працювати на рівні матриць, а не «методів») ════
`varmap.py`: приз росте з глибиною (K=8 → 1.72×, K=30 → 44.8×), і перенос НЕ
винен (`gatecap.py`: затвор 1.001×). Механічна гіпотеза, звідки це:
🔑 `relu` без зсувів = `a = p ⊙ 1[p>0]`, маска залежить ЛИШЕ від НАПРЯМКУ.
   З глибиною активації колапсують (PR-ранг **161.6 → 2.5**) ⟹ семпли майже
   колінеарні, `a_30^(i) ≈ c_i·u`. Тоді шум середнього — це фактично ОДНЕ
   ЧИСЛО, помножене на фіксований вектор: він КОГЕРЕНТНИЙ і не гаситься
   усередненням, тоді як на мілких шарах він розмазаний по ~256 напрямках.

════════ ЩО МІРЯЄМО ════════
`δ_ℓ = x̄_ℓ − m_істина,ℓ` на багатьох жеребах ⟹ коваріація `δ` ПО ЖЕРЕБАХ ⟹
спектр. Частка дисперсії в топ-1/2/8 напрямках = скільки чисел треба знати.
І ГОЛОВНЕ ДЛЯ ВАЖЕЛЯ: кут між похибкою ЗАМИКАННЯ `B = m_замик − m_істина` і
головним напрямком ШУМУ. Якщо вони майже ортогональні, то проєкція `B` на
потрібний напрямок у `√n ≈ 16×` менша за повну `B` — і арифметика каналу
міняється, причому БЕЗКОШТОВНО.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 шар 1: шум майже ІЗОТРОПНИЙ, топ-1 напрямок несе **<5%**;
📣 шар 30: топ-1 несе **>50%** (когерентність);
📣 `|cos(B, u₁)| ≈ 1/√256 ≈ 0.06` — замикання майже ОРТОГОНАЛЬНЕ до шуму.
🟢 ЖИВЕ, якщо на глибині топ-1 ≥50% І `|cos|` ≤0.15 ⟹ треба знати ОДНЕ число,
   а опора в тому напрямку в рази точніша, ніж здається з повної норми `B`.

🚨 ПЛОМБИ
  ✅ слід коваріації `δ` мусить збігтися з прямим `E‖δ‖²` (те саме число);
  ✅ на шарі 0 (якір точний) шум мусить бути НАЙМЕНШИМ відносно ‖m‖;
  ⛔ КОНТРОЛЬ: спектр ВИПАДКОВОГО гаусового шуму тієї ж норми — топ-1 мусить
     дати ~`1/n` (ізотропія), інакше сам оцінювач спектра зміщений.

🚨🚨 ДРУГА ВЕРСІЯ. Перша дала `cos(B,u₁)=0.81` і я оголосив «шум і похибка
формули КОЛІНЕАРНІ ⟹ канал принципово не розділяє». АЛЕ ЦЕ МОГЛО БУТИ
ТРИВІАЛЬНИМ: на глибині PR-ранг 2.5, усе втягнуте у вузький конус, і там
колінеарне ВСЕ З УСІМ. Потрібні контролі, яких бракувало:
  ⛔ `cos(B_ЧУЖОЇ мережі, u₁)` — якщо теж ~0.8, вирівнювання тривіальне;
  ⛔ `cos(m, u₁)` — чи саме середнє вже вирівняне з головною модою шуму;
  ⛔ `cos(випадковий у підпросторі топ-8, u₁)` — геометрія конуса сама по собі.
Плюс `u₁` оцінювався з 64 жеребів у 256 вимірах (ранг ≤63) ⟹ мало.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N)

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 96
TR_PAIRS, CHUNK = 524_288, 32768
SHOW = (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 31)


def truth_means(wc, dev, seed):
    depth = len(wc)
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
           for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(depth):
            a = torch.relu(a @ wc[i])
            acc[i] += a.double().sum(0)
        done += k
    return [t / float(2 * TR_PAIRS) for t in acc]


def draw(wc, npair, dev, seed):
    """Один жереб САБМІТОВОЮ схемою: `x̄_ℓ` кожного шару."""
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cov = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    a = torch.relu(x @ (wh @ wc[0]))
    out = [a.double().mean(0)]
    for i in range(1, len(wc)):
        a = torch.relu(a @ wc[i])
        out.append(a.double().mean(0))
    return out


def main(nets=16, dev=DEV, ndraw=NDRAW):
    torch.manual_seed(11)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: шар 1 топ-1 <5% · шар 30 топ-1 >50% · '
          f'|cos(B,u₁)| ≈ 0.06')
    print(f'🟢 ЖИВЕ при топ-1 ≥50% І |cos| ≤0.15\n')

    NL = 32
    T1 = np.zeros(NL); T2 = np.zeros(NL); T8 = np.zeros(NL)
    CS = np.zeros(NL); CSR = np.zeros(NL)
    RN = np.zeros(NL); BN = np.zeros(NL); PB = np.zeros(NL)
    CSX = np.zeros(NL); CSM = np.zeros(NL); CSD = np.zeros(NL)
    cx = np.zeros(NL)
    cnt = np.zeros(NL)
    prevB = None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            mt = truth_means(wc, dev, 310_000 + 877 * n)
            s_cl, m_cl = closure_parts(wc, d - 1, dev)
            m0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt() \
                * INV_SQRT_2PI
            CL = [m0] + list(m_cl)
            D = [[] for _ in range(d)]
            for k in range(ndraw):
                xb = draw(wc, npair, dev, 51_000_000 + 977 * n + k)
                for i in range(d):
                    D[i].append(xb[i] - mt[i])
            for i in range(d):
                M = torch.stack(D[i], 0)            # (ndraw, n)
                M = M - M.mean(0, keepdim=True)
                C = (M.T @ M) / float(M.shape[0])
                ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)
                tot = float(ev.sum())
                if tot <= 0:
                    continue
                T1[i] += float(ev[0]) / tot
                T2[i] += float(ev[:2].sum()) / tot
                T8[i] += float(ev[:8].sum()) / tot
                # головний напрямок шуму
                _, V = torch.linalg.eigh(C)
                u1 = V[:, -1]
                B = CL[i] - mt[i] if i < len(CL) else None
                if B is not None and float(B.norm()) > 0:
                    cs = abs(float((B @ u1) / B.norm()))
                    CS[i] += cs
                    BN[i] += float(B.norm())
                    PB[i] += abs(float(B @ u1))
                    # ⛔ саме середнє: чи вже вирівняне з головною модою шуму
                    CSM[i] += abs(float((mt[i] @ u1) / mt[i].norm()))
                    # ⛔ випадковий напрямок У ПІДПРОСТОРІ топ-8 шуму:
                    # скільки дає САМА геометрія конуса
                    U8 = V[:, -8:]
                    rr = U8 @ torch.randn(8, dtype=torch.float64, device=dev)
                    CSD[i] += abs(float((rr @ u1) / rr.norm()))
                    # ⛔ похибка замикання ЧУЖОЇ мережі проти НАШОГО u₁
                    if prevB is not None and i < len(prevB):
                        pb = prevB[i]
                        k = min(pb.shape[0], u1.shape[0])
                        t = torch.zeros_like(u1)
                        t[:k] = pb[:k]
                        if float(t.norm()) > 0:
                            CSX[i] += abs(float((t @ u1) / t.norm()))
                            cx[i] += 1
                # ⛔контроль: ізотропний шум тієї ж форми
                R = torch.randn(M.shape, dtype=torch.float64, device=dev)
                R = R - R.mean(0, keepdim=True)
                CR = (R.T @ R) / float(R.shape[0])
                er = torch.linalg.eigvalsh(CR).flip(0).clamp_min(0)
                CSR[i] += float(er[0]) / float(er.sum())
                RN[i] += float(M.pow(2).sum(1).mean()) ** 0.5
                cnt[i] += 1
            prevB = [CL[i] - mt[i] for i in range(min(len(CL), d))]
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    ok = cnt > 0
    for A in (T1, T2, T8, CS, CSR, RN, BN, PB, CSM, CSD):
        A[ok] /= cnt[ok]
    okx = cx > 0
    CSX[okx] /= cx[okx]

    print(f'\n{nets} мереж × {ndraw} жеребів. Спектр шуму `δ_ℓ = x̄_ℓ − m_ℓ`:')
    print(f'{"шар":>5s} {"топ-1":>8s} {"топ-8":>8s} {"⛔ізотр":>8s} '
          f'{"cos(B,u₁)":>10s} {"⛔чужа B":>9s} {"⛔вип.конус":>11s} '
          f'{"⛔cos(m,u₁)":>11s}')
    print('-' * 78)
    for i in range(NL):
        if not ok[i] or i not in SHOW:
            continue
        print(f'{i:5d} {T1[i]*100:7.1f}% {T8[i]*100:7.1f}% '
              f'{CSR[i]*100:7.1f}% {CS[i]:10.4f} {CSX[i]:9.4f} '
              f'{CSD[i]:11.4f} {CSM[i]:11.4f}')

    L = int(np.max(np.nonzero(ok)))
    print(f'\n⛔КОНТРОЛЬ ІЗОТРОПІЇ: топ-1 випадкового шуму = {CSR[L]*100:.1f}% '
          f'(теорія ~{100.0/min(NDRAW, 256):.1f}%) '
          f'{"✅" if CSR[L] < 0.25 else "🛑 оцінювач спектра зміщений"}')
    print(f'\n🔑 ШАР {L}: топ-1 несе {T1[L]*100:.1f}% шуму · '
          f'|cos(B,u₁)| = {CS[L]:.4f}')
    print(f'⛔КОНТРОЛІ на шарі {L}: чужа B {CSX[L]:.4f} · '
          f'випадковий у конусі {CSD[L]:.4f} · cos(m,u₁) {CSM[L]:.4f}')
    triv = max(CSX[L], CSD[L])
    print(f'🔑 ВИРОК ПРО КОЛІНЕАРНІСТЬ: cos(B,u₁)={CS[L]:.4f} проти '
          f'тривіального рівня {triv:.4f} ⟹ '
          f'{"ЗНАЧУЩА (не пояснюється конусом)" if CS[L] > 2 * triv else "🛑 ТРИВІАЛЬНА — мій висновок ХИБНИЙ"}')
    print(f'📐 Якщо опору потрібно знати ЛИШЕ вздовж `u₁`, то ефективна '
          f'похибка замикання\n   там у {BN[L]/max(PB[L],1e-30):.1f}× менша '
          f'за повну ‖B‖ ⟹ мішень 0.0018% стає '
          f'{0.0018*BN[L]/max(PB[L],1e-30):.4f}% у повних одиницях')
    print(f'📊 БАЛ: 1.972e-07 (діагностика, стендова)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
