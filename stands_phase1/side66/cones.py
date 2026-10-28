"""СТАТИСТИКА КОНУСІВ: чи є закон, за яким можна подавати семпли? (С73, ідея юзера)

════════ ГЕОМЕТРІЯ ════════
Мережа БЕЗ ЗСУВІВ ⟹ кожен із 7936 нейронів ділить простір гіперплощиною через
нуль ⟹ простір нарізаний на КОНУСИ з вершиною в нулі, і всередині конуса
мережа ТОЧНО ЛІНІЙНА: `f(z) = A_patt · z`. Патерн знаків визначає все.

════════ ЧОМУ ЦЕ НЕ ПУСТЕ ПИТАННЯ ════════
🔑 Наша `f = relu(...) ≥ 0` — СТРОГО ДОДАТНА. Для додатної функції ідеальний
importance sampling `q* ∝ f` дає РІВНО НУЛЬ дисперсії. Тобто стеля класу
«подавати семпли за законом» — нескінченна, і питання лише в тому, наскільки
дешево можна наблизити `q*`.

════════ ЩО МІРЯЄМО ════════
1. скільки РІЗНИХ конусів відвідує вибірка (очікую: всі різні);
2. розмір конуса — відстань до найближчої границі (чи є куди рухатись);
3. розподіл внеску семпла — чи є «важкі» конуси взагалі;
4. 🎯 ГОЛОВНЕ: чи передбачається внесок із ДЕШЕВИХ ознак (перші `k` шарів)?
   якщо так — importance sampling живий, і виграш рахується формулою
   `Var_MC/Var_IS`, де `q ∝ ĝ`;
5. стеля: скільки дав би ІДЕАЛЬНИЙ `q* ∝ f` (має бути ∞ для додатної `f`,
   тут — обмежена шумом, тому це ПЛОМБА приладу).

🚨 ПЛОМБИ
  ✅ однорідність: `f(2z) = 2f(z)` ⟹ вага мусить залежати ЛИШЕ від напрямку;
  ⛔ КОНТРОЛЬ: вага з ЧУЖОЇ мережі мусить дати ~1.00× (нуль виграшу);
  ⛔ вага = ВИПАДКОВЕ число тієї ж форми ⟹ мусить бути ≤1.00×.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_                      # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NS = 12400              # стільки ж точок, скільки в сабміті
KS = (1, 2, 4, 8, 16)   # після скількох шарів пробуємо вгадати внесок


def run(wc, x, upto=None):
    """Прохід; повертає активації після кожного шару (до `upto` включно)."""
    a = x
    outs = []
    for i, w in enumerate(wc):
        a = torch.relu(a @ w)
        outs.append(a)
        if upto is not None and i >= upto:
            break
    return outs


def main(nets=3, dev=DEV):
    torch.manual_seed(4242)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    agg = {'uniq': [], 'act': [], 'marg': [], 'cv': [], 'skew': [],
           'top1': [], 'top10': [], 'ideal': [], 'rnd': [], 'foreign': [],
           **{f'k{k}': [] for k in KS}}
    prev_w = None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            g = torch.Generator(device=dev).manual_seed(700 + n)
            z = torch.randn(NS // 2, W_, device=dev, generator=g)
            x = torch.cat([z, -z], 0)

            acts = run(wc, x)
            # ── 1) патерн і його унікальність ──
            patt = torch.cat([(a > 0) for a in acts[:-1]], 1)      # NS × 7936
            key = (patt.double() @ torch.randn(patt.shape[1], 4, device=dev,
                                               generator=g,
                                               dtype=torch.float64))
            uniq = len(torch.unique(key.round(decimals=6), dim=0))
            agg['uniq'].append(100.0 * uniq / NS)
            agg['act'].append(float(patt.double().mean()) * 100)

            # ── 2) розмір конуса: відстань до найближчої границі ──
            #    для нейрона з преактивацією h і рядком ваг: |h|/‖grad‖.
            #    беремо ПЕРШИЙ шар (там grad = стовпець W, точно) як оцінку.
            h1 = x @ wc[0]
            gn = wc[0].norm(dim=0).clamp_min(1e-20)
            marg = (h1.abs() / gn).min(1).values / x.norm(dim=1)
            agg['marg'].append(float(marg.median()))

            # ── 3) внесок семпла ──
            fv = acts[-1].double()                     # NS × out
            w_true = fv.sum(1)                         # скаляр-внесок
            agg['cv'].append(float(w_true.std() / w_true.mean()))
            agg['skew'].append(float(((w_true - w_true.mean()) ** 3).mean()
                                     / w_true.std() ** 3))
            srt = w_true.sort(descending=True).values
            agg['top1'].append(float(srt[:NS // 100].sum() / w_true.sum()) * 100)
            agg['top10'].append(float(srt[:NS // 10].sum() / w_true.sum()) * 100)

            # ── 4) виграш importance sampling з вагою `q ∝ ĝ` ──
            #    Var_IS/Var_MC = E[f²/ĝ]·E[ĝ]/E[f²]  (self-normalised, той самий n)
            def gain(gh):
                gh = gh.clamp_min(1e-12)
                gh = gh / gh.mean()
                v_is = float((w_true ** 2 / gh).mean()) - float(w_true.mean()) ** 2
                v_mc = float((w_true ** 2).mean()) - float(w_true.mean()) ** 2
                return v_mc / max(v_is, 1e-30)

            agg['ideal'].append(gain(w_true))                    # ⚠️стеля/пломба
            agg['rnd'].append(gain(torch.rand(NS, device=dev,
                                              generator=g,
                                              dtype=torch.float64) + 0.5))
            for k in KS:
                if k < len(wc):
                    agg[f'k{k}'].append(gain(acts[k].double().sum(1)))
            if prev_w is not None:
                fo = run(prev_w, x, upto=7)[-1].double().sum(1)
                agg['foreign'].append(gain(fo))
            prev_w = wc
        print(f'  … {n+1}/{nets}', flush=True)

    m = np.median
    print(f'\n{nets} мереж × {NS} точок\n')
    print('1) КОНУСИ')
    print(f'   різних патернів: {m(agg["uniq"]):.2f}% від усіх точок '
          f'⟹ кожна точка у СВОЄМУ конусі')
    print(f'   активних нейронів: {m(agg["act"]):.1f}%')
    print(f'   відстань до найближчої границі (у частках ‖z‖): '
          f'{m(agg["marg"]):.2e}')
    print(f'   ⟹ щоб лишитись у конусі, крок має бути <{m(agg["marg"]):.1e}·‖z‖')
    print('\n2) РОЗПОДІЛ ВНЕСКУ СЕМПЛА')
    print(f'   коеф. варіації {m(agg["cv"]):.3f},  скіс {m(agg["skew"]):+.3f}')
    print(f'   верхній 1% точок несе {m(agg["top1"]):.2f}% суми  '
          f'(рівномірно було б 1%)')
    print(f'   верхні 10% несуть {m(agg["top10"]):.2f}%  (рівномірно 10%)')
    print('\n3) IMPORTANCE SAMPLING: у скільки разів падає дисперсія')
    print(f'   {"⚠️ІДЕАЛЬНА вага q∝f (стеля/пломба)":42s} {m(agg["ideal"]):8.3f}×')
    for k in KS:
        if agg[f'k{k}']:
            print(f'   {"вага з " + str(k) + " шарів (ціна " + f"{100*k/32:.0f}" + "% проходу)":42s} '
                  f'{m(agg[f"k{k}"]):8.3f}×')
    if agg['foreign']:
        print(f'   {"⛔вага з ЧУЖОЇ мережі (контроль)":42s} '
              f'{m(agg["foreign"]):8.3f}×')
    print(f'   {"⛔вага ВИПАДКОВА (контроль)":42s} {m(agg["rnd"]):8.3f}×')
    print('\n🔑 читати так: виграш мусить БИТИ ціну. Вага з `k` шарів коштує '
          '~k/32 зайвого\n   проходу ⟹ при 8 шарах треба >1.25×, при 16 — >1.5×.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
