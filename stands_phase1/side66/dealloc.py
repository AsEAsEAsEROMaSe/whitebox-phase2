"""ЩО САМЕ КОШТУЄ 3.3 мс «ЗВІЛЬНЕННЯ ЛОКАЛЬНИХ» (27% residual). (С73)

ЗВІДКИ. `phases.py` розклав residual сабміту пофазно (пломба Σфаз==R, Δ=4 мкс).
Найбільша фаза — 3.32 мс МІЖ останнім `mark` і виходом з контексту, при НУЛІ
операцій flopscope. Гіпотеза: деалокація локальних масивів `predict` на `return`.

🚨 ЦЕ ПЕРЕВІРКА ПРИЧИНИ, А НЕ НАЗИВАННЯ ЇЇ ([[feedback-measure-before-naming-a-cause]]).
ТЕСТ: якщо повернути локальні НАЗОВНІ (щоб вони пережили контекст), фаза мусить
впасти в нуль. Якщо не впаде — причина інша (GC / деструктори обгортки fnp).

КОНТРОЛІ
  ⚠️ той самий прогрів і те саме число повторів, що в `phases.py`;
  ✅ пломба: вихід у всіх варіантах РІВНО однаковий.
"""
import glob
import statistics as st
import sys
import time

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')

import numpy as np                                      # noqa: E402
import flopscope as flops                               # noqa: E402
import flopscope.numpy as fnp                           # noqa: E402
from whestbench import MLP                              # noqa: E402
from phases import predict_probed, Probe, B, LAM        # noqa: E402


def net():
    wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = set(int(v) for v in sd)
    f = [q for q in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(q)['seed']) in have][0]
    s0 = int(np.load(f)['seed'])
    j = int(np.where(sd == s0)[0][0])
    Wn = [fnp.array(np.asarray(wz[j][k], np.float32)) for k in range(32)]
    return MLP(width=256, depth=32, weights=Wn, seed=s0)


def probe_keep(mlp, keep):
    """Той самий `predict_probed`, але кадр функції або вмирає, або ні."""
    with flops.BudgetContext(flop_budget=B, quiet=True) as c:
        pr = Probe(c)
        if keep:
            # 🔑 трасування кадру: ловимо ЛОКАЛЬНІ й тримаємо їх живими
            box = []

            def tr(frame, ev, arg):
                if ev == 'return' and frame.f_code.co_name == 'predict_probed':
                    box.append(dict(frame.f_locals))
                return None
            sys.setprofile(None)
            old = sys.gettrace()
            sys.settrace(tr)
            frame_tr = True
        else:
            box, old, frame_tr = None, None, False
        r = predict_probed(mlp, B, pr)
        if frame_tr:
            sys.settrace(old)
        pr.mark('ХВІСТ (звільнення локальних)')
    return c, pr, r, box


def main(reps=9, warm=2):
    mlp = net()
    for _ in range(warm):
        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            predict_probed(mlp, B, Probe(c))

    # ── A: як є ──
    ta, tot_a = [], []
    ref = None
    for _ in range(reps):
        c, pr, r, _ = probe_keep(mlp, False)
        ta.append(pr.res['ХВІСТ (звільнення локальних)'])
        tot_a.append(c.summary_dict()['residual_wall_time_s'])
        ref = np.asarray(r) if ref is None else ref

    # ── B: локальні виживають (тримаємо їх поза контекстом) ──
    tb, tot_b, boxes = [], [], []
    for _ in range(reps):
        c, pr, r, box = probe_keep(mlp, True)
        tb.append(pr.res['ХВІСТ (звільнення локальних)'])
        tot_b.append(c.summary_dict()['residual_wall_time_s'])
        boxes.append(box)
        assert float(np.abs(np.asarray(r) - ref).max()) == 0.0, '🛑вихід змінився'
    boxes.clear()

    ma, mb = st.median(ta), st.median(tb)
    print(f'{reps} прогонів, медіана\n')
    print(f'{"варіант":42s} {"ХВІСТ, мс":>10s} {"R разом, мс":>12s}')
    print('-' * 68)
    print(f'{"A як є (локальні вмирають на return)":42s} {ma*1000:9.3f} '
          f'{st.median(tot_a)*1000:11.3f}')
    print(f'{"B локальні ВИЖИВАЮТЬ (трасування кадру)":42s} {mb*1000:9.3f} '
          f'{st.median(tot_b)*1000:11.3f}')
    print('-' * 68)
    print(f'\n🎯 хвіст A−B = {(ma-mb)*1000:.3f} мс  ⟹ '
          f'{"ЦЕ ДЕАЛОКАЦІЯ" if ma - mb > 0.5e-3 else "ПРИЧИНА ІНША"}')
    print('🚨 варіант B САМ ПО СОБІ дорожчий (трасування) — він лише ДІАГНОЗ, '
          'не рецепт.')

    # ── C: скільки коштує вивільнити ті самі байти вручну, поза контекстом ──
    z = [np.empty((12400, 155), np.float32) for _ in range(3)]
    for q in z:
        q[:] = 1.0
    t0 = time.perf_counter()
    del z
    print(f'\n📐 контрольна вартість free() трьох масивів 7.7 МБ: '
          f'{(time.perf_counter()-t0)*1e6:.0f} мкс')


if __name__ == '__main__':
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 9)
