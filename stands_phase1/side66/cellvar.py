"""РОЗКЛАД ДИСПЕРСІЇ ВСЕРЕДИНІ/МІЖ КОМІРКАМИ — гейт умовного МК (Рао-Блеквелл).

ЗВІДКИ (юзер, і він спіймав мою помилку). Я запропонував умовний МК «по одній
координаті при фіксованому `z_⊥`». Це ЗАКРИТО нашим власним виміром:
[[finding-superposition-dimension-is-high-univariate-part-is-linear]] каже, що
ВСЯ одновимірна частина лінійна (`U(ρ)=0.245ρ`, залишок 0.009), а лінійну
частину вже знімає антитетика — умовне очікування по одній координаті це рівно
одновимірний ANOVA-член, тобто 1.00× наперед.

ПРАВИЛЬНА форма для ReLU-мережі — обумовлювати на **КОМІРКУ**, тобто на патерн
активацій. Усередині комірки мережа ТОЧНО лінійна, тож `E[F|комірка]` є лінійна
функція від усічених гаусових моментів по політопу. Це справжня
Рао-Блеквелізація, і вона знімає рівно ВНУТРІШНЬОКОМІРКОВУ дисперсію.

🎯 ГЕЙТ ДО БУДЬ-ЯКОЇ РЕАЛІЗАЦІЇ (пропозиція юзера): розкласти
    `Var(F) = E[Var(F|комірка)] + Var(E[F|комірка])`.
Якщо більшість дисперсії МІЖ комірками — весь клас закривається одним числом.

ЯК МІРЯЮ (групувати семпли за патерном НЕМОЖЛИВО: у 256 вимірах із 8000
гіперплощин кожен семпл у своїй комірці). Замість групування — ПРЯМИЙ розмір
комірки: з точки `z` іду вздовж `k` ортонормованих напрямків `u` і геометричною
сіткою шукаю ПЕРШУ зміну патерну; це дає піврозмір `h_u`. Усередині комірки `F`
точно лінійна, тож внесок напрямку у внутрішню дисперсію ≤ `h_u²·‖J u‖²`
(межа Поповічу), а `J u` рахується протягуванням `u` з МАСКАМИ, взятими в `z`.

🔑 ЗАРАЗОМ — УСЯ КРИВА ЗА ГЛИБИНОЮ ОБУМОВЛЕННЯ. Для кожної точки сітки
запам'ятовую ПЕРШИЙ шар, де патерн розійшовся, тож одним прогоном виходить
`h` для обумовлення на патерн шарів 1..L при L = 1, 2, 4, 8, 16, 32. Грубіше
обумовлення = більші комірки = більше знімається, але за межами L мережа вже
не лінійна — крива показує, де компроміс.

КОНТРОЛІ
  * ✅ ПОЗ: МІЛКА мережа (2 шари) — гіперплощин мало, комірки великі ⟹ частка
    внутрішньої дисперсії мусить бути ВЕЛИКОЮ. Не так ⟹ стенд зламаний;
  * ✅ ЗВІРКА з `newman_1d.py`: там уздовж гаусової прямої нараховано 1384–1945
    зламів ⟹ типовий піврозмір комірки має вийти ~3/1600 ≈ 2e−3;
  * ⛔ НЕГ: `‖J u‖` з ЧУЖОЇ мережі — частка мусить утратити сенс (друкуємо
    для порівняння масштабу).
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                                  # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NBASE = 48
NDIR = 32
NGRID = 72
TMIN, TMAX = 1e-6, 10.0
LGRID = [1, 2, 4, 8, 16, 32]
NVAR = 32768


def first_diff_layer(wc, X, ref):
    """Для кожного рядка `X` — НОМЕР першого шару, де патерн розійшовся з `ref`
    (або depth+1, якщо збігається скрізь)."""
    fd = torch.full((X.shape[0],), len(wc) + 1, dtype=torch.long,
                    device=X.device)
    x = X
    for i, w in enumerate(wc):
        p = x @ w
        s = (p > 0)
        bad = (s != ref[i].unsqueeze(0)).any(1)
        fd = torch.minimum(fd, torch.where(bad, torch.full_like(fd, i + 1), fd))
        x = torch.relu(p)
    return fd


def ref_pattern(wc, z):
    x = z.unsqueeze(0)
    out = []
    for w in wc:
        p = x @ w
        out.append((p > 0).squeeze(0))
        x = torch.relu(p)
    return out


def jac_dirs(wc, z, U):
    """`J u` для всіх `u` в `U` (маски зафіксовані в `z`). Повертає ‖J u‖²."""
    x = z.unsqueeze(0)
    V = U.clone()
    for w in wc:
        p = x @ w
        m = (p > 0).to(p.dtype)
        V = (V @ w) * m
        x = p * m
    return (V * V).sum(1).double()


def total_var(wc, dev, seed, n=NVAR, chunk=8192):
    """tr Cov(F) — та сама величина, що бере MSE (сума по компонентах)."""
    s1 = s2 = None
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < n:
        k = min(chunk, n - done)
        x = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        for w in wc:
            x = torch.relu(x @ w)
        xd = x.double()
        s1 = xd.sum(0) if s1 is None else s1 + xd.sum(0)
        s2 = (xd * xd).sum(0) if s2 is None else s2 + (xd * xd).sum(0)
        done += k
    return float((s2 / n - (s1 / n) ** 2).clamp_min(0).sum())


def probe(wc, dev, seed, nbase=NBASE, ndir=NDIR, tag=''):
    g = torch.Generator(device=dev).manual_seed(int(seed))
    ts = torch.exp(torch.linspace(np.log(TMIN), np.log(TMAX), NGRID,
                                  device=dev))
    depth = len(wc)
    inner = {L: 0.0 for L in LGRID}
    hmed = {L: [] for L in LGRID}
    for b in range(nbase):
        z = torch.randn(W_, device=dev, generator=g)
        ref = ref_pattern(wc, z)
        U = torch.linalg.qr(torch.randn(W_, W_, device=dev,
                                        generator=g))[0][:, :ndir].T  # (k,256)
        jn = jac_dirs(wc, z, U)                       # ‖J u‖²
        # сітка: (k напрямків) × (NGRID кроків) × (±)
        pts = (z.unsqueeze(0).unsqueeze(0)
               + ts.view(1, -1, 1) * U.unsqueeze(1))
        pts = torch.cat([pts, z.unsqueeze(0).unsqueeze(0)
                         - ts.view(1, -1, 1) * U.unsqueeze(1)], 1)
        fd = first_diff_layer(wc, pts.reshape(-1, W_), ref).view(ndir, -1)
        for L in LGRID:
            if L > depth:
                continue
            ok = (fd > L)                              # патерн шарів 1..L цілий
            tt = torch.cat([ts, ts]).unsqueeze(0).expand(ndir, -1)
            h = torch.where(ok, tt, torch.zeros_like(tt))
            # ПЕРША зміна: беремо найбільше t, до якого ВСІ менші t цілі
            srt = torch.argsort(tt, dim=1)
            okk = torch.gather(ok, 1, srt)
            run = torch.cumprod(okk.long(), 1).bool()
            hh = torch.gather(h, 1, srt) * run
            hu = hh.max(1).values.double()
            inner[L] += float((hu * hu * jn).sum()) * (W_ / ndir)
            hmed[L].append(float(hu.median()))
    tv = total_var(wc, dev, seed + 1)
    print(f'\n{tag}  tr Cov(F) = {tv:.5g}, {nbase} точок × {ndir} напрямків')
    print(f'{"обумовлення на шари":>20s} {"медіана h":>11s} '
          f'{"ВНУТРІШНЯ частка":>17s}')
    for L in LGRID:
        if L > depth:
            continue
        fr = inner[L] / nbase / max(tv, 1e-300)
        print(f'{"1.." + str(L):>20s} {np.median(hmed[L]):11.2e} {fr:16.4f}'
              + ('  ← ТОЧНА лінійність лише тут' if L == depth else ''))
    return tv


def main(nets=3, dev=DEV):
    torch.manual_seed(31337)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    for q, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][r], np.float32), device=dev)
             for r in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            if q == 0:
                print('=' * 66)
                print('✅ ПОЗ.КОНТРОЛЬ — мережі з МАЛОЮ кількістю гіперплощин.')
                print('   4 нейрони на 256 вимірів ⟹ комірки величезні ⟹')
                print('   ВНУТРІШНЯ частка мусить бути ~1. Далі — свіп до 512.')
                print('=' * 66)
                gt = torch.Generator(device=dev).manual_seed(4242)
                for nn in (4, 16, 64):
                    Wt = [torch.randn(W_, nn, device=dev, generator=gt)
                          / np.sqrt(W_)]
                    probe(Wt, dev, 400 + nn, nbase=16,
                          tag=f'✅ОДИН шар, {nn} нейронів ({nn} гіперплощин)')
                probe(wc[:2], dev, 500, nbase=16,
                      tag='2 шари реальної мережі (512 гіперплощин)')
                print('\n' + '=' * 66)
                print('РЕАЛЬНА МЕРЕЖА, 32 шари')
                print('=' * 66)
            probe(wc, dev, 900 + 17 * q, tag=f'мережа {q + 1}')
        del W, wc
        torch.cuda.empty_cache()
    print('\n🔑 читати так: «ВНУТРІШНЯ частка» при обумовленні на ВСІ 32 шари — '
          'це стеля Рао-Блеквелізації по комірках (і то ЗАВИЩЕНА, бо межа '
          'Поповічу по описаному паралелепіпеду). Мала ⟹ вся дисперсія МІЖ '
          'комірками, і клас умовного МК закрито одним числом.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
